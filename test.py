import os

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import argparse
from tqdm import tqdm
from utils.dataset import get_loader
from utils.models.models import get_model
from utils.trainer import Trainer
from utils.saver import Saver
from utils import utils
import torch
import GPUtil
import re

# Graph visualization on browser
import socket
import threading
import matplotlib
matplotlib.use("WebAgg")
matplotlib.rcParams['webagg.address'] = '127.0.0.1'
matplotlib.rcParams['webagg.open_in_browser'] = False
matplotlib.rcParams['figure.max_open_warning'] = 0
import sys
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from matplotlib import pyplot as plt
import webbrowser

# Explainability M3d-Cam
import scipy
import PIL
import io
from medcam import medcam
from medcam.backends import base as medcam_backends_base
from captum.attr import LayerAttribution
#torch.backends.cudnn.enabled = False # if RNN explainability
#import numpy as np

# Explainability pytorch-gradcam-book
from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, AblationCAM, ScoreCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad, DeepFeatureFactorization
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

def parse():
    '''Returns args passed to the train.py script.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='batch size', default=4)
    parser.add_argument('--logdir', type=str, help='path for choosing best checkpoint')
    parser.add_argument('--cache_rate', type=float, help='fraction of dataset to be cached in RAM [0.0-1.0]', default=0.0)
    parser.add_argument('--start_tensorboard_server', type=bool, help='start tensorboard server', default=False)
    parser.add_argument('--tensorboard_port', type=int, help='if starting tensorboard server, port (if unavailable, try the next ones)', default=6006)
    parser.add_argument('--start_tornado_server', type=bool, help='start tornado server to view current inference plots', default=False)
    parser.add_argument('--tornado_port', type=int, help='if starting tornado server, port (if unavailable, try the next ones)', default=8800) # matplotlib web interface
    parser.add_argument('--saveLogs', type=bool, help='save detailed logs of prediction/scores', default=True)
    parser.add_argument('--enable_explainability', type=bool, help='enable explainability images', default=False)
    parser.add_argument('--explainability_mode', type=str, help='explainability method [medcam, pytorchgradcambook]', choices=['medcam', 'pytorchgradcambook'], default='medcam')

    parser.add_argument('--enable_cudaAMP', type=bool, help='enable CUDA amp', default=True)
    parser.add_argument('--device', type=str, help='device to use (cpu, cuda, cuda:[number], if distributed cuda:[number],cuda:[number])', default='cuda')
    parser.add_argument('--distributed', type=bool, help='enable distribuited trainining', default=False)
    parser.add_argument('--dist_url', type=str, help='if using distributed training, other process path (ex: "env://" if same none)', default='env://')

    args = parser.parse_args()
    return args


# disable printing when not in master process
import builtins as __builtin__
builtin_print = __builtin__.print
def print_mod(*args, **kwargs):
    force = kwargs.pop('force', False)
    if 'RANK' in os.environ:
        rank = int(os.environ["RANK"])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
    else:
        RuntimeError("No RANK found!")
    if (rank==0) or force:
        builtin_print(*args, **kwargs)


def main():
    # Load configuration
    args = parse()

    # Check logdir if is a dir
    if not os.path.isdir(args.logdir):
        raise EnvironmentError('logdir must be an existing dir.')
    
    # Check/Mod batch_size
    if args.enable_explainability:
        if args.distributed:
            raise RuntimeError("Please not use distribuited mode when explainability enabled for too many ram usage.")
        args.batch_size = 1 # mandatory 1 if explainability
    
    # Load hyperparameters
    args2 = Saver.load_hyperparams(args.logdir)
    args = vars(args)
    for key in args:
        if key in args2:
            del args2[key]
    args.update(args2)
    args = argparse.Namespace(**args)

    # Enable retrocompatibility
    ##

    # choose device
    if args.distributed:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ['WORLD_SIZE'])
            if args.device == 'cuda':
                args.gpu = int(os.environ['LOCAL_RANK'])
            elif re.match(r"^cuda:[0-9]{1,3}(,cuda:[0-9]{1,3})+$", args.device): # verifica se è una stringa del tipo cuda:***,cuda:***...
                raise RuntimeError("Instead of specify list of cuda device in args.device, please type 'export CUDA_VISIBLE_DEVICES=list_of_device_to_used_separate_by_comma' in terminal and then start this script using arg.device='cuda'.")
                list_devices = args.device.replace('cuda:','').split(',')
                #device_count = len(list_devices)
                args.gpu = int(list_devices[int(os.environ['LOCAL_RANK'])])
            else:
                raise RuntimeError("Invalid device: "+args.device)
        elif 'SLURM_PROCID' in os.environ:
            args.rank = int(os.environ['SLURM_PROCID'])
            if args.device == 'cuda':
                args.gpu = args.rank % torch.cuda.device_count()
            elif re.match(r"^cuda:[0-9]{1,3}(,cuda:[0-9]{1,3})+$", args.device): # verifica se è una stringa del tipo cuda:***,cuda:***...
                raise RuntimeError("Instead of specify list of cuda device in args.device, please type 'export CUDA_VISIBLE_DEVICES=list_of_device_to_used_separate_by_comma' in terminal and then start this script using arg.device='cuda'.")
                list_devices = args.device.replace('cuda:','').split(',')
                #device_count = len(list_devices)
                args.gpu = int(list_devices[args.rank])
            else:
                raise RuntimeError("Invalid device: "+args.device)
        else:
            raise RuntimeError("Can't use distributed mode! Check if you don't run correct command: 'torchrun --master_addr=localhost --nproc_per_node=NUMBER_GPUS train.py'")
        print('| distributed init (rank {}, gpu {}): {}'.format(args.rank, args.gpu, args.dist_url), flush=True)
        torch.cuda.set_device(args.gpu)
        args.dist_backend = 'nccl' # gloo, nccl
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
        device = torch.device(args.gpu)
        # disable printing when not in master process
        #if args.rank != 0:
        #    __builtin__.print = no_print
    else:
        if args.device == 'cuda': # choose the most free gpu
            #mem = [(torch.cuda.memory_allocated(i)+torch.cuda.memory_reserved(i)) for i in range(torch.cuda.device_count())]
            mem = [gpu.memoryUtil for gpu in GPUtil.getGPUs()]
            args.device = 'cuda:' + str(mem.index(min(mem)))
        device = torch.device(args.device)
        print('Using device', args.device)

    # Print hyperparameters
    if (not args.distributed) or (args.distributed and (args.rank==0)):
        print("---Configs/Hyperparams---")
        for key in vars(args):
            print(key+":", vars(args)[key])
        print("-------------------------")

    # Dataset e Loader
    print("Dataset: balanced nested cross-validation use fold (test-set) " + str(args.num_fold) + " and inner_loop (validation-set) " + str(args.inner_loop) + ".")
    loaders, samplers, loss_weights = get_loader(args)
    del loaders['train']
    del loaders['validation']

    # Model
    model = get_model(num_classes=args.num_classes,
                        model_name=args.model,
                        use_rdm=args.use_rdm1,
                        use_rdm2=args.use_rdm2,
                        use_rdm3=args.use_rdm3,
                        use_rdmmti=args.use_rdmmti1,
                        use_rdmmti2=args.use_rdmmti2,
                        use_rdmmti3=args.use_rdmmti3)
    checkpoint, epoch = Saver.load_model(args.logdir, return_epoch=True)
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)

    # Enable explainability on model
    if args.enable_explainability:
        if args.explainability_mode == 'medcam':
            # Modify _BaseWrapper.forward() functin in /site-packages/medcam/backends/base.py to work with model's outputs
            if args.enable_multibranch_ffr or args.enable_multibranch_ifr:
                def forward_modding(self, batch):
                    """Calls the forward() of the model."""
                    self.model.zero_grad()
                    outputs = self.model.model_forward(batch)
                    self.logits = outputs[0]
                    self._extract_metadata(batch, self.logits)
                    self._set_postprocessor_and_label(self.logits)
                    self.remove_hook(forward=True, backward=False)
                    return outputs
                medcam_backends_base._BaseWrapper.forward = forward_modding
            # Inject model to get attention maps
            #print(medcam.get_layers())
            model = medcam.inject(model, backend='gcampp', save_maps=False, layer='auto')# layer4 # layer='auto'/'full'
        elif args.explainability_mode == 'pytorchgradcambook':
            #def find_layer_predicate_recursive(model, prefix=''):
            #    for name, layer in model._modules.items():
            #        tmp=prefix+'.'+name
            #        print(tmp)
            #        find_layer_predicate_recursive(layer, tmp)
            #find_layer_predicate_recursive(model)
            cam = GradCAM(model=model, target_layers=[model.avgpool[1].layers[-1]], use_cuda=(args.device!='cpu'), reshape_transform=None)
    
    # Enable model distribuited if it is
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model:', args.model, '(number of params:', n_parameters, ')')
    
    if args.enable_cudaAMP:
        # Creates GradScaler for CUDA AMP
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if (not args.distributed) or (args.distributed and (args.rank==0)):
        # TensorBoard Daemon
        if args.start_tensorboard_server:
            tensorboard_port = args.tensorboard_port
            i = 0
            while(True):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    i += 1
                    if s.connect_ex(('localhost', tensorboard_port)) == 0: # check if port is busy
                        tensorboard_port = tensorboard_port + 1
                    else:
                        break
                    if i > 100:
                        raise RuntimeError('Tensorboard: can not find free port at +100 from your chosen port!')
            t = threading.Thread(target=lambda: os.system('tensorboard --logdir=' + str(args.logdir) + ' --port=' + str(tensorboard_port)))
            t.start()
            webbrowser.open('http://localhost:' + str(tensorboard_port) + '/', new=1)

        # Setup server per matplotlib
        if args.start_tornado_server:
            tornado_port = args.tornado_port
            i = 0
            while(True):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    i += 1
                    if s.connect_ex(('localhost', tornado_port)) == 0: # check if port is busy
                        tornado_port = tornado_port + 1
                    else:
                        break
                    if i > 100:
                        raise RuntimeError('Tornado (matplotlib web interface): can not find free port at +100 from your chosen port!')
            matplotlib.rcParams['webagg.port'] = tornado_port

    tot_predicted_labels_last = {split:{} for split in loaders}
    for split in loaders:
        if args.distributed:
            samplers[split].set_epoch(0)

        data_loader = loaders[split]

        epoch_metrics = {}
        tot_true_labels = []
        tot_predicted_labels = []
        tot_predicted_scores = []
        tot_labels_name = []
        tot_sample_paths = []
        for batch in tqdm(data_loader, desc=f'{split}'):
            epoch_metrics = {}
            tot_true_labels = []
            tot_predicted_labels = []
            tot_predicted_scores = []
            tot_labels_name = []
            tot_sample_paths = []
            for i_batch, batch in enumerate(tqdm(data_loader, desc=f'{split}, {epoch}/{args.epochs}')):
                # Retrieve batch data
                sample_path = batch['sample_path']
                
                if args.load_embeddings:
                    embedding = batch['embedding']
                else:
                    if args.use_rdm1:
                        rdm1 = batch['rdm1']
                    if args.use_rdm2:
                        rdm2 = batch['rdm2']
                    if args.use_rdm3:
                        rdm3 = batch['rdm3']
                    if args.use_rdmmti1:
                        rdmmti1 = batch['rdmmti1']
                    if args.use_rdmmti2:
                        rdmmti2 = batch['rdmmti2']
                    if args.use_rdmmti3:
                        rdmmti3 = batch['rdmmti3']
                
                labels = batch['label']
                
                labels_name = batch['label_name']
                
                tot_true_labels.extend(labels.tolist())
                tot_labels_name.extend(labels_name.copy())

                # Move data to device
                if args.load_embeddings:
                    embedding = embedding.to(device)
                else:
                    if args.use_rdm1:
                        rdm1 = rdm1.to(device)
                    if args.use_rdm2:
                        rdm2 = rdm2.to(device)
                    if args.use_rdm3:
                        rdm3 = rdm3.to(device)
                    if args.use_rdmmti1:
                        rdmmti1 = rdmmti1.to(device)
                    if args.use_rdmmti2:
                        rdmmti2 = rdmmti2.to(device)
                    if args.use_rdmmti3:
                        rdmmti3 = rdmmti3.to(device)
                
                labels = labels.to(device)

                # Forward batch
                if args.load_embeddings:
                    input_datas = embedding
                else:
                    input_datas = []
                    if args.use_rdm1:
                        input_datas.append(rdm1)
                    if args.use_rdm2:
                        input_datas.append(rdm2)
                    if args.use_rdm3:
                        input_datas.append(rdm3)
                    if args.use_rdmmti1:
                        input_datas.append(rdmmti1)
                    if args.use_rdmmti2:
                        input_datas.append(rdmmti2)
                    if args.use_rdmmti3:
                        input_datas.append(rdmmti3)
            
            returned_values = Trainer.forward_batch_testing(task=args.task,
                                                            load_embeddings=args.load_embeddings,
                                                            net=model,
                                                            input_datas=input_datas,
                                                            labels=labels,
                                                            class_weights=loss_weights.to(device),
                                                            scaler=scaler)
            predicted_labels, predicted_scores = returned_values
            
            tot_predicted_labels.extend(predicted_labels.tolist())
            tot_predicted_scores.extend(predicted_scores.tolist())
            
            if args.enable_explainability:
                if args.explainability_mode == 'medcam':
                    interpolate_dims = rdm1.shape[2:]

                    if args.distributed:
                        layer_attribution = model.module.get_attention_map()
                    else:
                        #layer_attribution = np.expand_dims(model.get_attention_map(),0)
                        layer_attribution = model.get_attention_map()
                    upsamp_attr_lgc = LayerAttribution.interpolate(torch.from_numpy(layer_attribution), interpolate_dims)
                        
                    upsamp_attr_lgc = upsamp_attr_lgc.cpu().detach().numpy()
                    
                    if args.dataset3d and (not args.dataset2d):
                        tot_images_3d_gradient.extend(upsamp_attr_lgc.tolist())
                    elif (not args.dataset3d) and args.dataset2d:
                        tot_images_gradient.extend(upsamp_attr_lgc.tolist())
                    else:
                        raise NotImplementedError()
                    
                elif args.explainability_mode == 'pytorchgradcambook':
                    input_tensor = rdm1.clone().detach().requires_grad_(True)
                    
                    # targets = specify the target to generate the Class Activation Maps
                    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(1)], aug_smooth=True, eigen_smooth=True)
                    grayscale_cam = grayscale_cam.cpu().detach().numpy()
                    
                    if args.dataset3d and (not args.dataset2d):
                        tot_images_3d_gradient.extend(grayscale_cam.tolist())
                    elif (not args.dataset3d) and args.dataset2d:
                        tot_images_gradient.extend(grayscale_cam.tolist())
                    else:
                        raise NotImplementedError()
        
        if args.distributed:
            torch.distributed.barrier()

            if args.enable_explainability:
                if args.dataset3d:
                    tot_images_3d_output = [None for _ in range(args.world_size)]
                    tot_images_3d_gradient_output = [None for _ in range(args.world_size)]
                if args.dataset2d:
                    tot_images_output = [None for _ in range(args.world_size)]
                    tot_images_gradient_output = [None for _ in range(args.world_size)]
            tot_true_labels_output = [None for _ in range(args.world_size)]
            tot_predicted_labels_output = [None for _ in range(args.world_size)]
            tot_predicted_scores_output = [None for _ in range(args.world_size)]
            tot_labels_name_output = [None for _ in range(args.world_size)]
            tot_sample_paths_output = [None for _ in range(args.world_size)]

            torch.distributed.all_gather_object(tot_true_labels_output, tot_true_labels)
            torch.distributed.all_gather_object(tot_predicted_labels_output, tot_predicted_labels)
            torch.distributed.all_gather_object(tot_predicted_scores_output, tot_predicted_scores)
            torch.distributed.all_gather_object(tot_labels_name_output, tot_labels_name)
            torch.distributed.all_gather_object(tot_sample_paths_output, tot_sample_paths)

            if args.enable_explainability:
                print("Gathering volumes...")
                if args.dataset3d:
                    torch.distributed.all_gather_object(tot_images_3d_output, tot_images_3d)
                if args.dataset2d:
                    torch.distributed.all_gather_object(tot_images_output, tot_images)
                print("Gathering volume's gradients...")
                if args.dataset3d:
                    torch.distributed.all_gather_object(tot_images_3d_gradient_output, tot_images_3d_gradient)
                if args.dataset2d:
                    torch.distributed.all_gather_object(tot_images_gradient_output, tot_images_gradient)

            if args.enable_explainability:
                if args.dataset3d:
                    tot_images_3d = []
                    tot_images_3d_gradient = []
                if args.dataset2d:
                    tot_images = []
                    tot_images_gradient = []
            tot_true_labels = []
            tot_predicted_labels = []
            tot_predicted_scores = []
            tot_labels_name = []
            tot_sample_paths = []
            for i in range(args.world_size):
                tot_true_labels.extend(tot_true_labels_output[i].copy())
                tot_predicted_labels.extend(tot_predicted_labels_output[i].copy())
                tot_predicted_scores.extend(tot_predicted_scores_output[i].copy())
                tot_labels_name.extend(tot_labels_name_output[i].copy())
                tot_sample_paths.extend(tot_sample_paths_output[i].copy())
        
        if (not args.distributed) or (args.distributed and (args.rank==0)):
            # Accuracy Balanced classification
            accuracy_balanced = utils.calc_accuracy_balanced_classification(tot_true_labels, tot_predicted_labels, num_classes=args.num_classes)
            print(split, epoch, "epoch - Accuracy Balanced:", accuracy_balanced)
            
            # Accuracy classification
            accuracy = utils.calc_accuracy_classification(tot_true_labels, tot_predicted_labels, num_classes=args.num_classes)
            print(split, epoch, "epoch - Accuracy:", accuracy)

            # Precision
            precision = utils.calc_precision(tot_true_labels, tot_predicted_labels, num_classes=args.num_classes)
            print(split, epoch, "epoch - Precision:", precision.mean() if len(precision) > 0 else 0.0)

            # Recall
            recall = utils.calc_recall(tot_true_labels, tot_predicted_labels, num_classes=args.num_classes)
            print(split, epoch, "epoch - Recall:", recall.mean() if len(recall) > 0 else 0.0)

            # Specificity
            specificity = utils.calc_specificity(tot_true_labels, tot_predicted_labels, num_classes=args.num_classes)
            print(split, epoch, "epoch - Specificity:", specificity.mean() if len(specificity) > 0 else 0.0)

            # F1 Score
            f1score = utils.calc_f1(tot_true_labels, tot_predicted_labels, num_classes=args.num_classes)
            print(split, epoch, "epoch - F1 Score:", f1score)

            # Prediction Agreement Rate: same-sample evaluation agreement between current and previous epoch
            predictionAgreementRate, tot_predicted_labels_last[split] = utils.calc_predictionAgreementRate(tot_predicted_labels, tot_predicted_labels_last[split], tot_sample_paths)
            print(split, epoch, "epoch - Prediction Agreement Rate", predictionAgreementRate)

            if args.start_tornado_server:
                # Confusion Matrix
                confusionMatrix_array = utils.calc_confusionMatrix(tot_true_labels, tot_predicted_labels, num_classes=args.num_classes)
                confusionMatrix_figure, confusionMatrix_image = utils.plot_confusionMatrix(confusionMatrix_array, list(range(args.num_classes)), "Confusion Matrix"+split)
                utils.show_images(split + " " + str(epoch) + " epoch - Confusion Matrix", confusionMatrix_image)

            # Print logs of error
            dict_other_info = {}
            Saver.printLogsError(args.logdir/f'{split}_logs_error_{epoch:05d}', tot_true_labels, tot_predicted_labels, tot_predicted_scores, dict_other_info, split, epoch)
        
            # Save logs
            if args.saveLogs:
                Saver.saveLogs(args.logdir/f'{split}_logs_{epoch:05d}', tot_true_labels, tot_predicted_labels, tot_predicted_scores, dict_other_info, split, epoch)
            
            # Plot GradCAM
            if args.enable_explainability:
                print("Exporting explainability...")
                
                if not os.path.exists(args.logdir + '/export_fold' + str(args.num_fold)):
                    os.makedirs(args.logdir + '/export_fold' + str(args.num_fold))
                
                if args.dataset3d and (not args.dataset2d):
                    len_tot_images = len(tot_images_3d)
                elif (not args.dataset3d) and args.dataset2d:
                    len_tot_images = len(tot_images)
                else:
                    raise NotImplementedError()
                
                for i2 in tqdm(range(len_tot_images), desc='Explainability'):
                    if args.dataset3d and (not args.dataset2d):
                        tot_images_3d[i2] = torch.tensor(tot_images_3d[i2])
                        tot_images_3d_gradient[i2] = torch.tensor(tot_images_3d_gradient[i2])
                        imgs=[]
                        plt.clf()
                        for i in range(tot_images_3d[i2].shape[1]):
                            if args.explainability_mode == 'medcam':
                                plt.imshow(tot_images_3d[i2][0,i,:,:].cpu().squeeze().numpy(), cmap='gray')
                                plt.imshow(scipy.ndimage.gaussian_filter(tot_images_3d_gradient[i2][0,i,:,:], sigma=10), interpolation='nearest', alpha=0.25)
                            elif args.explainability_mode == 'pytorchgradcambook':
                                plt.imshow(show_cam_on_image(tot_images_3d[i2][0,i,:,:].cpu().squeeze().numpy(), tot_images_3d_gradient[i2][0,i,:,:], use_rgb=True, colormap=cv2.COLORMAP_JET, image_weight=0.5))
                            plt.axis('off')
                            buf = io.BytesIO()
                            plt.savefig(buf, format='jpeg')
                            buf.seek(0)
                            image = PIL.Image.open(buf)
                            imgs.append(image)
                            plt.clf()
                            #plt.show()
                        utils.saveGridImages(args.logdir + '/export_fold' + str(args.num_fold) + '/' + '_'.join(tot_sample_paths[i2].replace('\\', '/').split('/')[-3:])[:-4], imgs, n_colonne=8)
                    elif (not args.dataset3d) and args.dataset2d:
                        tot_images[i2] = torch.tensor(tot_images[i2])
                        tot_images_gradient[i2] = torch.tensor(tot_images_gradient[i2])
                        if args.explainability_mode == 'medcam':
                            plt.imshow(tot_images[i2][0,:,:].cpu().squeeze().numpy(), cmap='gray')
                            plt.imshow(scipy.ndimage.gaussian_filter(tot_images_gradient[i2][0,:,:], sigma=10), interpolation='nearest', alpha=0.25)
                        elif args.explainability_mode == 'pytorchgradcambook':
                            plt.imshow(show_cam_on_image(tot_images[i2][0,:,:].cpu().squeeze().numpy(), tot_images_gradient[i2][0,:,:], use_rgb=True, colormap=cv2.COLORMAP_JET, image_weight=0.5))
                        plt.axis('off')
                        plt.savefig(args.logdir + '/export_fold' + str(args.num_fold) + '/' + '_'.join(tot_sample_paths[i2].replace('\\', '/').split('/')[-3:])[:-4] + '.jpg', format='jpeg')
                        plt.clf()
                        #plt.show()
                    else:
                        raise NotImplementedError()

    if args.distributed:
        torch.distributed.barrier()
    
    if (not args.distributed) or (args.distributed and (args.rank==0)):
        if args.start_tornado_server:
            # Show graph on browser
            webbrowser.open('http://127.0.0.1:' + str(tornado_port) + '/', new=1)

            # Start Tornado server
            plt.show()

    if args.distributed:
        torch.distributed.destroy_process_group()
        

if __name__ == '__main__':
    main()