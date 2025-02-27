from multiprocessing.sharedctypes import Value
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import argparse
from pathlib import Path
from tqdm import tqdm
from utils.dataset import get_loader
from utils.models.models import get_model
from utils.trainer import Trainer
from utils.saver import Saver
import glob
from utils import utils
import torch
import GPUtil
import re
import numpy as np

def parse():
    '''Returns args passed to the train.py script.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=Path, help='dataset folder path')
    parser.add_argument('--split_path', type=Path, help='pt dataset metadata')
    parser.add_argument('--load_embeddings', type=int, help='if classification with backbone freezed, you can load embeddings instead of mp4', default=0)
    parser.add_argument('--video_type', type=str, help='video type (mp4 or jpeg)', default='mp4')
    parser.add_argument('--multiCrop', type=int, help='reduce video dims (only frames)', choices=[0,1], default=0)
    parser.add_argument('--multiCropMode', type=str, help='if multicrop, how to crop (random or consecutives)', default='random')
    parser.add_argument('--frame_per_crop', type=int, help='if multicrop, number of frame per crop', default=100)
    parser.add_argument('--number_crop', type=int, help='if multicrop, number of group of frames', default=10)
    parser.add_argument('--resize', type=int, help='if use_video, (H,W) resize dimensions (use -1 to save proportions)', nargs=3, default=[64,128,1024])
    parser.add_argument('--mean', type=float, help='(ch1,ch2,ch3) normalization mean', nargs=3, default=[0.0])
    parser.add_argument('--std', type=float, help='(ch1,ch2,ch3) normalization standard deviation', nargs=3, default=[1.0])
    parser.add_argument('--num_fold', type=int, help='test fold for nested cross-validation', default=0)
    parser.add_argument('--inner_loop', type=int, help='validation fold for nested cross-validation', default=0)

    parser.add_argument('--task', type=str, help='task (reconstruction, classification)', default='classification')

    parser.add_argument('--model', type=str, help='model (autoencoder, autoencoder_noSkip, endToEnd, endToEnd_noSkip, endToEnd_fast, endToEnd_noSkip_fast, endToEnd_finetuning, endToEnd_noSkip_finetuning)', default='endToEnd_Resnet')

    parser.add_argument('--num_classes', type=int, help='number of classes as array of dim 1', default=126)
    parser.add_argument('--use_rdm1', type=int, help='use range doppler map from rx1 in the model', choices=[0,1], default=1)
    parser.add_argument('--use_rdm2', type=int, help='use range doppler map from rx2 in the model', choices=[0,1], default=1)
    parser.add_argument('--use_rdm3', type=int, help='use range doppler map from rx3 in the model', choices=[0,1], default=1)
    parser.add_argument('--use_rdmmti1', type=int, help='use range doppler maps with mti from rx1 in the model', choices=[0,1], default=1)
    parser.add_argument('--use_rdmmti2', type=int, help='use range doppler maps with mti from rx2 in the model', choices=[0,1], default=1)
    parser.add_argument('--use_rdmmti3', type=int, help='use range doppler maps with mti from rx3 in the model', choices=[0,1], default=1)

    parser.add_argument('--gradient_clipping_value', type=int, help='gradient clipping value', default=0)
    parser.add_argument('--optimizer', type=str, help='optimizer (SGD, Adam, AdamW, RMSprop, LBFGS)', choices=['SGD', 'Adam', 'AdamW', 'RMSprop', 'LBFGS'], default='AdamW')
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=1e-4)
    parser.add_argument('--weight_decay', type=float, help='L2 regularization weight', default=1e-3) # 5e-4 if Adam, 1e-3 if AdamW o LBFGS
    parser.add_argument('--enable_scheduler', type=int, help='enable learning rate scheduler', choices=[0,1], default=0)
    parser.add_argument('--scheduler_factor', type=float, help='if using scheduler, factor of increment/redution', default=8e-2)
    parser.add_argument('--scheduler_threshold', type=float, help='if using scheduler, threshold for learning rate update', default=1e-2)
    parser.add_argument('--scheduler_patience', type=int, help='if using scheduler, number of epochs before updating the learning rate', default=5)

    parser.add_argument('--batch_size', type=int, help='batch size', default=16)
    parser.add_argument('--epochs', type=int, help='number of training epochs', default=10000)
    parser.add_argument('--experiment', type=str, help='experiment name (in None, default is timestamp_modelname)', default=None)
    parser.add_argument('--logdir', type=str, help='log directory path', default='./logs')
    parser.add_argument('--start_tensorboard_server', type=int, help='start tensorboard server', choices=[0,1], default=0)
    parser.add_argument('--tensorboard_port', type=int, help='if starting tensorboard server, port (if unavailable, try the next ones)', default=6006)
    parser.add_argument('--saveLogsError', type=int, help='save detailed logs of error prediction/scores', default=1)
    parser.add_argument('--saveLogs', type=int, help='save detailed logs of prediction/scores', default=1)
    parser.add_argument('--ckpt_every', type=int, help='checkpoint saving frequenct (in epochs); -1 saves only best-validation and best-test checkpoints', default=1)
    parser.add_argument('--resume', type=str, help='if not None, checkpoint path to resume', default=None)
    parser.add_argument('--save_array_file', type=int, help='save all plots as array', default=0)
    parser.add_argument('--save_image_file', type=int, help='save all plots as image', default=0)

    parser.add_argument('--enable_cudaAMP', type=int, help='enable CUDA amp', choices=[0,1], default=1)
    parser.add_argument('--device', type=str, help='device to use (cpu, cuda, cuda:[number], if distributed cuda:[number],cuda:[number])', default='cuda')
    parser.add_argument('--distributed', type=int, help='enable distribuited trainining', choices=[0,1], default=0)
    parser.add_argument('--dist_url', type=str, help='if using distributed training, other process path (ex: "env://" if same none)', default='env://')

    args = parser.parse_args()

    # Convert boolean (as integer) args to boolean type
    if args.load_embeddings == 0:
        args.load_embeddings = False
    else:
        args.load_embeddings = True
    if args.load_embeddings and "_fast" not in args.model:
        raise ValueError("If you want to load embeddings, you must use a model with '_fast' suffix.")
    if args.use_rdm1 == 0:
        args.use_rdm1 = False
    else:
        args.use_rdm1 = True
    if args.use_rdm2 == 0:
        args.use_rdm2 = False
    else:
        args.use_rdm2 = True
    if args.use_rdm3 == 0:
        args.use_rdm3 = False
    else:
        args.use_rdm3 = True
    if args.use_rdmmti1 == 0:
        args.use_rdmmti1 = False
    else:
        args.use_rdmmti1 = True
    if args.use_rdmmti2 == 0:
        args.use_rdmmti2 = False
    else:
        args.use_rdmmti2 = True
    if args.use_rdmmti3 == 0:
        args.use_rdmmti3 = False
    else:
        args.use_rdmmti3 = True
    if args.multiCrop == 0:
        args.multiCrop = False
    else:
        args.multiCrop = True
    if args.enable_scheduler == 0:
        args.enable_scheduler = False
    else:
        args.enable_scheduler = True
    if args.start_tensorboard_server == 0:
        args.start_tensorboard_server = False
    else:
        args.start_tensorboard_server = True
    if args.saveLogsError == 0:
        args.saveLogsError = False
    else:
        args.saveLogsError = True
    if args.saveLogs == 0:
        args.saveLogs = False
    else:
        args.saveLogs = True
    if args.save_array_file == 0:
        args.save_array_file = False
    else:
        args.save_array_file = True
    if args.save_image_file == 0:
        args.save_image_file = False
    else:
        args.save_image_file = True
    if args.enable_cudaAMP == 0:
        args.enable_cudaAMP = False
    else:
        args.enable_cudaAMP = True
    if args.distributed == 0:
        args.distributed = False
    else:
        args.distributed = True

    # Generate experiment tags if not defined
    if args.experiment == None:
        args.experiment = args.model
    
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
def no_print(*args, **kwargs):
    pass


def main():
    args = parse()

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
        args.dist_backend = 'gloo' # gloo, nccl
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
        device = torch.device(args.gpu)
        # disable printing when not in master process
        #if args.rank != 0:
        #    __builtin__.print = no_print
    else:
        if args.device == 'cuda': # choose the most free gpu
            #mem = [(torch.cuda.memory_allocated(i)+torch.cuda.memory_reserved(i)) for i in range(torch.cuda.device_count())]
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                cuda_visible_devices =  [int(index) for index in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
            else:
                cuda_visible_devices = list(range(len(GPUtil.getGPUs())))
            gpu_list = [gpu for gpu in GPUtil.getGPUs() if gpu.id in cuda_visible_devices]
            mem = [gpu.memoryUtil for gpu in gpu_list]
            args.device = 'cuda:' + str(mem.index(min(mem)))
        device = torch.device(args.device)
        print('Using device', args.device)

    # Dataset e Loader
    print("Dataset: balanced nested cross-validation use fold (test-set) " + str(args.num_fold) + " and inner_loop (validation-set) " + str(args.inner_loop) + ".")
    loaders, samplers, loss_weights = get_loader(args)

    # Model
    model = get_model(num_classes=args.num_classes,
                        model_name=args.model,
                        num_frames=args.resize[0] if args.resize[0] > 0 else None)
    if args.resume is not None:
        model.load_state_dict(Saver.load_model(args.resume), strict=True)
    model.to(device)

    # Enable model distribuited if it is
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model:', args.model, '(number of params:', n_parameters, ')')

    # Create optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params=model_without_ddp.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params=model_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params=model_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(params=model_without_ddp.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'LBFGS':
        optimizer = torch.optim.LBFGS(params=model_without_ddp.parameters(), lr=args.learning_rate)
    else:
        raise ValueError("Optimizer chosen not implemented!")
    
    # Create scheduler
    if args.enable_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                mode='min',
                                                                factor=args.scheduler_factor,
                                                                patience=args.scheduler_patience,
                                                                threshold=args.scheduler_threshold,
                                                                threshold_mode='rel',
                                                                cooldown=0,
                                                                min_lr=0,
                                                                eps=1e-08,
                                                                verbose=True)

    if args.enable_cudaAMP:
        # Creates GradScaler for CUDA AMP
        scaler = torch.amp.GradScaler()
    else:
        scaler = None
    
    # Trainer
    class_trainer = Trainer(task=args.task,
                            load_embeddings=args.load_embeddings,
                            net=model,
                            class_weights=loss_weights.to(device),
                            optim=optimizer,
                            gradient_clipping_value=args.gradient_clipping_value,
                            scaler=scaler)

    # Saver
    if (not args.distributed) or (args.distributed and (args.rank==0)):
        saver = Saver(Path(args.logdir),
                        vars(args),
                        sub_dirs=list(loaders.keys()),
                        tag=args.experiment)
    else:
        saver = None

    tot_predicted_labels_last = {split:{} for split in loaders}
    if (saver is not None) and (args.ckpt_every <= 0):
        max_validation_accuracy_balanced = 0
        max_test_accuracy_balanced = 0
        save_this_epoch = True
    for epoch in range(args.epochs):
        try:
            for split in loaders:
                if args.distributed:
                    samplers[split].set_epoch(epoch)

                data_loader = loaders[split]

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
                    try:
                        returned_values = class_trainer.forward_batch(input_datas, labels, split)
                    except:
                        print(f"Error in forward batch ({sample_path}).")
                        raise RuntimeError("Error in forward batch!")
                    metrics_dict, (predicted_labels, predicted_scores) = returned_values
                    
                    # Save logs
                    tot_predicted_labels.extend(predicted_labels.tolist() if predicted_labels is not None else [])
                    tot_predicted_scores.extend(predicted_scores.tolist() if predicted_scores is not None else [])

                    tot_sample_paths.extend(sample_path.copy())
                    
                    for k, v in metrics_dict.items():
                        epoch_metrics[k] = epoch_metrics[k] + [v] if k in epoch_metrics else [v]
                        if args.task == 'reconstruction':
                            if saver is not None:
                                saver.log_scalar("Classifier Batch/"+k+"_"+split, v, (epoch * len(data_loader)) + i_batch)
                    
                    # Empty cache
                    del batch
                    del sample_path, labels, labels_name
                    if args.load_embeddings:
                        del embedding
                    else:
                        if args.use_rdm1:
                            del rdm1
                        if args.use_rdm2:
                            del rdm2
                        if args.use_rdm3:
                            del rdm3
                        if args.use_rdmmti1:
                            del rdmmti1
                        if args.use_rdmmti2:
                            del rdmmti2
                        if args.use_rdmmti3:
                            del rdmmti3
                    del input_datas
                    del metrics_dict
                    del predicted_labels
                    del predicted_scores
                    if 'cuda' in device.type:
                        torch.cuda.empty_cache()
                
                # Run scheduler
                if args.enable_scheduler and split=="train":
                    scheduler.step(sum(epoch_metrics['loss'])/len(epoch_metrics['loss']))

                # Print metrics
                for k, v in epoch_metrics.items():
                    avg_v = sum(v)/len(v)
                    if args.distributed:
                        torch.distributed.barrier()
                        avg_v_output = [None for _ in range(args.world_size)]
                        torch.distributed.all_gather_object(avg_v_output, avg_v)
                        avg_v = sum(avg_v_output)/len(avg_v_output)
                    if saver is not None:
                        saver.log_scalar("Classifier Epoch/"+k+"_"+split, avg_v, epoch)
                
                if args.distributed:
                    torch.distributed.barrier()

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
                    
                    # Empty cache
                    del tot_true_labels
                    del tot_predicted_labels
                    del tot_predicted_scores
                    del tot_labels_name
                    del tot_sample_paths
                    if 'cuda' in device.type:
                        torch.cuda.empty_cache()

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
                
                if saver is not None:
                    if args.task == 'classification':
                        # Accuracy Balanced classification
                        accuracy_balanced = utils.calc_accuracy_balanced_classification(tot_true_labels, tot_predicted_labels, num_classes=args.num_classes)
                        saver.log_scalar("Classifier Epoch/accuracy_balanced_"+split, accuracy_balanced, epoch)
                        if (saver is not None) and (args.ckpt_every <= 0):
                            if (split == "validation") and (accuracy_balanced >= max_validation_accuracy_balanced):
                                max_validation_accuracy_balanced = accuracy_balanced
                                save_this_epoch = True
                            if (split == "test") and (accuracy_balanced >= max_test_accuracy_balanced):
                                max_test_accuracy_balanced = accuracy_balanced
                                save_this_epoch = True
                        
                        # Accuracy classification
                        accuracy = utils.calc_accuracy_classification(tot_true_labels, tot_predicted_labels, num_classes=args.num_classes)
                        saver.log_scalar("Classifier Epoch "+split+"/accuracy_"+split, accuracy, epoch)

                        # Precision
                        precision_i = utils.calc_precision(tot_true_labels, tot_predicted_labels, num_classes=args.num_classes)
                        precision = precision_i.mean() if len(precision_i) > 0 else 0.0
                        saver.log_scalar("Classifier Epoch Advanced "+split+"/"+"Precision", precision, epoch)

                        # Recall
                        recall_i = utils.calc_recall(tot_true_labels, tot_predicted_labels, num_classes=args.num_classes)
                        recall = recall_i.mean() if len(recall_i) > 0 else 0.0
                        saver.log_scalar("Classifier Epoch Advanced "+split+"/"+"Recall", recall, epoch)

                        # Specificity
                        specificity_i = utils.calc_specificity(tot_true_labels, tot_predicted_labels, num_classes=args.num_classes)
                        specificity = specificity_i.mean() if len(specificity_i) > 0 else 0.0
                        saver.log_scalar("Classifier Epoch Advanced "+split+"/"+"Specificity", specificity, epoch)

                        # F1 Score
                        f1score = utils.calc_f1(tot_true_labels, tot_predicted_labels, num_classes=args.num_classes)
                        saver.log_scalar("Classifier Epoch Advanced "+split+"/"+"F1 Score", f1score, epoch)
                        
                        # Prediction Agreement Rate: same-sample evaluation agreement between current and previous epoch
                        predictionAgreementRate, tot_predicted_labels_last[split] = utils.calc_predictionAgreementRate(tot_predicted_labels, tot_predicted_labels_last[split], tot_sample_paths)
                        saver.log_scalar("Classifier Epoch Advanced "+split+"/"+"Prediction Agreement Rate", predictionAgreementRate, epoch)
                        
                        # Save logs of error
                        dict_other_info = {}
                        if args.saveLogsError:
                            saver.saveLogsError(saver.output_path[split]/f'{split}_logs_error_{epoch:05d}', tot_true_labels, tot_predicted_labels, tot_predicted_scores, dict_other_info, split, epoch)
                    
                        # Save logs
                        if args.saveLogs:
                            Saver.saveLogs(saver.output_path[split]/f'{split}_logs_{epoch:05d}', tot_true_labels, tot_predicted_labels, tot_predicted_scores, dict_other_info, split, epoch)
                    elif args.task == 'reconstruction':
                        None
                    else:
                        raise ValueError("Task not recognized!")
                
                # Save checkpoint
                if args.distributed:
                    torch.distributed.barrier()
                if saver is not None:
                    if args.ckpt_every > 0:
                        if (split ==  "train") and (epoch % args.ckpt_every == 0):
                            saver.save_model(model_without_ddp, class_trainer.optim, args.experiment, epoch)
                    else: # args.ckpt_every <= 0
                        if save_this_epoch:
                            for filename in glob.glob(str(saver.ckpt_path / (args.experiment+"_best_"+split+"_*"))):
                                os.remove(filename)
                            saver.save_model(model_without_ddp, class_trainer.optim, args.experiment+"_best_"+split, epoch)
                        save_this_epoch = False
                
                # Empty cache
                if args.distributed:
                    del tot_true_labels_output
                    del tot_predicted_labels_output
                    del tot_predicted_scores_output
                    del tot_labels_name_output
                    del tot_sample_paths_output
                del tot_true_labels
                del tot_predicted_labels
                del tot_predicted_scores
                del tot_labels_name
                del tot_sample_paths
                if 'cuda' in device.type:
                    torch.cuda.empty_cache()
        except KeyboardInterrupt:
            print('Caught Keyboard Interrupt: exiting...')
            break

    # Save last checkpoint
    if args.distributed:
        torch.distributed.barrier()
    if saver is not None:
        if args.ckpt_every > 0:
            saver.save_model(model_without_ddp, class_trainer.optim, args.experiment, epoch)
        saver.close()

    if args.start_tensorboard_server:
        print("Finish (Press CTRL+C to close tensorboard and quit)")
    else:
        print("Finish")

    if args.distributed:
        torch.distributed.destroy_process_group()
        

if __name__ == '__main__':
    main()