import os
#import platform
import copy
import torch
from torch.utils import data as torchdata
from torchvision.transforms import v2 as torchtransforms
if __name__ == "__main__":
    import transforms as mytransforms
else:
    import utils.transforms as mytransforms
import argparse
import sys
from os.path import dirname, abspath

class Dataset(torchdata.Dataset):
    def __init__(self, split_path, section, num_fold, num_classes, transforms, inner_loop = 0):  
        self.section = section
        self.num_fold = num_fold
        self.num_classes = num_classes
        self.transforms = transforms # dict of list of trasforms
        self.inner_loop = inner_loop
        
        self.datas = self._generate_data_list(split_path)
    

    #split data in train, val and test sets in a reproducible way
    def _generate_data_list(self, split_path):
        path=torch.load(split_path)
        datas = []
        
        if self.section == 'test':
            datas = path[f'fold{self.num_fold}']['test']
        elif self.section == 'training':
            datas = path[f'fold{self.num_fold}'][f'inner{self.inner_loop}']['train']
        elif self.section == 'validation':
            datas = path[f'fold{self.num_fold}'][f'inner{self.inner_loop}']['val']
        else: 
            raise ValueError(
                    f"Unsupported section: {self.section}, "
                    "available options are ['training', 'validation', 'test']."
                )
        
        #if platform.system() != 'Windows':
        #    for sample in data:
        #        for key in sample.keys():
        #            sample[key] = sample[key].replace('\\', '/')
        return datas
    

    def get_label_proportions(self):
        c = [None]*self.num_classes
        label_props = [None]*self.num_classes
        for i in range(self.num_classes):
            c[i] = len([None for data in self.datas if data['label'] == i])
        for i in range(len(c)):
            if c[i] == 0:
                label_props[i] = 0
            else:
                label_props[i] = max(c)/c[i]
        return label_props


    def __len__(self):
        return len(self.datas)
    

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datas[idx])
        data['sample_path'] = data['rdm1'].split('_radar_rdm1_')[0]
        data['embedding'] = data['rdm1'].split('_radar_rdm1_')[0]+'.pt'
        if self.transforms:
            for item in self.transforms:
                keys, transform = item
                assert isinstance(keys, list), "Keys within a transform must be a list."
                for k in keys:
                    data[k] = transform(data[k])
        return data




def get_loader(args):
    basics_1 = []
    
    if args.load_embeddings:
        basics_1 = [
            *basics_1,
            (['embedding'], mytransforms.Load(args.video_path))
        ]
    else:
        video_keys = []
        if args.use_rdm1:
            video_keys.append('rdm1')
        if args.use_rdm2:
            video_keys.append('rdm2')
        if args.use_rdm3:
            video_keys.append('rdm3')
        if args.use_rdmmti1:
            video_keys.append('rdmmti1')
        if args.use_rdmmti2:
            video_keys.append('rdmmti2')
        if args.use_rdmmti3:
            video_keys.append('rdmmti3')
            
        basics_1 = [
            *basics_1,
            (video_keys, mytransforms.LoadVideo(args.video_path, args.video_type))
        ]
        basics_1 = [
            *basics_1,
            (video_keys, mytransforms.MoveChannelFirst(original_ch_dim=1)),
            (video_keys, mytransforms.ReduceChannels(mean=False)),
            (video_keys, mytransforms.ResizeWithRatio(size=args.resize[1:3], interpolation=torchtransforms.InterpolationMode.NEAREST_EXACT, antialias=True)),
            (video_keys, mytransforms.ReduceFrame(number_frame=args.resize[0])),
            (video_keys, mytransforms.ToTensor(dtype=torch.float32)),
            (video_keys, mytransforms.Squeeze(dim=0)),
            (video_keys, mytransforms.Rescale01()),
            (video_keys, mytransforms.Normalize(mean=args.mean,std=args.std))
        ]
        if args.multiCrop:
            if args.multiCropMode == 'random':
                basics_1 = [
                    *basics_1,
                    (video_keys, mytransforms.MultiCropRandom(frame_per_crop=args.frame_per_crop, number_crop=args.number_crop)),
                ]
            elif args.multiCropMode == 'consecutives':
                basics_1 = [
                    *basics_1,
                    (video_keys, mytransforms.MultiCropConsecutives(frame_per_crop=args.frame_per_crop, number_crop=args.number_crop)),
                ]
            else:
                raise ValueError(
                    f"Unsupported multiCropMode: {args.multiCropMode}, "
                    "available options are ['random', 'consecutives']."
                )

    basics_1.append( (['label'], mytransforms.ToTensor(dtype=torch.long)) )
    
    train_transforms = [*basics_1]
    train_transforms = [
        *train_transforms,
        #(video_keys, torchtransforms.RandomHorizontalFlip(p=0.5)) # for rdm is not applicable because it would invert the speed
        #(video_keys, torchtransforms.RandomVerticalFlip(p=0.5)) # for rdm is not applicable because it would reverse the distance from the antenna
        ]

    val_transforms = [*basics_1]
    val_transforms = [
        *val_transforms,
        ]
    
    dataset = {}
    dataset["train"] = Dataset(split_path = args.split_path, section = 'training', num_fold = args.num_fold, num_classes=args.num_classes, transforms = train_transforms, inner_loop = args.inner_loop)
    dataset["validation"] = Dataset(split_path = args.split_path, section = 'validation', num_fold = args.num_fold, num_classes=args.num_classes, transforms = val_transforms, inner_loop = args.inner_loop)
    dataset["test"] = Dataset(split_path = args.split_path, section = 'test', num_fold = args.num_fold, num_classes=args.num_classes, transforms = val_transforms, inner_loop = args.inner_loop)
    
    samplers = {}
    if args.distributed:
        samplers["train"] = torchdata.DistributedSampler(dataset["train"], shuffle=True)
        samplers["validation"] = torchdata.DistributedSampler(dataset["validation"], shuffle=False)
        samplers["test"] = torchdata.DistributedSampler(dataset["test"], shuffle=False)
    else:
        samplers["train"] = torchdata.RandomSampler(dataset["train"])
        samplers["validation"] = torchdata.SequentialSampler(dataset["validation"])
        samplers["test"] = torchdata.SequentialSampler(dataset["test"])

    batch_sampler = {}
    batch_sampler['train'] = torchdata.BatchSampler(samplers["train"], args.batch_size, drop_last=True)
    batch_sampler['validation'] = torchdata.BatchSampler(samplers["validation"], args.batch_size, drop_last=False)
    batch_sampler['test'] = torchdata.BatchSampler(samplers["test"], args.batch_size, drop_last=False)

    loaders = {}
    loaders["train"] = torchdata.DataLoader(dataset["train"], batch_sampler = batch_sampler['train'], num_workers=24, pin_memory=False, persistent_workers=False) # if big dataset: pin_memory=False and persistent_worker=False
    loaders["validation"] = torchdata.DataLoader(dataset["validation"], batch_sampler = batch_sampler['validation'], num_workers=24, pin_memory=False, persistent_workers=False) # if big dataset: pin_memory=False and persistent_worker=False
    loaders["test"] = torchdata.DataLoader(dataset["test"], batch_sampler = batch_sampler['test'], num_workers=24, pin_memory=False, persistent_workers=False) # if big dataset: pin_memory=False and persistent_worker=False
    
    loss_weights = torch.Tensor(dataset["train"].get_label_proportions())

    return loaders, samplers, loss_weights

import matplotlib.pyplot as plt
if __name__ == '__main__':
    args={}
    args['load_embeddings'] = True
    args['video_type'] = 'mp4' # 'jpeg'
    args['multiCrop'] = False
    args['multiCropMode'] = 'random' # 'consecutives'
    args['frame_per_crop'] = 100
    args['number_crop'] = 10
    args['resize'] = [64,256,1024]
    args['mean'] = [0.0]
    args['std'] = [1.0]
    args['split_path'] = os.path.join('../data/path/to/BNCV5F.pt')
    args['video_path'] = os.path.join('../data/path/to/checkpoint')
    args['num_fold'] = 0
    args['inner_loop'] = 0
    args['distributed'] = False
    args['batch_size'] = 8
    args['num_classes'] = 126
    args['use_rdm1'] = True
    args['use_rdm2'] = True
    args['use_rdm3'] = True
    args['use_rdmmti1'] = True
    args['use_rdmmti2'] = True
    args['use_rdmmti3'] = True
    torch.manual_seed(0)
    
    args = argparse.Namespace(**args)

    # Dataset e Loader
    loaders, samplers, loss_weights = get_loader(args)
    
    print(loss_weights)

    # Get samples
    tmp = next(iter(loaders["train"]))

    print(tmp['sample_path'][0])
    
    print(tmp['label'][0])
    
    if args.load_embeddings:
        print(tmp['embedding'][0].shape)
        print(tmp['embedding'][0].min())
        print(tmp['embedding'][0].max())
    else:
        print(tmp['rdm1'][0].shape)
        print(tmp['rdm1'][0].min())
        print(tmp['rdm1'][0].max())
        
        plt.imshow( tmp['rdm1'][0][0,:,:].numpy() )
        plt.savefig('dataset-test.png')