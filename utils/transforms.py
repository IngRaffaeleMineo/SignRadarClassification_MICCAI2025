import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision.transforms import v2 as torchtransforms
import torch
#import mmcv # (python 3.9) pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
import cv2
import torchvision
#torchvision.set_video_backend("video_reader")
import os
import numpy as np

class ToOneHot():
    """One hot encoding of long value"""

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, x):
        return F.one_hot(x, num_classes=self.num_classes)
    

class Load():
    """Load PT from path provided as tensor"""

    def __init__(self, video_path, dtype=torch.float32):
        self.video_path = video_path
        self.dtype = dtype

    def __call__(self, x):
        path_complete = os.path.join(self.video_path, x)

        if (x[-3:] != ".pt") and (x[-4:] != ".pth"):
            raise ValueError(f"File {x} is not saved with torch.save")

        embedding = torch.load(path_complete)
        
        return embedding.to(dtype=self.dtype)


class LoadVideo():
    """Load MP4 video from path provided as tensor of frames"""

    def __init__(self, video_path, video_type, dtype=torch.float32):
        self.video_path = video_path
        self.video_type = video_type
        self.dtype = dtype

    def __call__(self, x):
        path_complete = os.path.join(self.video_path, x)

        if self.video_type == 'jpeg':
            if not os.path.isdir(path_complete):
                path_complete = '.'.join(path_complete.split('.')[:-1])
        elif self.video_type == 'mp4':
            if not os.path.isfile(path_complete):
                path_complete = path_complete + '.mp4'
        else:
            raise ValueError(f"Unsupported video_type: {self.video_type}, available options are ['jpeg', 'mp4'].")

        if os.path.isfile(path_complete):
            if (x[-3:] != "mp4"):
                raise ValueError(f"Video {x} is not mp4")
        
            #video = mmcv.VideoReader(path_complete)
            video = torchvision.io.VideoReader(path_complete, "video")
            
            frames = next(video)['data'].flip(dims=[-1]).unsqueeze(0)
            
            for frame in video:
                frames = torch.cat( (frames,frame['data'].flip(dims=[-1]).unsqueeze(0)) , dim=0)
            
            return frames.to(dtype=self.dtype) # dim = frames, h, w, c
        else: #folder
            frames = []
            for img_name in sorted(os.listdir(path_complete)):
                img = cv2.imread(os.path.join(path_complete, img_name))
                if img is not None:
                    frames.append(img)
            return torch.from_numpy(np.array(frames)).to(dtype=self.dtype) # dim = frames, h, w, c


class MoveChannelFirst():
    """Move channel to from indicate dimension to 1rd dimension"""
    def __init__(self, original_ch_dim):
        self.original_ch_dim = original_ch_dim

    def __call__(self, x):
        return torch.moveaxis(x, self.original_ch_dim, 0)


class ReduceChannels():
    """Reduce channels to 1 channel, averaging (mean=True) or taking first channel. Expecte channel first """

    def __init__(self, mean):
        self.mean = mean

    def __call__(self, x):
        if self.mean:
            return torch.mean(x, dim=0).unsqueeze(0)
        else:
            return x[0:1, ...]


class Squeeze():
    """Squeeze tensor"""

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x):
        x.squeeze_(self.dim)
        return x


class ToTensor():
    """Convert ndarray to Tensors."""

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).type(dtype=self.dtype)
        elif isinstance(x, torch.Tensor):
            return x.type(dtype=self.dtype)
        elif isinstance(x, list) or isinstance(x, tuple):
            return torch.from_numpy(np.array(x)).type(dtype=self.dtype)
        elif isinstance(x, int) or isinstance(x, float) or \
                isinstance(x, np.int8) or isinstance(x, np.int16) or isinstance(x, np.int32) or isinstance(x, np.int64) or \
                isinstance(x, np.float8) or isinstance(x, np.float16) or isinstance(x, np.float32) or isinstance(x, np.float64):
            return torch.tensor(x).type(dtype=self.dtype)


class Rescale01():
    """Rescale tensor in [0,1]."""

    def __init__(self):
        None

    def __call__(self, x):
        return (x - x.min()) / (x.max() - x.min())


class ResizeVideo():
    """Resize video as ndarray to size (h,w) provided. Require channel last"""

    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        out = np.empty((len(x), self.size[0], self.size[1], 3))
        for i in range(len(x)):
            out[i] = cv2.resize(x[i], dsize=self.size, interpolation=cv2.INTER_CUBIC)
        return out
    

class ResizeWithRatio():
    def __init__(self, size, interpolation, antialias):
        self.h, self.w = size
        self.interpolation = interpolation
        self.antialias = antialias
        
    def __call__(self, inout): # inout shape = (c, frames, h, w)
        if (self.h == -1 and self.w == -1) or (self.h == inout.shape[-2] and self.w == inout.shape[-1]):
            return inout
        
        if self.h == -1 and self.w != -1:
            w = self.w
            wpercent = (w/float(inout.shape[-1]))
            h = int(float(inout.shape[-2])*float(wpercent))
        elif self.w == -1 and self.h != -1:
            h = self.h
            hpercent = (h/float(inout.shape[-2]))
            w = int(float(inout.shape[-1])*float(hpercent))
        else:
            h,w = self.h, self.w
        
        inout = torchtransforms.functional.resize(inout, size=(h,w), interpolation=self.interpolation, antialias=self.antialias)
        #inout = inout.astype(float)

        return inout


class Normalize():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, inout):
        if self.mean == [0.0] and self.std == [1.0]:
            return inout
        return torchtransforms.Normalize(mean=self.mean,std=self.std)(inout)

