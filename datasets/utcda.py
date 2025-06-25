import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from IPython.core.debugger import set_trace

from .utils.video_sampler import *

__all__ = ['UTCDA_Dataset']


class UTCDA_Dataset(Dataset):
    def __init__(self,
                 video_dir,
                 annotation_file_path="",
                 sampler=SystematicSampler(n_frames=16),
                 to_rgb=True,
                 transform=None,
                 use_albumentations=False,
                 views="front rear"              
                 ):
        self.video_dir = video_dir
        self.annotation_file_path = annotation_file_path
        self.sampler = sampler
        self.to_rgb = to_rgb
        self.transform = transform
        self.use_albumentations = use_albumentations
        self.views=views
        self.clips = []
        self.labels = []

        with open(self.annotation_file_path, 'r') as file:
            for line in file:                
                video_file,label=line.strip().split(",")               
                if not os.path.isabs(video_file):
                    video_file=os.path.join(video_dir,video_file)
                video_name=os.path.basename(video_file).split(".")[0]
                _,userId,_,_,start,stop,view=video_name.split('_')
                print(video_file, " : ",label, " : ",view)
                cap = cv2.VideoCapture(video_file)
                frame_count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if view in self.views:
                    label = int(label)                    
                    idxs=list(range(0,int(frame_count)-61,20)) # fps=20
                    for idx in idxs:
                        self.clips.append((video_file,idx,idx+60))
                        self.labels.append(label)                
                else:
                    print("View name not found: " + view)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, item):
        video_file, start, stop = self.clips[item]     
        frames = self.sampler(video_file,start, stop, sample_id=item)        
        if self.to_rgb:
            frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        if self.transform is not None:
            frames = [self.transform(frame) if not self.use_albumentations
                      else self.transform(image=frame)['image'] for frame in frames]                
        data = torch.from_numpy(np.stack(frames).transpose((1, 0, 2, 3)))        
        # print(data.shape) # data shape (c , s , w, h)  s for seq_len, c for channel
        return data, self.labels[item],video_file   