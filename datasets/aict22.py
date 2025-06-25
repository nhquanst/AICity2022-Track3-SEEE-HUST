import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from IPython.core.debugger import set_trace

from .utils.video_sampler import *

__all__ = ['AICT22Dataset']


class AICT22Dataset(Dataset):
    def __init__(self,
                 video_dir,
                 annotation_file_path="",
                 sampler=SystematicSampler(n_frames=16),
                 to_rgb=True,
                 transform=None,
                 use_albumentations=False,
                 views=['Dashboard','Rear_view','Rightside_window'] # Rear_view, Rightside_window, Dashboard
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
        # set_trace()
        # for video_file in filter(lambda _: _.endswith('.mp4'),
                                 # sorted(os.listdir(self.video_dir))):
        for video_file in glob.iglob(self.video_dir + '/**/*.mp4', recursive=True):            
            video_name=os.path.basename(video_file)                   
            video_name=video_name.split('.')[0]
            video_name=video_name.replace('Rearview','Rear_view')
            video_name=video_name.replace('Rightside window','Rightside_window')
            video_name=video_name.replace('Right_window','Rightside_window')            
            seq,numbLock,userId,view,label,block=video_name.split('-')            
            if view in self.views:
                label = int(label)
                # video_file = os.path.join(self.video_dir, video_file)
                self.clips.append((video_file, userId))
                self.labels.append(label)
            else:
                print("View name not found: " + video_name)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, item):
        video_file, subject = self.clips[item]     
        frames = self.sampler(video_file, sample_id=item)        
        if self.to_rgb:
            frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        if self.transform is not None:
            frames = [self.transform(frame) if not self.use_albumentations
                      else self.transform(image=frame)['image'] for frame in frames]                
        data = torch.from_numpy(np.stack(frames).transpose((1, 0, 2, 3)))        
        # print(data.shape) # data shape (c , s , w, h)  s for seq_len, c for channel
        return data, self.labels[item],video_file   