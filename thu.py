import os
import glob
import cv2
import numpy as np
count=0
for video_file in glob.iglob('./data/' + '**/*.mp4', recursive=True):  
    count +=1
    print(video_file)          
    video_name=os.path.basename(video_file)   
    print(video_name)         
    video_name=video_name.split('.')[0]
    video_name.replace('Rearview','Rear_view')
    video_name.replace('Rightside window','Rightside_window')
    print(video_name)
    seq,numbLock,userId,view,label,block=video_name.split('-')
    print(int(label))
    print(count)