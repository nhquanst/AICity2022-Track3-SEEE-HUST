import os
import cv2 as cv
import pandas as pd
import numpy as np
import sys
from utils_aict22 import *


def LoadClassList():    
    with open('classes.txt', "r") as f:
        lines=f.read().split('\n')
    return lines


labels=LoadClassList()
annoFilePath='./output/A2/Dashboard_user_id_42271_NoAudio_3.csv'
print(annoFilePath)
labeLdata = pd.read_csv(annoFilePath)
# names=('UserID','Filename', 'CameraView','ActivityType','StartTime','EndTime','Label','AppearanceBlock'))
labeLdata=labeLdata.values
print(labeLdata.shape)    
outputData=Convert2SubmitFormat('user_id_42271',list(labeLdata))
for video_id, cls, start_time, end_time in outputData:
    str="%s, %02d, %03d, %03d, %s" % (video_id,cls,start_time,end_time,labels[cls])                 
    print(str)