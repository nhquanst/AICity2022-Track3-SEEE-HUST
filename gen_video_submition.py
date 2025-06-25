import os
import cv2 as cv
import pandas as pd
import numpy as np
import sys


def LoadClassList():    
    with open('classes.txt', "r") as f:
        lines=f.read().split('\n')
    return lines

'''
    strtime: xx:xx:xx
'''
def Time2Frame(strtime,fps):
    hh,mm,ss=strtime.split(':');    
    hh=int(hh)
    mm=int(mm)
    ss=int(ss)
    return fps*(ss+mm*60+hh*3600)

DataDir="/media/nhquan/Data_potable/datasets/AI_CITY_CHALLENGE/AIC22_Track3_updated/A2_640x360/User_id_42271"
outputDir="./visualize/A2_640x360"
subjectDirs=os.listdir(DataDir)
# fourcc=cv.VideoWriter_fourcc(*'X264')      # 'M','J','P','G'  *'X264' *'mp4v' 'H','2','6','4'
fourcc=cv.VideoWriter_fourcc(*'mp4v') 
font = cv.FONT_HERSHEY_SIMPLEX
frame_sz=(640,360)  

labels=LoadClassList()

#['user_id_24026','user_id_24491','user_id_35133','user_id_38058']: #, 'user_id_49381'

for subjectDir in subjectDirs:
    if subjectDir not in ['user_id_56306']:
        continue    
    count=0
    os.makedirs(os.path.join(outputDir,subjectDir), exist_ok=True)
    subject=int(subjectDir.split('_')[-1])
    subjectPath = os.path.join(DataDir, subjectDir)    
    annoFilePath=os.path.join(subjectPath,subjectDir + '.csv')
    print(annoFilePath)
    labeLdata = pd.read_csv(annoFilePath)
    # names=('UserID','Filename', 'CameraView','ActivityType','StartTime','EndTime','Label','AppearanceBlock'))
    labeLdata=labeLdata.values
    print(labeLdata.shape)    
    fileName=""
    cap=None
    vfile=None
    for i in range (labeLdata.shape[0]):
        count +=1
        userId=labeLdata[i][0]
        if subject !=userId:
            sys.exit("Error in csv file %s <> %s" % (subject,userId))            
        if fileName!=labeLdata[i][1].strip():
            print(fileName + ":" + labeLdata[i][1].strip())
            fileName=labeLdata[i][1].strip()
            file_seq=int(fileName.split('_')[-1])
            videoPath=os.path.join(subjectPath,fileName + '.MP4')
            print(videoPath)
            cap = cv.VideoCapture(videoPath) 
            frameCount=0
            fps = cap.get(cv.CAP_PROP_FPS)
            if fps==0: fps=30
            videoOutputFile=os.path.join(outputDir,subjectDir,fileName+'.mp4')                        
            vfile=cv.VideoWriter(videoOutputFile,fourcc,30,frame_sz)            
        view=labeLdata[i][2].strip()
        activityType=labeLdata[i][3]
        if 'Distracted' not in activityType:
            print(activityType)
        startTime=labeLdata[i][4].strip()
        startFrame=Time2Frame(startTime,fps)
        endTime=labeLdata[i][5].strip()
        endFrame=Time2Frame(endTime,fps)         
        cls=labeLdata[i][6]
        try:
            cls=int(cls)
        except:
            cls=18
        # print(label)  
        label=labels[cls]      
        appearanceBlock=labeLdata[i][7].strip() 
        while frameCount <endFrame:        
            frameCount +=1
            res, frame = cap.read()
            if not res:
                print("Error reading video:")
                break
            frame=cv.resize(frame,frame_sz)
            string="%s: %s --> %s" % (label,startTime,endTime)
            if frameCount>=startFrame:
                cv.putText(frame,string,(30, 30), font, 0.5, (255,255,0), 2)
            vfile.write(frame)              
                        
