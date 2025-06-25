import os
import sys
import os.path as osp

'''
    strtime: xx:xx:xx
'''
def Time2Frame(strtime,fps):
    hh,mm,ss=strtime.split(':');    
    hh=int(hh)
    mm=int(mm)
    ss=int(ss)
    return fps*(ss+mm*60+hh*3600)

'''
    strtime: xx:xx:xx
'''
def Time2Second(strtime):
    hh,mm,ss=strtime.split(':');    
    hh=int(hh)
    mm=int(mm)
    ss=int(ss)
    return (ss+mm*60+hh*3600)

def Second2Time(n):
    hh=n//3600
    n=n-3600 *hh    
    mm=n//60
    ss=n-60*mm    
    return ("%02d:%02d:%02d" % (hh,mm,ss)) 

def Convert2SubmitFormat(video_id,inputData):
    outputData=[]
    start_idx=0
    start_time,_,_,_,cls1=inputData[start_idx]
    for i in range (len(inputData)):
        end_time,_,_,_,cls=inputData[i]
        if cls1!= cls:                    
            outputData.append([video_id, cls1, start_time, end_time])
            start_idx=i
            start_time,_,_,_,cls1=inputData[start_idx]    
    outputData.append([video_id, cls, start_time, end_time+1])
    return outputData    