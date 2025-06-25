import os
import cv2 as cv
import pandas as pd
import numpy as np
import sys
from utils_aict22 import *

Views=['Dashboard', 'Rear_view','Rearview_mirror', 'Right_side_window','Rightside_window','Right_window']     
A1_Users= ['User_id_24026', 'User_id_24491', 'User_id_35133', 'User_id_38058', 'User_id_49381']
A2_Users= ['User_id_42271', 'User_id_56306', 'User_id_65818', 'User_id_72519', 'User_id_79336']

def LoadClassList():    
	with open('classes.txt', "r") as f:
		lines=f.read().split('\n')
	return lines

def evaluate(gt_data, sm_data, diff_time_theshold):
	TP=0
	FP=0
	FN=0
	for i in range (len(gt_data)):
		detected=False
		_,gt_cls,gt_StartTime,gt_EndTime=gt_data[i]
		# sm_data=gt_data
		for j in range (len(sm_data)):
			_,sm_cls,sm_StartTime,sm_EndTime=sm_data[j]          
			if abs(sm_StartTime - gt_StartTime)<=diff_time_theshold and abs(sm_EndTime - gt_EndTime)<=diff_time_theshold and int(gt_cls)==int(sm_cls):
					print("%03d(%s)-%03d(%s): (%s)%s" % (sm_StartTime,Second2Time(sm_StartTime), 
						sm_EndTime,Second2Time(sm_EndTime),sm_cls,labels[int(gt_cls)]))
					TP +=1                
			if sm_StartTime>gt_EndTime:
				continue
	FP=len(sm_data)-TP
	FN=len(gt_data)-TP     
	F1=(2*TP)/((2*TP)+FP+FN)        
	return TP, FP, FN, F1

def visualize(gt_data, sm_data, labels):
	font = cv.FONT_HERSHEY_SIMPLEX
	colors=[]
	for R in [0, 128, 255]:
		for G in [0, 128, 255]:
			for B in [0, 128, 255]:
				colors.append([R,G,B])

	width= 1440#max(len(gt_data),len(sm_data)) + 10
	img = np.ones((720,width,3), np.uint8)*255


	for i in range (len(gt_data)):        
		_,gt_cls,gt_StartTime,gt_EndTime=gt_data[i]
		cv.line(img, (2*gt_StartTime, 30), (2*gt_EndTime, 30), colors[int(gt_cls)], thickness=4)
		cv.putText(img, str(gt_cls), (2*gt_StartTime, 10), font, 0.3, colors[0], 1, cv.LINE_AA)

	cv.putText(img, 'Ground truth', (2*gt_StartTime, 50), font, 0.5, colors[0], 1, cv.LINE_AA)

	# sm_data=gt_data
	for j in range (len(sm_data)):
		_,sm_cls,sm_StartTime,sm_EndTime=sm_data[j]
		cv.line(img, (2*sm_StartTime, 130), (2*sm_EndTime, 130), colors[int(sm_cls)], thickness=4) 
		if sm_cls not in [0,18]:		
			cv.putText(img, str(sm_cls), (2*sm_StartTime, 110), font, 0.3, colors[0], 1, cv.LINE_AA) 
	cv.putText(img, 'Predict', (2*gt_StartTime, 150), font, 0.5, colors[0], 1, cv.LINE_AA)

	for l in range(len(labels)):
		top_margin=200
		vspace=20
		cv.line(img, (20, top_margin+vspace*l), (40, top_margin+vspace*l), colors[int(l)], thickness=4)
		cv.putText(img, '%d - %s' %(l,labels[l]), (50, top_margin+5+vspace*l), font, 0.6, colors[l], 1, cv.LINE_AA)

	return img

labels=LoadClassList()

'''

'''
def process(sm_data):    
	for j in range (1,len(sm_data)-1):
		videoName1,cls1,startTime1,endTime1=sm_data[j-1]
		videoName2,cls2,startTime2,endTime2=sm_data[j]
		videoName3,cls3,startTime3,endTime3=sm_data[j+1]
		if (cls1==cls3 and cls1!=cls2 and endTime2-startTime2<=3):
			sm_data[j][1]=cls1


	new_sm_data=[] 
	videoName1,cls1,startTime1,endTime1=sm_data[0]
	for j in range (1,len(sm_data)):
		videoName,cls,startTime,endTime=sm_data[j]
		if int(cls) != int(cls1):
			new_sm_data.append([videoName1,cls1,startTime1,endTime1])
			if (j==len(sm_data)-1):
				new_sm_data.append([videoName,cls,startTime,endTime1])
			else:
				videoName1,cls1,startTime1,endTime1=sm_data[j]
		else:
			if (j==len(sm_data)-1):
				new_sm_data.append([videoName1,cls1,startTime1,endTime1])
			else:
				endTime1=endTime
	return new_sm_data


def main():             
	testSet='A2' 
	diff_time_theshold=3
	gt_dir='./data/%s_224x224' % (testSet)
	sm_dir='./output/Dashboard/%s' % (testSet)

	print('Test on %s' % (testSet))
	if testSet=='A1':
		userIds=A1_Users
	elif testSet=='A2':
		userIds=A2_Users    
	F1_AVG=0
	for userId in userIds:
		for block in ['0','1','2','3','4']:
			for view in Views:                
				videoName='%s_%s_%s' % (view,userId,block)                  
				
				###### Reading predicted data                                
				submit_file='%s/tmp2_%s.csv' % (sm_dir,videoName)
				if not os.path.exists(submit_file):
					continue
				submit_data = pd.read_csv(submit_file)
				submit_data=list(submit_data.values) 

				##### Do some mothod to improve acc                
				print('gt before process: %d' %(len(submit_data)))
				submit_data=process(submit_data)
				

				##### save predicted data after processing
				with open(os.path.join('%s/tmp3_%s.csv' % (sm_dir,videoName)),'w') as f_submit:
					for i in range (len(submit_data)):
						video_id, cls1, start_time, end_time=submit_data[i]
						str="%s,%02d,%03d,%03d\n" % (video_id, cls1, start_time, end_time)
						f_submit.write(str)
						i +=1      

				#### Read ground truth                 
				gt_file = '%s/%s/%s.csv' % (gt_dir,userId,userId)
				if not os.path.exists(gt_file):
					print(gt_file + " not found")
					continue
				gt_data = pd.read_csv(gt_file)
				labeLdata=gt_data.values                

				gt_data=[]
				for i in range (labeLdata.shape[0]):
					gt_userId,gt_fileName,gt_view, activityType, gt_StartTime, gt_EndTime, gt_cls, gr_Block=labeLdata[i]
					gt_fileName=gt_fileName.replace('Rearview','Rear_view')
					gt_fileName=gt_fileName.replace('Rightside window','Rightside_window') 	
					gt_fileName=gt_fileName.strip()
					if ('Rear' in gt_fileName) or ('Right' in gt_fileName):
						continue
					gt_block=gt_fileName[-1];
					if gt_block!=block:
						continue

					gt_StartTime=Time2Second(gt_StartTime)
					gt_EndTime=Time2Second(gt_EndTime)
					gt_data.append([gt_fileName,gt_cls,gt_StartTime,gt_EndTime])

				###### Calculate F1 score
				print(videoName)
				print("gt data: ",len(gt_data))
				print("submit data: ",len(submit_data))
				TP,FP,FN, F1= evaluate(gt_data, submit_data,diff_time_theshold)
				print("TP: %d, FP: %d, FN: %d,F1: %.03f" % (TP,FP,FN,F1))
				F1_AVG=F1_AVG+F1
				print('----------------------------------------------------')
				# print(gt_data)
				print(submit_data)
				img=visualize(gt_data, submit_data,labels)
				cv.imshow("foo",img)
				k=cv.waitKey()
				if k==27:
					sys.exit("Abort by user")
			print('=====================================================================')
	print('F1_AVG: ', F1_AVG/10)
	cv.destroyAllWindows()
if __name__ == '__main__':
	main()