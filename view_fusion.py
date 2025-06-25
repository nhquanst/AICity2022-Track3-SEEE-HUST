import numpy as np
import os
import glob
import cv2 as cv
import pandas as pd
from utils_aict22 import *

def doViewFusion(video_id,view1file,view2file,view3file):
	fusionData=[]
	view1data = pd.read_csv(view1file)
	view1data=list(view1data.values)

	view2data = pd.read_csv(view2file)
	view2data=list(view2data.values)

	view3data = pd.read_csv(view3file)
	view3data=list(view3data.values)

	minlen=min(len(view1data),len(view2data),len(view3data))
	for i in range(minlen):
		time1,_,_,_,cls1,_=view1data[i]
		time2,_,_,_,cls2,_=view2data[i]
		time3,_,_,_,cls3,_=view3data[i]

		if cls1==cls2:
			cls_final=cls1
		elif cls2==cls3:
			cls_final=cls2
		else:
			cls_final=18
		fusionData.append([time1,cls1,cls2,cls3,cls_final])
		result=Convert2SubmitFormat(video_id,fusionData)
	return result

def fineTune(sm_data):    
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
	videoIdFile='/ext_data2/comvis/nhquan/works/AI_CITY_2022_track3/A2/video_ids.csv'
	output_dir='./output/AllViewModel_w180_s30'
	

	with open(os.path.join(output_dir,'submit.txt'),'w') as f_submit:
		with open(videoIdFile, "r") as f:
			lines=f.read().split('\n')
			for i in range(1,len(lines)):
				# print(lines[i])
				try:
					video_id,videoFile1,videoFile2,videoFile3=lines[i].split(',') 
					view1file=os.path.join(output_dir, videoFile1.split('.')[0]+'.csv')
					view2file=os.path.join(output_dir,videoFile2.split('.')[0]+'.csv')
					view3file=os.path.join(output_dir,videoFile3.split('.')[0]+'.csv')

					fusResult=doViewFusion(video_id,view1file,view2file,view3file)
					with open(os.path.join('%s/fusion_%s.csv' % (output_dir,video_id)),'w') as f:
						for j in range (len(fusResult)):
							video_id, cls1, start_time, end_time=fusResult[j]
							str="%s,%02d,%03d,%03d\n" % (video_id, cls1, start_time, end_time)
							f.write(str)						    

					submit_data_view=fineTune(fusResult)
					with open(os.path.join('%s/sm_view_%s.csv' % (output_dir,video_id)),'w') as f:
						for j in range (len(submit_data_view)):
							video_id, cls1, start_time, end_time=submit_data_view[j]
							str="%s,%02d,%03d,%03d\n" % (video_id, cls1, start_time, end_time)
							f.write(str)						    

					##### save to submit file
					for j in range (len(submit_data_view)):
						video_id, cls1, start_time, end_time=submit_data_view[j]
						if cls1 !=18:
							str="%s,%02d,%03d,%03d\n" % (video_id, cls1, start_time, end_time)
							f_submit.write(str)						  

				except:					
					sys.exit('error, videoId ' + video_id)

if __name__ == '__main__':
	main()