import numpy as np
import os
import glob
import cv2 as cv
import argparse
import torch
import torch.nn as nn
from torchvision import transforms as T
from torch.autograd import Variable
import models
from datasets.utils.video_sampler import *
from utils_aict22 import *

from IPython.core.debugger import set_trace

parser = argparse.ArgumentParser(description='Infernce Action recognition')

parser.add_argument('-a', '--arch', type=str, default='movinet_a0', 
	help="c3d_bn,resnet503d,movinet_a0, movinet_a2, movinet_a5,mobilenet3d_v2")
parser.add_argument('--pretrained-model', type=str, default='./log/best_model.pth.tar',
	help='need to be set for resnet3d models')
parser.add_argument('--pretrained', type=str, default='./log/aicity22/movinet_a0/all_view/best_model.pth.tar')
parser.add_argument('--output-dir', type=str, default='output')
parser.add_argument('--height', type=int, default=224,help="height of an image (default: 224)")
parser.add_argument('--width', type=int, default=224, help="width of an image (default: 112)")
parser.add_argument('--seq-len', type=int, default=16, help="number of images to sample in a tracklet")
parser.add_argument('--video-dir', type=str, default='Dashboard_user_id_42271_NoAudio_3.MP4')
parser.add_argument('--gpu-devices', default='cuda:0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--window-size', type=int, default=3,help="size of sliding window in second")
parser.add_argument('--stride', type=int, default=1,help="moving step of sliding windows")

args = parser.parse_args()
device = torch.device(args.gpu_devices if torch.cuda.is_available() else 'cpu')

labels=['Normal Forward Driving','Drinking','Phone Call(right)','Phone Call(left)',
	'Eating','Text (Right)','Text (Left)','Hair / makeup','Reaching behind',
	'Adjust control panel','Pick up from floor (Driver)','Pick up from floor (Passenger)',
	'Talk to passenger at the right','Talk to passenger at backseat','yawning',
	'Hand on head','Singing with music','shaking or dancing with music','N/A']

DEBUG=True

def LoadVideoIds(video_dir):	
	videoIdFile=os.path.join(video_dir,'video_ids.csv')

	video_dic={}
	video_files=glob.iglob(os.path.join(video_dir, '**/*.mp4'), recursive=True)
	for video_file in video_files:
		key=os.path.basename(video_file).split('.')[0]
		video_dic[key]=video_file
	print(video_dic)

	videoIds={}
	with open(videoIdFile, "r") as f:
		lines=f.read().split('\n')
	for i in range(1,len(lines)):
		if DEBUG:
			print(lines[i])
		# try:
		video_id,videoFile1,videoFile2,videoFile3=lines[i].split(',') 
		videoIds[videoFile1]=[video_id,video_dic[videoFile1]]	
		videoIds[videoFile2]=[video_id,video_dic[videoFile2]]	
		videoIds[videoFile3]=[video_id,video_dic[videoFile3]]					
		# except:
			# print('err: ',video_id)		
	return videoIds

def Predict(model, data, device):		
	data= data.to(device)
	with torch.no_grad():	
		data = Variable(data)		
		outputs = model(data)			
		_, predicted = torch.max(outputs.data, 1)
	return predicted.cpu().numpy()[0]

def process(sm_data):    
    for j in range (1,len(sm_data)-1):
        videoName1,cls1,startTime1,endTime1=sm_data[j-1]
        videoName2,cls2,startTime2,endTime2=sm_data[j]
        videoName3,cls3,startTime3,endTime3=sm_data[j+1]
        if (cls1==cls3 and cls1!=cls2):
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
	if DEBUG:
		print("***** Labels list *****")
		print(labels)
		print("**********************")
	num_classes=len(labels)
	'''
	Load videoIds
	'''
	
	videoIds=LoadVideoIds(args.video_dir)

	if DEBUG:
		print("***** videoIds dict *****")
		print(videoIds)	
		print("************************")
	# sys.exit('Halt for debug')
	'''
	Initial network model
	'''
	print("Initializing model: {}".format(args.arch))
	if args.arch=='movinet_a0':
		from movinets import MoViNet
		from movinets.config import _C
		model = MoViNet(_C.MODEL.MoViNetA0, causal = False, pretrained = True)
		model.classifier[3] = torch.nn.Conv3d(2048, num_classes, (1,1,1))
	else:
		model = models.init_model(name=args.arch, num_classes=num_classes, loss={'xent', 'htri'})	

	pretrained_model=os.path.join(args.pretrained)
	print("Loading checkpoint from '{}'".format(pretrained_model))		
	print("using device: ", device)
	checkpoint = torch.load(pretrained_model) 
	print(checkpoint['rank1'])     
	print(checkpoint['epoch'])  			
	model.load_state_dict(checkpoint['state_dict'])
	model.to(device)
	model.eval()

	'''
	Create a tranforms
	'''
	scaler = T.Resize(((args.height, args.width)))
	normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
							std=[0.229, 0.224, 0.225])    
	transform= T.Compose([
		T.ToPILImage(),
		# scaler,
		T.ToTensor(),
		normalize      
		])  	

	sampler=SystematicSampler(n_frames=args.seq_len)

	for videoId in  videoIds:
		video_file=videoIds[videoId][1]
		print(video_file)
		name=os.path.basename(video_file)	
		name=name.split('.')[0]		
		cap = cv.VideoCapture(video_file)	
		video_fps = 30.0  # cap.get(cv.CAP_PROP_FPS)	
		frameCount=cap.get(cv.CAP_PROP_FRAME_COUNT)
		w_size=args.window_size*video_fps # 3s
		w_step=args.stride*video_fps # 1s
		startFrame=0
		endFrame=startFrame+w_size-1
		cls1=18
		cls2=18
		cls3=18
		loopCount=0
		tmp_data=[]
		output_file=os.path.join(args.output_dir, '%s.csv' % (name))
		if os.path.exists(output_file):
			continue
		with open(output_file,'w') as f:
			while(endFrame<frameCount):
				loopCount +=1
				frames=	sampler(cap,startFrame,endFrame, sample_id=startFrame) 
				frames = [transform(frame) for frame in frames] 
				data = torch.from_numpy(np.stack(frames).transpose((1, 0, 2, 3)))
				c,s, w,h=data.shape
				data=data.view(1,c,s,w,h)			
				predictedId=Predict(model,data,device)	
				'''
					cls1 |---|---|---|
					cls2     |---|---|---|
					cls3         |---|---|---|
				'''
				cls1=cls2
				cls2=cls3
				cls3=predictedId

				if cls1==cls2:
					cls_final=cls1
				elif cls2==cls3:
					cls_final=cls2
				else:
					cls_final=18
				str="%04d,%02d,%02d,%02d,%02d,%s\n" % (int(startFrame/video_fps),cls1,cls2,cls3,cls_final,labels[cls_final])
				print(str)				
				f.write(str)			
				startFrame=startFrame+w_step
				endFrame=startFrame+w_size-1		

			while (startFrame<frameCount):
				cls1=18
				str="%04d,%02d,%02d,%02d,%02d,%s\n" % (int(startFrame/video_fps),cls1,cls2,cls3,cls_final,labels[cls_final])
				print(str)
				f.write(str)							
				startFrame=startFrame+w_step		
if __name__ == '__main__':
	main()