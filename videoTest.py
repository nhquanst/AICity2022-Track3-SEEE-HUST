import numpy as np
import os
import cv2 as cv
import argparse
import torch
import torch.nn as nn
from torchvision import transforms as T
from torch.autograd import Variable
import models
from datasets.utils.video_sampler import *
from IPython.core.debugger import set_trace

parser = argparse.ArgumentParser(description='Infernce Action recognition')

parser.add_argument('-a', '--arch', type=str, default='resnet503d', 
	help="c3d_bn,resnet503d,movinet_a0, movinet_a2, movinet_a5,mobilenet3d_v2")
parser.add_argument('--pretrained-model', type=str, default='./log/best_model.pth.tar',
	help='need to be set for resnet3d models')
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--height', type=int, default=224,help="height of an image (default: 224)")
parser.add_argument('--width', type=int, default=224, help="width of an image (default: 112)")
parser.add_argument('--seq-len', type=int, default=16, help="number of images to sample in a tracklet")
parser.add_argument('--video-file', type=str, default='Dashboard_user_id_42271_NoAudio_3.MP4')
parser.add_argument('--gpu-devices', default='cuda:0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()
device = torch.device(args.gpu_devices if torch.cuda.is_available() else 'cpu')

def LoadClassList():    
	with open('classes.txt', "r") as f:
		lines=f.read().split('\n')
	return lines

def Predict(model, data, device):		
	data= data.to(device)
	with torch.no_grad():	
		# data = Variable(data)		
		outputs = model(data)			
		print(outputs.data)
		_, predicted = torch.max(outputs.data, 1)
	return predicted
def main():	
	best_model_file = os.path.join(args.save_dir, 'best_model.pth.tar')
	scaler = T.Resize(((args.height, args.width)))
	normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
							std=[0.229, 0.224, 0.225])    
	transform= T.Compose([
		T.ToPILImage(),
		scaler,
		T.ToTensor(),
		normalize      
		])  	

	sampler=RandomSystematicSampler(n_frames=args.seq_len,window_size=60)
	num_classes=7
	labels=LoadClassList()
	print(labels)

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

	print("Loading checkpoint from '{}'".format(args.pretrained_model))		
	print("using device: ", device)
	checkpoint = torch.load(args.pretrained_model) 
	print(checkpoint['rank1'])     
	print(checkpoint['epoch'])  			
	model.load_state_dict(checkpoint['state_dict'])
	model.to(device)
	model.eval()
	

	video_fps = 20.0  # cap.get(cv.CAP_PROP_FPS)			
	framesz=(720,480)	
	fourcc = cv.VideoWriter_fourcc(*'mp4v')	
	name=os.path.basename(args.video_file)
	out = cv.VideoWriter('./output/%s' % (name), fourcc, video_fps, framesz)

	font = cv.FONT_HERSHEY_SIMPLEX   
	cap = cv.VideoCapture(args.video_file)
	cap1 = cv.VideoCapture(args.video_file)
	frameCount=cap.get(cv.CAP_PROP_FRAME_COUNT)
	ret=True			
	ret, frame = cap.read()	
	frameId=1
	predictedLabel=""
	while(ret):
		if frameId%video_fps==0 and frameId>=30 and frameId<frameCount-30-1:
			frames=	sampler(cap1,frameId-29,frameId+29, sample_id=frameId) 
			frames = [transform(frame) for frame in frames] 
			data = torch.from_numpy(np.stack(frames).transpose((1, 0, 2, 3)))
			c,s, w,h=data.shape
			print(data.shape)
			data=data.view(1,c,s,w,h)	
			predictedId=Predict(model,data,device)
			predictedLabel=labels[predictedId]
			print(predictedLabel)
		cv.putText(frame,predictedLabel,(30, 30), font, 1, (255,255,0), 2)
		cv.imshow("preview", frame) 	
		vframe=cv.resize(frame,framesz)
		out.write(vframe)		
		key = cv.waitKey(1)
		if key == 27: # exit on ESC
			break		
		ret, frame = cap.read()	
		frameId +=1
	cv.destroyWindow("preview")
	cap.release()
	cap1.release()
	out.release()

if __name__ == '__main__':
	main()

