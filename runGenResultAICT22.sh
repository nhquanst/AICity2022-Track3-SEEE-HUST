#!/bin/bash
dataset="aicity22" 
hostname=$(hostname)
echo "[bash] Running on $hostname"
batch=8

case "$hostname" in
    "X9DRL-3F-iF")
		# video='/media/nhquan/Data_potable/datasets/AI_CITY_CHALLENGE/AIC22_Track3_updated/A1/user_id_24026/Rear_view_User_id_24026_3.MP4'		
		video='/media/nhquan/Data_potable/datasets/AI_CITY_CHALLENGE/AIC22_Track3_updated/A2_224x224/'		
    	cd /mnt/works/ActionRecognition/AI_CITY_2022/code
		source /mnt/works/projectComvis/AFOSR-2020/multi_stream_videonet/.env/bin/activate
    	;;
    "Server2")
    	data="/mnt/data3t/datasets/EgoGesture/normal"
		trlist="/mnt/data3t/datasets/EgoGesture/train.txt"
		telist="/mnt/data3t/datasets/EgoGesture/test.txt" 
    	export CUDA_VISIBLE_DEVICES="0"
    	cd /mnt/disk3/users/nhquan/projectComvis/AFOSR-2020/video_classification
		source ~/pytorch_env/bin/activate
    	;;
  	"hungvuong")
    	data="/mnt/data3t/datasets/EgoGesture/normal"
		trlist="/mnt/data3t/datasets/EgoGesture/train.txt"
		telist="/mnt/data3t/datasets/EgoGesture/test.txt" 
    	export CUDA_VISIBLE_DEVICES="0"    	
	;;
    *)
	echo other
 	;;
esac

# views="Dashboard Rear_view Rightside_window", all_view
model="movinet_a0" 			# resnet183d resnet503d c3d_bn movinet_a0 movinet_a2 efficientnet3d mobilenet3d_v2
saveDir="./log/$dataset/$model"		
# views="Dashboard Rear_view Rightside_window" # Trained view 
views="Rightside_window" # trained view
output='./output/Rightside_window/A2' 
echo "$saveDir/$views"

date "+%H:%M:%S   %d/%m/%y"
if [ "$model" = "c3d_bn" ]; then
	python3 main.py --job evaluate -a $model --dataset-dir $video --train-annotation-file $trlist \
	--test-annotation-file $telist --height 112 --width 112 --dataset $dataset --test-batch $batch \
	--max-epoch 30 --eval-step 10 --print-freq 5 --save-dir $saveDir
elif [ "$model" = "resnet503d" ]; then
	python3 main.py --job evaluate -a $model --dataset-dir $data --train-annotation-file $trlist \
	--test-annotation-file $telist --height 224 --width 224 --dataset $dataset --test-batch $batch \
	--max-epoch 30 --eval-step 10 --print-freq 5 --save-dir $saveDir 
elif [ "$model" = "resnet183d" ]; then
	python3 main.py --job evaluate -a $model --dataset-dir $data --train-annotation-file $trlist \
	--test-annotation-file $telist --height 224 --width 224 --dataset $dataset --test-batch $batch \
	--save-dir $saveDir 
elif [ "${model:0:7}" = "movinet" ]; then
	python3 genResult.py -a $model --video-dir $video --views $views\
	--height 172 --width 172 --model-dir $saveDir --output-dir $output --seq-len 32
elif [ "$model" = "efficientnet3d" ]; then
	python3 main.py --job evaluate -a $model --dataset-dir $data --train-annotation-file $trlist \
	--test-annotation-file $telist --height 224 --width 224 --dataset $dataset --test-batch $batch \
	--save-dir $saveDir 
elif [ "${model:0:14}" = "mobilenet3d_v2" ]; then
	widthMult=1.0 # using with mobilenet
	python3 main.py --job evaluate -a $model --dataset-dir $data --train-annotation-file $trlist \
	--test-annotation-file $telist --height 112 --width 112 --dataset $dataset --test-batch $batch \
	--save-dir $saveDir --width-mult $widthMult
elif [ "$model" = "slow_fast_r3d_18" ]; then
	python3 main.py --job evaluate -a $model --dataset-dir $data --train-annotation-file $trlist \
	--test-annotation-file $telist --height 112 --width 112 --dataset $dataset --test-batch $batch \
	--save-dir $saveDir
else	
	echo "Wrong parameter --> code cound not  be runed"
fi	
date "+%H:%M:%S   %d/%m/%y"
