#!/bin/bash
dataset="UTCDA" 
hostname=$(hostname)
echo "[bash] Running on $hostname"
batch=8

case "$hostname" in
    "X9DRL-3F-iF")
		data="/mnt/tasks/2024_UTC/code_SEEE-HUST/data/UTCDA/videos_splited"
		# trainviews="Dashboard Rear_view Rightside_window"		
    	# cd /mnt/works/ActionRecognition/AI_CITY_2022/code
		# conda activate cobot_39
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
    "js4-desktop")
		data="/media/js4/Data_potable/AFOSR-2020/datasets/data"
		trlist="/media/js4/Data_potable/AFOSR-2020/datasets/train.txt"
		telist="/media/js4/Data_potable/AFOSR-2020/datasets/val.txt"
		cd /media/js4/Data_potable/AFOSR-2020/video_classification
		source ../.env/bin/activate
		batch=1
	;;
    *)
	echo other
 	;;
esac

model="movinet_a0" 			# resnet183d resnet503d c3d_bn movinet_a0 movinet_a2 efficientnet3d mobilenet3d_v2
trainviews="front rear"
valviews="front rear"
saveDir="./log/$dataset/$model"		
echo "$saveDir/$trainviews"

date "+%H:%M:%S   %d/%m/%y"
if [ "$model" = "c3d_bn" ]; then
	python3 main.py --job evaluate -a $model --dataset-dir $data --train-annotation-file $trlist \
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
	python3 main.py --job evaluate -a $model --train-dir $data --train-view $trainviews \
	--val-dir $data  --val-view $valviews --height 172 --width 172 --dataset $dataset \
	--test-batch $batch --save-dir $saveDir --seq-len 32
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
