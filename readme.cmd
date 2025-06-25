
#https://github.com/kenshohara/3D-ResNets-PyTorch
#https://github.com/okankop/Efficient-3DCNNs
## Train model
	sh runTrainingAllTrials.sh
	==> best_model is stored in ./log/[method]/...

## Read all result (At last line in train_log file)
	python3 readFinalResult.py
	==> copy output text

## Test the best_model
	sh runTestingAllTrials.sh

## Extract feature
	sh runExtractFeatAllTrials.sh
	==> ./_output
