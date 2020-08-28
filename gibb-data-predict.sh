#!/bin/bash

ANACONDA_PATH=/home/ec2-user/anaconda3
CONDA_ENV=tensorflow2_p36

MODEL=$(pwd)/deepchannel-model-20200716T033708Z.h5
INFERENCE_PATH=$(pwd)/nmdar_data/gibb_data
INFERENCE_OUT="$INFERENCE_PATH"/004-base_12_invert_predictions.csv.gz

source "$ANACONDA_PATH"/bin/activate "$CONDA_ENV"

python predict.py --invert -o "$INFERENCE_OUT" "$MODEL" "$INFERENCE_PATH"/004-base_[12].csv

conda deactivate
