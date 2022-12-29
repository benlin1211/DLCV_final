#!/bin/bash

# download and upzip dataset
pip install gdown
bash get_dataset.sh

# download model
get_ckpt.sh

# Copy your txt file to my folder
cp "$1/train.txt" ./scannet_200

# Copy your ply folder to my folder
cp -r "$2/train/" ./scannet_200

echo "Run"

source ./scripts/train_models_sample_tail.sh Res16UNet34C 4 sample_tail --seed=1211

# bash train.sh "./scannet_200_mock" "./scannet_200_mock/" 

# source scripts/test_weight_and_scale_tail_continue.sh Res16UNet34C 1 toy2 --seed=1211 
