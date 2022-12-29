#!/bin/bash

# download and upzip dataset
pip install gdown
bash get_dataset.sh

# download model
get_ckpt.sh

# Copy your txt file to my folder
cp $1/test.txt ./scannet_200

# Copy your ply folder to my folder
cp -r $2/test/ ./scannet_200

source scripts/test_sample_tail.sh Res16UNet34C 1 sample_tail --seed=1211 

# rm -rf $3
# mkdir $3
# cp -r "./output/Scannet200Voxelization2cmDataset/Res16UNet34C-sample_tail/visualize/fulleval/*.txt" "./$3"
cp ./output/Scannet200Voxelization2cmDataset/Res16UNet34C-sample_tail/visualize/fulleval/*.txt $3
# 
# zip submit_final.zip *.txt
# cp ./submit_final.zip ../..
# rm -rf ./submit_final.zip
# cd ../../../../..

# bash inference.sh ./scannet_200_mock ./scannet_200_mock ./output_mock
