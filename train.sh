#!/bin/bash

source ./scripts/train_models_sample_tail.sh Res16UNet34C 4 sample_tail --seed=1211 $1 $2

# bash train.sh "./final-project-challenge-2-peiyuanwu/scannet_200" "./final-project-challenge-2-peiyuanwu/scannet_200/train.txt" 

# source scripts/test_weight_and_scale_tail_continue.sh Res16UNet34C 1 toy2 --seed=1211 
