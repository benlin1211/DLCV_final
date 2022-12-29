#!/bin/bash

source scripts/test_sample_tail.sh Res16UNet34C 1 sample_tail --seed=1211 

# cd ./output/Scannet200Voxelization2cmDataset/Res16UNet34C-sample_tail/visualize/fulleval
# zip submit_final.zip *.txt
# cp ./submit_final.zip ../..
# rm -rf ./submit_final.zip
# cd ../../../../..
