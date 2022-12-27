#!/bin/bash

export PYTHONUNBUFFERED="True"

export DATASET=Scannet200Voxelization2cmDataset

export MODEL=$1  #Res16UNet34C, Res16UNet34D
export BATCH_SIZE=$2
export SUFFIX=$3
export ARGS=$4

export WEIGHTS_SUFFIX=$5

export DATA_ROOT="/home/pywu/final-project-challenge-2-peiyuanwu/scannet_200"
export OUTPUT_DIR_ROOT="/home/pywu/final-project-challenge-2-peiyuanwu/output"

export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export LOG_DIR=$OUTPUT_DIR_ROOT/$DATASET/$MODEL-$SUFFIX

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR

LOG="$LOG_DIR/$TIME.txt"

python -m main \
    --log_dir $LOG_DIR \
    --dataset $DATASET \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --val_batch_size $BATCH_SIZE \
    --scannet_path $DATA_ROOT \
    --stat_freq 100 \
    --visualize True\
    --visualize_path  $LOG_DIR/visualize \
    --num_gpu 1 \
    --balanced_category_sampling True \
    --loss_type=weighted_ce \
    --resume $LOG_DIR \
    --is_train False \
	--test_original_pointcloud True \
	--save_prediction True \
    $ARGS \
    2>&1 | tee -a "$LOG"

#    
#    --weights $PRETRAINED_WEIGHTS \
#    --sample_tail_instances True \

# source scripts/test_sample_tail.sh Res16UNet34C 1 sample_tail --seed=1211 

# cd ./output/Scannet200Voxelization2cmDataset/Res16UNet34C-sample_tail/visualize/fulleval
# zip submit_tail6033.zip *.txt
# cp ./submit_tail6033.zip ../../../../../..
# rm -rf ./submit_tail6033.zip
# cd ../../../../..

# scp ./submit.zip pywu@140.112.18.221:/home/pywu/zhongwei/