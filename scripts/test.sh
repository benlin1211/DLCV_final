#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED="True"

export DATASET=Scannet200Voxelization2cmDataset

export MODEL=$1  #Res16UNet34C or Res16UNet34D
export BATCH_SIZE=$2
export SUFFIX=$3
export ARGS=$4

# export WEIGHTS_SUFFIX=$5

export DATA_ROOT="/home/twsbvze943/zhong/LanguageGroundedSemseg/dataset"
# export PRETRAINED_WEIGHTS="/mnt/Data//weights/"$WEIGHTS_SUFFIX
export OUTPUT_DIR_ROOT="/home/twsbvze943/zhong/LanguageGroundedSemseg/output"

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
    --visualize_path  $LOG_DIR/visualize \
    --num_gpu 1 \
    --balanced_category_sampling True \
    --resume $LOG_DIR \
    --is_train False \
	--test_original_pointcloud True \
	--save_prediction True \
    --visualize True\
    $ARGS \
    2>&1 | tee -a "$LOG"

#    
#    --weights $PRETRAINED_WEIGHTS \


# source scripts/test.sh Res16UNet34C 1 toyexample --seed=1211 