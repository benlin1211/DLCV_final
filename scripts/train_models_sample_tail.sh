#!/bin/bash

export PYTHONUNBUFFERED="True"

export DATASET=Scannet200Voxelization2cmDataset

export MODEL=$1  #Res16UNet34C, Res16UNet34D
# echo $MODEL
export BATCH_SIZE=$2
export SUFFIX=$3
export ARGS=$4
# echo $ARGS

export DATA_ROOT="./scannet_200"
# echo $DATA_ROOT
export PRETRAINED_WEIGHTS="./pre_train/pretrain.ckpt"
# export PRETRAINED_WEIGHTS="./pre_train/balance_True_v3.ckpt"

export OUTPUT_DIR_ROOT="./output"

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
    --visualize False \
    --visualize_path  $LOG_DIR/visualize \
    --num_gpu 2 \
    --sample_tail_instances True \
    --balanced_category_sampling True \
    --loss_type=weighted_ce \
    --resume $LOG_DIR \
    $ARGS \
    2>&1 | tee -a "$LOG"

#     --weights $PRETRAINED_WEIGHTS \
# sample_tail+weighted_ce
# source scripts/train_models_sample_tail.sh Res16UNet34C 4 sample_tail --seed=1211 \
# "./scannet_200"  \
# "./scannet_200/train.txt" 

# NO --instance_augmentation=raw ? multi-target not supported at /opt/conda/conda-bld/pytorch_1623448278899/work/aten/src/THCUNN/generic/ClassNLLCriterion.cu:15
# --sample_tail_instances True \ V