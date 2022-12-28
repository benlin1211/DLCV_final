#!/bin/bash

export PYTHONUNBUFFERED="True"

export DATASET=Scannet200Voxelization2cmDataset

export MODEL=$1  #Res16UNet34C, Res16UNet34D
export BATCH_SIZE=$2
export SUFFIX=$3
export ARGS=$4

export WEIGHTS_SUFFIX=$5

export DATA_ROOT="/home/pywu/final-project-challenge-2-peiyuanwu/scannet_200"
# export PRETRAINED_WEIGHTS="/home/pywu/LanguageGroundedSemseg/pre_train/pretrain.ckpt"
# export PRETRAINED_WEIGHTS="/home/pywu/final-project-challenge-2-peiyuanwu/pre_train/balance_True_v3.ckpt"
export PRETRAINED_WEIGHTS="/home/pywu/final-project-challenge-2-peiyuanwu/output/Scannet200Voxelization2cmDataset/Res16UNet34C-sample_tail/checkpoint-val_miou=60.69-step=25270.ckpt"
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
    --visualize False \
    --visualize_path  $LOG_DIR/visualize \
    --num_gpu 2 \
    --sample_tail_instances True \
    --balanced_category_sampling True \
    --loss_type=weighted_ce \
    --instance_sampling_weights "feature_data/common_double_tail_split_inst_sampling_weights.pkl" \
    --instance_augmentation 'raw' \
    --weights $PRETRAINED_WEIGHTS \
    --resume $LOG_DIR \
    $ARGS \
    2>&1 | tee -a "$LOG"

# sample_tail+weighted_ce
# source scripts/train_weight_and_scale_tail_continue.sh Res16UNet34C 4 tail_continue --seed=1211 


