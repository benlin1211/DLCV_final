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
export PRETRAINED_WEIGHTS="/home/pywu/final-project-challenge-2-peiyuanwu/pre_train/balance_True_v3.ckpt"
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

# sample_tail+weighted_ce
# source scripts/test_weight_and_scale_tail_continue.sh Res16UNet34C 1 tail_continue --seed=1211 

# cd ./output/Scannet200Voxelization2cmDataset/Res16UNet34C-tail_continue/visualize/fulleval
# zip submit_tail_continue.zip *.txt
# cp ./submit_tail_continue.zip ../../../../../..
# rm -rf ./submit_tail_continue.zip
# cd ../../../../..

# NO --instance_augmentation=raw ? multi-target not supported at /opt/conda/conda-bld/pytorch_1623448278899/work/aten/src/THCUNN/generic/ClassNLLCriterion.cu:15
# --sample_tail_instances True \ V