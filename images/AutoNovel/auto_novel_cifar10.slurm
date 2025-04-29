#!/bin/bash
#SBATCH --job-name=AutoNovel_c10
#SBATCH --output=autonovel_output_cifar10_%j.log
#SBATCH --error=autonovel_error_cifar10_%j.log
#SBATCH --partition=lyceum
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=109G
#SBATCH --time=60:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dk2g21@soton.ac.uk

source /local/software/conda/miniconda-py3-new/etc/profile.d/conda.sh
conda activate auto_novel

module load cuda/10.0
module load gcc/8.5.0

export CUDA_VISIBLE_DEVICES=0,1,2,3

DATASET_ROOT=/mainfs/scratch/dk2g21/AutoNovel/data/cifar10/
EXP_ROOT=/mainfs/scratch/dk2g21/AutoNovel/outputs/cifar10/
WARMUP_MODEL_DIR=/mainfs/scratch/dk2g21/AutoNovel/data/experiments/pretrained/supervised_learning/resnet_rotnet_cifar10.pth

python auto_novel.py \
        --dataset_root $DATASET_ROOT \
        --exp_root $EXP_ROOT \
        --warmup_model_dir $WARMUP_MODEL_DIR \
        --lr 0.1 \
        --gamma 0.1 \
        --weight_decay 1e-4 \
        --step_size 170 \
        --batch_size 128 \
        --epochs 200 \
        --rampup_length 50 \
        --rampup_coefficient 5.0 \
        --dataset_name cifar10 \
        --seed 0 \
        --model_name resnet_cifar10 \
        --mode train