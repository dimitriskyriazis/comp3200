#!/bin/bash
#SBATCH --job-name=ESC50_Step4_Joint
#SBATCH --output=logs/step4/step4_%A_%a.log
#SBATCH --error=logs/step4/step4_%A_%a.err
#SBATCH --partition=lyceum
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-6
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dk2g21@soton.ac.uk

source /local/software/conda/miniconda-py3-new/etc/profile.d/conda.sh
conda activate autonovel_noise

module load cuda/10.0
module load gcc/8.5.0

export CUDA_VISIBLE_DEVICES=0

FOLDERS=("clean" "gaussian_5" "gaussian_15" "gaussian_30" "saltpepper_001" "saltpepper_005" "saltpepper_010")

IDX=$SLURM_ARRAY_TASK_ID
FOLDER=${FOLDERS[$IDX]}

ROOT=/mainfs/scratch/dk2g21/AutoNovel
python auto_novel.py \
    --spectrogram_root $ROOT/datasets/ESC-50-spectrograms/$FOLDER \
    --exp_root $ROOT/outputs/esc50 \
    --pretrained_model $ROOT/outputs/esc50/supervised_learning/resnet_esc50_sup_$FOLDER.pth \
    --model_name resnet_esc50_joint_$FOLDER \
    --num_labeled_classes 25 \
    --num_unlabeled_classes 25 \
    --batch_size 16 \
    --epochs 100 \
    --num_workers 4 \
    --rampup_coefficient 10 \
    --temperature 0.5 \
    --mse_weight 2.0 \
    --bce_weight 1.0
