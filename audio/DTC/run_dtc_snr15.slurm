#!/bin/bash
#SBATCH --job-name=dtc_snr15
#SBATCH --output=logs/dtc_snr15_%j.log
#SBATCH --error=logs/dtc_snr15_%j.err
#SBATCH --partition=lyceum
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dk2g21@soton.ac.uk

source /local/software/conda/miniconda-py3-new/etc/profile.d/conda.sh
conda activate musan_noise_env

module load cuda/10.2

export CUDA_VISIBLE_DEVICES=0,1,2,3

export OMP_NUM_THREADS=16

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python -u imagenet_DTC.py \
    --data_path datasets/ESC50-noisy-spectrograms/snr15 \
    --pretrained_path /mainfs/scratch/dk2g21/DTC/pretrained_weights/resnet101.pth \
    --batch_size 128 \
    --warmup_epochs 15 \
    --epochs 100 \
    --update_interval 15 \
    --aug twice \
    --temperature 0.5 \
    --label_smoothing 0.1 \
    --ema_decay 0.995 \
    --conf_thresh 0.02 \
    --num_workers 16
    