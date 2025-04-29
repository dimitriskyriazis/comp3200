#!/bin/bash
#SBATCH --job-name=wavpng
#SBATCH --output=logs/wavpng_%j.log
#SBATCH --error=logs/wavpng_%j.err
#SBATCH --partition=lyceum
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dk2g21@soton.ac.uk


source /local/software/conda/miniconda-py3-new/etc/profile.d/conda.sh
conda activate musan_noise_env

python wav_to_png.py