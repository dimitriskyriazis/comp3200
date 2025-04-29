#!/bin/bash
#SBATCH --job-name=test_loader
#SBATCH --output=logs/test_loader_%j.log
#SBATCH --error=logs/test_loader_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=lyceum
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dk2g21@soton.ac.uk

source /local/software/conda/miniconda-py3-new/etc/profile.d/conda.sh
conda activate musan_noise_env

python test_loader.py
