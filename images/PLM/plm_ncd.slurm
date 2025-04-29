#!/bin/bash
#SBATCH --job-name=plm_c10_pair_02
#SBATCH --output=plm_output_cifar10_pair_02log
#SBATCH --error=plm_error_cifar10_pair_02.log
#SBATCH --partition=lyceum
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dk2g21@soton.ac.uk

module load cuda/11.3

source /local/software/conda/miniconda-py3-new/etc/profile.d/conda.sh
conda activate ncd_noisy

nvidia-smi

export CUDA_VISIBLE_DEVICES=0,1

python lnl.py --ds cifar-10 --data_gen pair --flip_rate 0.2 --ncd_mode --data_root /mainfs/scratch/dk2g21/PLM/data/cifar10/ --save_dir /mainfs/scratch/dk2g21/PLM/res/