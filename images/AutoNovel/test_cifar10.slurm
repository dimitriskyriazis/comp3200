#!/bin/bash

#SBATCH --job-name=test_cifar10
#SBATCH --output=test_cifar10.out
#SBATCH --error=test_cifar10.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --nodelist=indigo51,indigo53,indigo54
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dk2g21@soton.ac.uk

module load cuda/10.0
module load gcc/8.5.0
module load conda

CUDA_VISIBLE_DEVICES=0 

python auto_novel.py --mode test --dataset_name cifar10 --model_name resnet_cifar10 --exp_root ./data/experiments/pretrained/
