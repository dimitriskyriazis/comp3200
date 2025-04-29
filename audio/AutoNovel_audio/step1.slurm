#!/bin/bash
#SBATCH --job-name=ESC50_Step1_Precompute
#SBATCH --output=logs/step1/step1_%A_%a.log
#SBATCH --error=logs/step1/step1_%A_%a.err
#SBATCH --partition=lyceum
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-5
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dk2g21@soton.ac.uk

source /local/software/conda/miniconda-py3-new/etc/profile.d/conda.sh
conda activate autonovel_noise

module load cuda/10.0
module load gcc/8.5.0

export CUDA_VISIBLE_DEVICES=0,1,2,3

NOISE_TYPES=("gaussian" "gaussian" "gaussian" "saltpepper" "saltpepper" "saltpepper")
PARAMS=("5" "15" "30" "001" "005" "010")
STD_LIST=(5 15 30 - - -)
SP_LIST=(- - - 0.01 0.05 0.1)
FOLDERS=("gaussian_5" "gaussian_15" "gaussian_30" "saltpepper_001" "saltpepper_005" "saltpepper_010")

IDX=$SLURM_ARRAY_TASK_ID
ROOT=/mainfs/scratch/dk2g21/AutoNovel
INPUT=$ROOT/datasets/ESC-50-spectrograms/clean
OUT=$ROOT/datasets/ESC-50-spectrograms/${FOLDERS[$IDX]}

NOISE=${NOISE_TYPES[$IDX]}
STD=${STD_LIST[$IDX]}
SP=${SP_LIST[$IDX]}

echo "Generating $NOISE noise for ${FOLDERS[$IDX]}"
if [ "$NOISE" == "gaussian" ]; then
    python precompute_spectrograms.py --input_dir $INPUT --output_dir $OUT --noise_type gaussian --std $STD
else
    python precompute_spectrograms.py --input_dir $INPUT --output_dir $OUT --noise_type saltpepper --sp_ratio $SP
fi
