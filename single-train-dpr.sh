#!/bin/bash

# SCRIPT NAME: single-dpr.sh
#SBATCH --job-name=dpr-stat
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/cedar/sebel/DPR-fr/slurm_log/job_%j.out
#SBATCH --error=/home/cedar/sebel/DPR-fr/slurm_log/job_%j.err

source /home/cedar/sebel/miniconda3/bin/activate dpr
set -x
export PYTHONPATH=/home/cedar/sebel/miniconda3/envs/dpr/bin/python
# Launch our script
srun python train_dense_encoder.py \
    train_datasets=["/scratch/sebel/dpr-stat/insee_ref_dpr/train.json"] \
    dev_datasets=["/scratch/sebel/dpr-stat/insee_ref_dpr/dev.json"] \
    train=biencoder_local \
    output_dir="/scratch/sebel/dpr-stat/dpr-ckpt/" \
    encoder=camembert
