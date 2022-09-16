#!/bin/bash

# SCRIPT NAME: multi-dpr.sh
#SBATCH --job-name=dpr-stat
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/home/cedar/sebel/DPR-fr/slurm_log/job_%j.out
#SBATCH --error=/home/cedar/sebel/DPR-fr/slurm_log/job_%j.err

source /home/cedar/sebel/miniconda3/bin/activate dpr
set -x
export PYTHONPATH=/home/cedar/sebel/miniconda3/envs/dpr/bin/python
# Launch our script
srun python -m torch.distributed.launch --nproc_per_node=1 \
    generate_dense_embeddings.py \
    model_file=/scratch/sebel/dpr-stat/dpr-ckpt/dpr_biencoder.99 \
    ctx_src=dpr_insee \
    out_file=/scratch/sebel/dpr-stat/dpr-emb/dpr-insee/
