#!/bin/bash

# SCRIPT NAME: gen_dense.sh
#SBATCH --job-name=dpr-gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --output=/home/cedar/sebel/DPR-fr/slurm_log/gen_dense/job_%j.out
#SBATCH --error=/home/cedar/sebel/DPR-fr/slurm_log/gen_dense/job_%j.err

source /home/cedar/sebel/miniconda3/bin/activate dpr
set -x
export PYTHONPATH=/home/cedar/sebel/miniconda3/envs/dpr/bin/python
# Launch our script
srun python generate_dense_embeddings.py \
    model_file=/scratch/sebel/dpr-stat/dpr-ckpt/dpr_biencoder.99 \
    ctx_src=dpr_insee \
    shard_id=0 num_shards=1 \
    out_file=/scratch/sebel/dpr-stat/dpr-emb/dpr-insee/
