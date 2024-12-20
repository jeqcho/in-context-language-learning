#!/bin/bash
#SBATCH --job-name=train-hmm-explore-100
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=0-011:00
#SBATCH --mem=500G
#SBATCH --output=logs/train-hmm-explore-100-%A-%a.out
#SBATCH --error=logs/train-hmm-explore-100-%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu

# Load modules
module load python

# Activate conda environment (optional)
mamba activate olmo2

PORT=25840

torchrun --master_port=$PORT --nproc_per_node=1 ./train_hmm_explore_100.py