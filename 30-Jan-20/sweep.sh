#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00
#SBATCH --mem=250G
#SBATCH --output=logs/sweep-%A-%a.out
#SBATCH --error=logs/sweep-%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu

# wandb api key
source ~/.wandb_key

# Load modules
module load python

# Activate conda environment (optional)
mamba activate olmo2

PORT=25880

torchrun --master_port=$PORT --nproc_per_node=1 ./sweep.py