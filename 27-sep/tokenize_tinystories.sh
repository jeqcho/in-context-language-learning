#!/bin/bash
#SBATCH --job-name=tokenize-tinystories
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00
#SBATCH --mem=80G
#SBATCH --output=logs/%A-%a.out
#SBATCH --error=logs/%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu

# Load modules
module load python

# Activate conda environment (optional)
mamba activate olmo2

PORT=25635

torchrun --master_port=$PORT --nproc_per_node=1 ./tokenize_tinystories.py