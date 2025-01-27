#!/bin/bash
#SBATCH --job-name=test-1-gpu
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=0-05:00
#SBATCH --mem=250G
#SBATCH --output=logs/test-1-gpu-%A-%a.out
#SBATCH --error=logs/test-1-gpu-%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu

# Load modules
module load python

# Activate conda environment (optional)
mamba activate olmo2

PORT=25845

torchrun --master_port=$PORT --nproc_per_node=1 ./train_hmm.py