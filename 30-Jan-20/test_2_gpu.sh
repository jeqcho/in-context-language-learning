#!/bin/bash
#SBATCH --job-name=test-2-gpu
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --time=0-00:10
#SBATCH --mem=500G
#SBATCH --output=logs/test-2-gpu-%A-%a.out
#SBATCH --error=logs/test-2-gpu-%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu

# Load modules
module load python

# Activate conda environment (optional)
mamba activate olmo2

PORT=25840

torchrun --master_port=$PORT --nproc_per_node=1 ./train_hmm.py