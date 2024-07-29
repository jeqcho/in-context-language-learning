#!/bin/bash
#SBATCH --job-name=12-markov-baseline
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30
#SBATCH --mem=128G
#SBATCH --output=logs/%A-%a.out
#SBATCH --error=logs/%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu


# Load modules
module load python

# Activate conda environment (optional)
mamba activate olmo2

# EMITS=(3 5 7 10 15)
# EMIT=${EMITS[$SLURM_ARRAY_TASK_ID]}

torchrun --master_port=25612 --nproc_per_node=1 ../scripts/train.py model.yaml --save_overwrite=true
