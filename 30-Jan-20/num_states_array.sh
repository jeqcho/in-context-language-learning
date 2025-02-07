#!/bin/bash
#SBATCH --job-name=scan_num_states
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00
#SBATCH --mem=250G
#SBATCH --output=logs/scan_num_states-%A-%a.out
#SBATCH --error=logs/scan_num_states-%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu
#SBATCH --array=1-6

# wandb api key
source ~/.wandb_key

# Load modules
module load python

# Activate conda environment (optional)
mamba activate olmo2

PORT=25868
NUM_STATES=(100 200 300 400 500 600)
BATCH_SIZES=(1024 256 128 128 64 64)
PORTS=(25868 25869 25870 25871 25872 25873)
INDEX=$((SLURM_ARRAY_TASK_ID - 1))

torchrun --master_port=${PORTS[$INDEX]} --nproc_per_node=1 ./train_hmm.py --num_emissions=100 --num_states=${NUM_STATES[$INDEX]} --seq_length=100 --batch_size=${BATCH_SIZES[$INDEX]} --num_epoch=50