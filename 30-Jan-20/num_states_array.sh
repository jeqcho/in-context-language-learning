#!/bin/bash
#SBATCH --job-name=scan_num_states
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00
#SBATCH --mem=250G
#SBATCH --output=logs/scan_num_states-%A-%a.out
#SBATCH --error=logs/scan_num_states-%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu
#SBATCH --array=1-7

# wandb api key
source ~/.wandb_key

# Load modules
module load python

# Activate conda environment (optional)
mamba activate olmo2

PORTS=()
START_PORT=26080
NUM_PORTS=7

for ((i=0; i<NUM_PORTS; i++)); do
    PORTS+=($((START_PORT + i)))
done

UPDATE_FREQS=(1 2 16 64 128 256 512)
INDEX=$((SLURM_ARRAY_TASK_ID - 1))

torchrun --master_port=${PORTS[$INDEX]} --nproc_per_node=1 ./train_hmm.py --num_emissions=100 --num_states=100 --seq_length=100 --batch_size=1024 --num_epoch=500 --update_freq=${UPDATE_FREQS[$INDEX]}