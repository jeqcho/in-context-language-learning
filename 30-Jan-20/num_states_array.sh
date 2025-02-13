#!/bin/bash
#SBATCH --job-name=scan_num_states
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00
#SBATCH --mem=250G
#SBATCH --output=logs/scan_num_states-%A-%a.out
#SBATCH --error=logs/scan_num_states-%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu
#SBATCH --array=1-5

# wandb api key
source ~/.wandb_key

# Load modules
module load python

# Activate conda environment (optional)
mamba activate olmo2

PORTS=()
START_PORT=26110
NUM_PORTS=7

for ((i=0; i<NUM_PORTS; i++)); do
    PORTS+=($((START_PORT + i)))
done

UPDATE_FREQS=(1 2 8 16 32)
INDEX=$((SLURM_ARRAY_TASK_ID - 1))

torchrun --master_port=${PORTS[$INDEX]} --nproc_per_node=1 ./train_hmm.py --num_emissions=200 --num_states=500 --seq_length=100 --batch_size=256 --num_epoch=500 --update_freq=${UPDATE_FREQS[$INDEX]}