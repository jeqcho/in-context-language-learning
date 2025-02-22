#!/bin/bash
#SBATCH --job-name=generate-data
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=0-10:00
#SBATCH --mem=250G
#SBATCH --output=logs/generate-data-%A-%a.out
#SBATCH --error=logs/generate-data-%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu

# wandb api key
source ~/.wandb_key

# Load modules
module load python

# Activate conda environment (optional)
mamba activate olmo2

PORT=25944

torchrun --master_port=$PORT --nproc_per_node=1 ./generate_data.py \
    --num_emissions=200 \
    --num_states=200 \
    --seq_length=100 \
    --batch_size=1024 \
    --update_freq=32 \
    --num_epoch=1000 \
    --load_model_with_epoch=50 \
    --gen_seq_len=100 \
    --num_seq=100''000''000 \
    --permutate_emissions