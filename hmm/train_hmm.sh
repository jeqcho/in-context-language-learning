#!/bin/bash
#SBATCH --job-name=train_hmm
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=0-23:00
#SBATCH --mem=250G
#SBATCH --output=logs/train_hmm-%A-%a.out
#SBATCH --error=logs/train_hmm-%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu

# wandb api key
source ~/.wandb_key

# Load modules
module load python

# Activate conda environment (optional)
mamba activate olmo2

# To expose imports to hmm
export PYTHONPATH=$PYTHONPATH:/n/home07/jchooi/in-context-language-learning

PORT=25929

torchrun --master_port=$PORT --nproc_per_node=1 ./train_hmm.py \
    --num_emissions=200 \
    --num_states=500 \
    --seq_length=100 \
    --batch_size=256 \
    --update_freq=32 \
    --num_epoch=1000 \
    --save_epoch_freq=5