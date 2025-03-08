#!/bin/bash
#SBATCH --job-name=generate-data
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=0-35:00
#SBATCH --mem=250G
#SBATCH --output=logs/generate-data-%A-%a.out
#SBATCH --error=logs/generate-data-%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu
#SBATCH --array=1-1

# wandb api key
source ~/.wandb_key

# Load modules
module load python

# Activate conda environment (optional)
mamba activate olmo2

# To expose imports to hmm
export PYTHONPATH=$PYTHONPATH:/n/home07/jchooi/in-context-language-learning

PORTS=()
START_PORT=26122
NUM_PORTS=7

for ((i=0; i<NUM_PORTS; i++)); do
    PORTS+=($((START_PORT + i)))
done

INDEX=$((SLURM_ARRAY_TASK_ID-1))
PORT=${PORTS[$INDEX]}
BATCH_SIZES=(1024 1600 2048 3000 4096)

torchrun --master_port=$PORT --nproc_per_node=1 ./generate_data.py \
    --num_emissions=200 \
    --num_states=500 \
    --seq_length=100 \
    --batch_size=256 \
    --update_freq=32 \
    --num_epoch=1000 \
    --load_model_with_epoch=10 \
    --gen_seq_len=100 \
    --num_seq=2''000''000''000 \
    --gen_batch_size=${BATCH_SIZES[$INDEX]} \
    --permutate_emissions