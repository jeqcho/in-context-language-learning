#!/bin/bash
#SBATCH --job-name=llama-on-h-500-e-200
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:2
#SBATCH --time=0-01:00
#SBATCH --mem=720G
#SBATCH --array=0-8%3  # Run 3 jobs at a time (out of 9 total combinations)
#SBATCH --output=logfiles/llama-on-h-500-e-200-%A-%a.out
#SBATCH --error=logfiles/llama-on-h-500-e-200-%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu

# Load modules
module load python

# Activate conda environment (optional)
mamba activate olmo2

# Define arrays for parameters
num_chunks_array=(3 5 7)
chunk_size_array=(100000 200000 300000)

# Calculate indices for current array task
chunk_idx=$((SLURM_ARRAY_TASK_ID / 3))
size_idx=$((SLURM_ARRAY_TASK_ID % 3))

# Get values for this run
num_chunks=${num_chunks_array[$chunk_idx]}
chunk_size=${chunk_size_array[$size_idx]}

echo "Running with num_chunks=$num_chunks, chunk_size=$chunk_size"

# Set proper environment variables for distributed training
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$((26037 + SLURM_ARRAY_TASK_ID))  # Unique port for each array job
export PYTHONPATH=$PYTHONPATH:/n/home07/jchooi/in-context-language-learning

echo "Master IP: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
export LOGLEVEL=INFO

# Run with parameter combinations
torchrun \
    --nproc_per_node=2 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nnodes=1 \
    /n/home07/jchooi/in-context-language-learning/llm/train.py \
    --config=run.yaml \
    --num_chunks=$num_chunks \
    --chunk_size=$chunk_size