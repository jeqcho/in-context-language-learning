#!/bin/bash
#SBATCH --job-name=1-gpu-1-nproc
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00
#SBATCH --mem=720G
#SBATCH --array=0-1%2  # Run 2 jobs at a time (for nproc 1 and 2)
#SBATCH --output=logfiles/1-gpu-1-nproc-%A-%a.out
#SBATCH --error=logfiles/1-gpu-1-nproc-%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu

# Load modules
module load python

# Activate conda environment (optional)
mamba activate olmo2

# Define arrays for parameters
num_chunks_array=(7)
chunk_size_array=(100000)

# Use fixed values for chunks and size
num_chunks=${num_chunks_array[0]}
chunk_size=${chunk_size_array[0]}

echo "Running with num_chunks=$num_chunks, chunk_size=$chunk_size"

# Set proper environment variables for distributed training
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$((26047 + SLURM_ARRAY_TASK_ID))  # Unique port for each array job
export PYTHONPATH=$PYTHONPATH:/n/home07/jchooi/in-context-language-learning

echo "Master IP: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
export LOGLEVEL=INFO

# Run with parameter combinations
torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /n/home07/jchooi/in-context-language-learning/llm/train.py \
    --config=run.yaml \
    --num_chunks=$num_chunks \
    --chunk_size=$chunk_size