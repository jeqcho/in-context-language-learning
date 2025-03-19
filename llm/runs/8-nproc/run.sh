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
#SBATCH --array=0-1%2  # Run 2 jobs at a time (for nproc 1 and 2)
#SBATCH --output=logfiles/llama-on-h-500-e-200-%A-%a.out
#SBATCH --error=logfiles/llama-on-h-500-e-200-%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu

# Load modules
module load python

# Activate conda environment (optional)
mamba activate olmo2

# Define arrays for parameters
num_chunks_array=(7)
chunk_size_array=(100000)
nproc_array=(1 2)  # Sweep over nproc 1 and 2

# Get nproc value for this run
nproc=${nproc_array[$SLURM_ARRAY_TASK_ID]}

# Use fixed values for chunks and size
num_chunks=${num_chunks_array[0]}
chunk_size=${chunk_size_array[0]}

echo "Running with num_chunks=$num_chunks, chunk_size=$chunk_size, nproc=$nproc"

# Set proper environment variables for distributed training
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$((26037 + SLURM_ARRAY_TASK_ID))  # Unique port for each array job
export PYTHONPATH=$PYTHONPATH:/n/home07/jchooi/in-context-language-learning

echo "Master IP: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Number of processes: $nproc"
export LOGLEVEL=INFO

# Run with parameter combinations
torchrun \
    --nproc_per_node=$nproc \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /n/home07/jchooi/in-context-language-learning/llm/train.py \
    --config=run.yaml \
    --num_chunks=$num_chunks \
    --chunk_size=$chunk_size