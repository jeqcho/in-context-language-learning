#!/bin/bash
#SBATCH --job-name=scan-num-chunks
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=0-03:00
#SBATCH --mem=360G
#SBATCH --array=3-3
#SBATCH --output=logfiles/scan-num-chunks-%A-%a.out
#SBATCH --error=logfiles/scan-num-chunks-%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu

module load python
mamba activate olmo2

# Define number of chunks options
num_chunks_array=(1 2 3 5)
chunk_size=100000

# Select number of chunks based on array task ID
num_chunks=${num_chunks_array[$SLURM_ARRAY_TASK_ID]}

echo "Running with num_chunks=$num_chunks, chunk_size=$chunk_size"

# Set environment variables for distributed training
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$((26077 + SLURM_ARRAY_TASK_ID))
export PYTHONPATH=$PYTHONPATH:/n/home07/jchooi/in-context-language-learning
export LOGLEVEL=INFO

echo "Master IP: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Testing num_chunks: ${num_chunks}"

# Run training
torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /n/home07/jchooi/in-context-language-learning/llm/train.py \
    --config=run.yaml \
    --num_chunks=$num_chunks \
    --chunk_size=$chunk_size