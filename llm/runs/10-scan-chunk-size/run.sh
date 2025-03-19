#!/bin/bash
#SBATCH --job-name=scan-chunk-size
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=0-03:00
#SBATCH --mem=360G
#SBATCH --array=0-2
#SBATCH --output=logfiles/scan-chunk-size-%A-%a.out
#SBATCH --error=logfiles/scan-chunk-size-%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu

module load python
mamba activate olmo2

# Define chunk size options
chunk_size_array=(50000 100000 150000)
num_chunks=3

# Select chunk size based on array task ID
chunk_size=${chunk_size_array[$SLURM_ARRAY_TASK_ID]}

echo "Running with num_chunks=$num_chunks, chunk_size=$chunk_size"

# Set environment variables for distributed training
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$((26047 + SLURM_ARRAY_TASK_ID))
export PYTHONPATH=$PYTHONPATH:/n/home07/jchooi/in-context-language-learning
export LOGLEVEL=INFO

echo "Master IP: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Testing chunk size: ${chunk_size}"

# Run training
torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /n/home07/jchooi/in-context-language-learning/llm/train.py \
    --config=run.yaml \
    --num_chunks=$num_chunks \
    --chunk_size=$chunk_size