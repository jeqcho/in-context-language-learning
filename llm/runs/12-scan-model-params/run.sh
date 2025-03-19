#!/bin/bash
#SBATCH --job-name=scan-model-params
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=0-03:00
#SBATCH --mem=360G
#SBATCH --array=0-2
#SBATCH --output=logfiles/scan-model-params-%A-%a.out
#SBATCH --error=logfiles/scan-model-params-%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu

module load python
mamba activate olmo2

# Fixed number of chunks and chunk size
num_chunks=3
chunk_size=100000

echo "Running with fixed num_chunks=$num_chunks, chunk_size=$chunk_size"

# Define array of number of heads to scan over
# We'll use array index to select number of heads: [1, 3, 6]
declare -a n_heads=(1 3 6)
selected_n_heads=${n_heads[$SLURM_ARRAY_TASK_ID]}

echo "Running with n_head=$selected_n_heads (array task ${SLURM_ARRAY_TASK_ID})"

# Set environment variables for distributed training
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$((26057 + SLURM_ARRAY_TASK_ID))
export PYTHONPATH=$PYTHONPATH:/n/home07/jchooi/in-context-language-learning
export LOGLEVEL=INFO

echo "Master IP: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# Run training
torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /n/home07/jchooi/in-context-language-learning/llm/train.py \
    --config=run.yaml \
    --num_chunks=$num_chunks \
    --chunk_size=$chunk_size \
    --n_head=$selected_n_heads