#!/bin/bash
#SBATCH --job-name=40M-llm-run-2-gpu
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --time=0-23:00
#SBATCH --mem=720G
#SBATCH --output=logfiles/40M-llm-run-2-gpu-%j.out
#SBATCH --error=logfiles/40M-llm-run-2-gpu-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu

module load python
mamba activate olmo2

# Set environment variables for distributed training
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=26081
export PYTHONPATH=$PYTHONPATH:/n/home07/jchooi/in-context-language-learning
export LOGLEVEL=INFO

echo "Master IP: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"

# Run training
torchrun \
    --nproc_per_node=2 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /n/home07/jchooi/in-context-language-learning/llm/train.py \
    --config=run.yaml \
    --run_name="40M-llm-run-2-gpu"