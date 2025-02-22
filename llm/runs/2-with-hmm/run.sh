#!/bin/bash
#SBATCH --job-name=llm-learns-hmm
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:2
#SBATCH --time=0-03:00
#SBATCH --mem=750G
#SBATCH --output=logfiles/llm-learns-hmm-%A-%a.out
#SBATCH --error=logfiles/llm-learns-hmm-%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu

# Load modules
module load python

# Activate conda environment (optional)
mamba activate olmo2

# Set master address and port
export MASTER_ADDR="127.0.0.0"
export PYTHONPATH=$PYTHONPATH:/n/home07/jchooi/in-context-language-learning

echo Node IP: $master
export LOGLEVEL=INFO

PORT=26022

torchrun --master_port=$PORT --nproc_per_node=1 /n/home07/jchooi/in-context-language-learning/llm/train.py run.yaml