#!/bin/bash
#SBATCH --job-name=train-hmm-vary-e-200
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=0-07:00
#SBATCH --mem=500G
#SBATCH --output=logs/train-hmm-vary-e-200-%A-%a.out
#SBATCH --error=logs/train-hmm-vary-e-200-%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu
#SBATCH --array=0-4

# Load modules
module load python

# Activate conda environment (optional)
mamba activate olmo2

PORTS=()
for ((i=0; i<10; i++)); do
    PORTS+=($((25940 + i)))
done
PORT=${PORTS[$SLURM_ARRAY_TASK_ID]}

torchrun --master_port=$PORT --nproc_per_node=1 ./train_hmm_vary_e_200.py $SLURM_ARRAY_TASK_ID