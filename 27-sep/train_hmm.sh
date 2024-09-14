#!/bin/bash
#SBATCH --job-name=train-hmm
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00
#SBATCH --mem=80G
#SBATCH --output=logs/test-%A-%a.out
#SBATCH --error=logs/%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu
#SBATCH --array=1-6

# Load modules
module load python

# Activate conda environment (optional)
mamba activate olmo2

PORTS=()

for (( i=25770; i<=25770+50; i++ ))
do
    PORTS+=($i)
done

PORT=${PORTS[$SLURM_ARRAY_TASK_ID-1]}

torchrun --master_port=$PORT --nproc_per_node=1 ./train_hmm.py --task_id $SLURM_ARRAY_TASK_ID