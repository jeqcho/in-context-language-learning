#!/bin/bash
#SBATCH --job-name=14-zipfian
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30
#SBATCH --mem=128G
#SBATCH --output=logs/%A-%a.out
#SBATCH --error=logs/%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu

# Load modules
module load python

# Activate conda environment (optional)
mamba activate olmo2

Hs=(5 5 5 10 10 10)
Es=(3 4 5 5 7 10)
H=${Hs[$SLURM_ARRAY_TASK_ID-1]}
E=${Es[$SLURM_ARRAY_TASK_ID-1]}
PORTS=()

for (( i=25620; i<=25620+50; i++ ))
do
    PORTS+=($i)
done

PORT=25640

torchrun --master_port=$PORT --nproc_per_node=1 ../scripts/train.py model.yaml