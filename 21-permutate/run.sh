#!/bin/bash
#SBATCH --job-name=21-permutate
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00
#SBATCH --mem=375G
#SBATCH --output=logs/%A-%a.out
#SBATCH --error=logs/%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu
#SBATCH --array=1-8

# Load modules
module load python

# Activate conda environment (optional)
mamba activate olmo2

Hs=(3 4 5 6 7 8 10 15)
H=${Hs[$SLURM_ARRAY_TASK_ID-1]}

for (( i=25650; i<=25650+50; i++ ))
do
    PORTS+=($i)
done

PORT=${PORTS[$SLURM_ARRAY_TASK_ID-1]}

torchrun --master_port=$PORT --nproc_per_node=1 ../scripts/train.py model.yaml --run_name=h-$H-e-5 --data.custom_data_config.hmm_dataset_config.num_hidden_states=$H