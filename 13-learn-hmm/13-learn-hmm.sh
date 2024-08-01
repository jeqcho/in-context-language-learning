#!/bin/bash
#SBATCH --job-name=13-learn-hmm
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
#SBATCH --array=1-1

# Load modules
module load python

# Activate conda environment (optional)
mamba activate olmo2

Hs=(5 5 5 10 10 10)
Es=(3 4 5 5 7 10)
H=${Hs[$SLURM_ARRAY_TASK_ID-1]}
E=${Es[$SLURM_ARRAY_TASK_ID-1]}
PORTS=()

for (( i=25650; i<=25650+50; i++ ))
do
    PORTS+=($i)
done

PORT=${PORTS[$SLURM_ARRAY_TASK_ID-1]}

torchrun --master_port=$PORT --nproc_per_node=1 ../scripts/train.py model.yaml --data.custom_data_config.hmm_dataset_config.num_hidden_states=$H --data.custom_data_config.hmm_dataset_config.num_symbols=$E --run_name=h-$H-e-$E
