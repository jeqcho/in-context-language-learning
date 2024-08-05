#!/bin/bash
#SBATCH --job-name=22-zipf-permutate
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
#SBATCH --array=1-5

# Load modules
module load python

# Activate conda environment (optional)
mamba activate olmo2

for (( i=25690; i<=25690+50; i++ ))
do
    PORTS+=($i)
done

PORT=${PORTS[$SLURM_ARRAY_TASK_ID-1]}

if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    FLAG=""
    RUN_NAME="normal"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    FLAG="--data.custom_data_config.hmm_dataset_config.zipfian=True"
    RUN_NAME="zipfian"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
    FLAG="--data.custom_data_config.hmm_dataset_config.zipfian=True --data.custom_data_config.hmm_dataset_config.zipfian_scale=3"
    RUN_NAME="zipfian_scaled"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 4 ]; then
    FLAG="--data.custom_data_config.hmm_dataset_config.zipfian=True --data.custom_data_config.hmm_dataset_config.permutate=True"
    RUN_NAME="zipfian_permutated"
else
    FLAG="--data.custom_data_config.hmm_dataset_config.zipfian=True --data.custom_data_config.hmm_dataset_config.permutate=True --data.custom_data_config.hmm_dataset_config.zipfian_scale=3"
    RUN_NAME="zipfian_permutated_scaled"
fi


torchrun --master_port=$PORT --nproc_per_node=1 ../scripts/train.py model.yaml --run_name=$RUN_NAME-h-5-e-10 $FLAG