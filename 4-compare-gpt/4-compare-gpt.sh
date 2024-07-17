#!/bin/bash
#SBATCH --job-name=4-compare-gpt
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=0-00:30
#SBATCH --mem=128G
#SBATCH --output=logs/%A-%a.out
#SBATCH --error=logs/%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu
#SBATCH --array=1-8

# Load modules
module load python

# Activate conda environment (optional)
mamba activate olmo2

# Initialize an empty array
PORTS=()

# Populate the array with numbers from 12 to 90
for i in $(seq 25650 25685); do
  PORTS+=($i)
done

nworkers=(2 2 4 4 4 4 8 8)
prefetchs=(2 8 2 4 8 16 2 8)
nworker=${nworkers[$SLURM_ARRAY_TASK_ID-1]}
prefetch=${prefetchs[$SLURM_ARRAY_TASK_ID-1]}

PORT=${PORTS[$SLURM_ARRAY_TASK_ID]}

torchrun --master_port=$PORT --nproc_per_node=2 ../scripts/train.py 4-compare-gpt.yaml --save_overwrite=true --run_name=big-model-2-gpus-nworker${nworker}-prefetch${prefetch} --data.num_workers=$nworker --data.prefetch_factor=$prefetch
