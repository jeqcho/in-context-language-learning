#!/bin/bash
#SBATCH --job-name=5-scan-efficiency
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
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
for i in $(seq 25620 25645); do
  PORTS+=($i)
done
device_train_microbatch_sizes=(8 16 32 64 128 256 512 1024)
device_train_microbatch_size=${device_train_microbatch_sizes[$SLURM_ARRAY_TASK_ID]}
PORT=${PORTS[$SLURM_ARRAY_TASK_ID]}

torchrun --master_port=$PORT --nproc_per_node=1 ../scripts/train.py 5-scan-efficiency.yaml --save_overwrite=true --run_name=scan-microbatch-$device_train_microbatch_size --device_train_microbatch_size=$device_train_microbatch_size
