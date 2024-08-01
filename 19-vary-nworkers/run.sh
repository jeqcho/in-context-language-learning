#!/bin/bash
#SBATCH --job-name=19-vary-nworkers
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --time=0-08:00
#SBATCH --mem=1500G
#SBATCH --output=logs/%A-%a.out
#SBATCH --error=logs/%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu
#SBATCH --array=1-5%2

# Load modules
module load python

# Activate conda environment (optional)
mamba activate olmo2

MICROBATCHs=(1 2 4 6 8)
PREFETCHs=(2 4 8 16 32)
NUMWORKERS=(8 16 32 64 72)

M=4
P=8
N=${NUMWORKERS[$SLURM_ARRAY_TASK_ID-1]}
PORTS=()

for (( i=25630; i<=25630+50; i++ ))
do
    PORTS+=($i)
done

PORT=25640

torchrun --master_port=$PORT --nproc_per_node=4 ../scripts/train.py model.yaml --run_name=m-$M-p-$P-n-$N --device_train_microbatch_size=$M --data.prefetch_factor=$P --data.num_workers=$N