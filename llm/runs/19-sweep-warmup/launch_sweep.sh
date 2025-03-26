#!/bin/bash
#SBATCH --job-name=llm-sweep
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --time=0-23:00
#SBATCH --mem=720G
#SBATCH --output=logfiles/sweep-%j.out
#SBATCH --error=logfiles/sweep-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jeqin_chooi+slurm@college.harvard.edu

module load python
mamba activate olmo2

# Set environment variables for distributed training
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=26081
export PYTHONPATH=$PYTHONPATH:/n/home07/jchooi/in-context-language-learning
export LOGLEVEL=INFO

echo "Master IP: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"

# Initialize the sweep with project name and capture output
echo "Initializing sweep..."
sweep_output=$(wandb sweep --project testing-llama sweep.yaml 2>&1)
echo "Sweep initialization output:"
echo "$sweep_output"

# Extract sweep ID from output
sweep_id=$(echo "$sweep_output" | grep -oP '(?<=Created sweep with ID: ).*')

if [ -z "$sweep_id" ]; then
    echo "Failed to get sweep ID. Full output:"
    echo "$sweep_output"
    exit 1
fi

echo "Starting sweep with ID: $sweep_id"

# Launch the sweep agent with project name
wandb agent --project testing-llama $sweep_id 