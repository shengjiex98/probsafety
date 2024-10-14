#!/bin/bash
#SBATCH --job-name=exp1_job
#SBATCH --output=results/exp1_%j.out
#SBATCH --error=results/exp1_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load required modules
module load python/3.9  # Load necessary modules based on your cluster

# Activate your virtual environment if needed
source ~/my_env/bin/activate

# Run your experiment script
python experiments/experiment1.py --config configs/exp1.yaml --output_dir results/exp1_$(date +%Y%m%d_%H%M)
