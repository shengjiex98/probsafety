#!/bin/bash
#SBATCH --job-name=probsafety_1dscan
#SBATCH --output=results/1dscan_%j.out
#SBATCH --error=results/1dscan_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=257
#SBATCH --mem=16G

# Load required modules
module load python/3.12  # Load necessary modules based on your cluster

# Activate your virtual environment if needed
conda activate ipy

# Run your experiment script
python experiments/1dscan.py --config configs/1dscan.toml --output_dir results/1dscan_$(date +%Y%m%d_%H%M)
