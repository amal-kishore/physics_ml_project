#!/bin/bash
#SBATCH --job-name=physics_ml               # Job name
#SBATCH --partition=dgx                     # GPU partition
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks=1                          # Single task
#SBATCH --gres=gpu:1                        # Request 1 GPU
#SBATCH --cpus-per-task=20                  # Adjust based on available CPUs
#SBATCH --mem=128G                          # Total memory (adjust as needed)
#SBATCH --time=48:00:00                     # Time limit (48 hours)
#SBATCH --output=physics_ml_%j.out          # Standard output
#SBATCH --error=physics_ml_%j.err           # Error log

# Load environment variables manually if module command is unavailable
export MASTER_ADDR="localhost"               # Use localhost if hostname command fails
export MASTER_PORT=12345                     # Choose an open port for communication

# Activate virtual environment
source /lustre/vishal.bharti/amal/physics_ml_project/amal_mp_project_venv/bin/activate

# Run the Python script
python /lustre/vishal.bharti/amal/physics_ml_project/launch.py

