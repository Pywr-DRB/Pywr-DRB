#!/bin/bash
#SBATCH --job-name=pywrdrb_policy        # Job name
#SBATCH --output=../logs/pywrdrb_%j.out  # Standard output log (%j = job ID)
#SBATCH --error=../logs/pywrdrb_%j.err   # Standard error log
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=40             # Number of tasks per node
#SBATCH --exclusive                      # Reserve node exclusively
#SBATCH --mail-type=END                  # Notify when job ends
#SBATCH --mail-user=ms3654@cornell.edu   # Your email

# === Ensure logs directory exists ===
mkdir -p ../logs

# === Load required module ===
module load python/3.11.5

# === Activate virtual environment ===
source /home/fs02/pmr82_0001/ms3654/envs/borg-env/bin/activate

# Define function to submit a single job iteration
submit_job() {

    datetime=$(date '+%Y-%m-%d %H:%M:%S')
    n_processors=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

    echo "[JobID $SLURM_JOB_ID] Starting pywrdrb simulation job"
    echo "Time       : $datetime"
    echo "Number of nodes: $SLURM_NNODES"
    echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
    echo "Working dir: $(pwd)"
    echo "Datetime: $datetime"
    echo "Total processors: $n_processors"

    # === Run your script ===
    python experiments/pywrdrb_policy_refactor.py
    echo "pywrdrb job complete."
    
    wait
}

