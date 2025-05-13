#!/bin/bash
#SBATCH --job-name=pywrdrb           # Job name
#SBATCH --output=../logs/pywrdrb_%j.out  # Standard output log file with job ID
#SBATCH --error=../logs/pywrdrb_%j.err   # Standard error log file with job ID
#SBATCH --nodes=1                          # Number of nodes to use
#SBATCH --ntasks-per-node=40                # Number of tasks (processes) per node
#SBATCH --exclusive                        # Use the node exclusively for this job
#SBATCH --mail-type=END                    # Send email at job end

# Remember to create ../logs/ first!

# Load Python module
module load python/3.11.5

# Activate Python virtual environment
source ~/VEnvs/drb/bin/activate

# Function to submit the job
submit_job() {
    # local seed=$1
    # Print start message and the number of nodes and tasks per node
    datetime=$(date '+%Y-%m-%d %H:%M:%S')
    n_processors=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

    echo "[JobID $SLURM_JOB_ID] Running sa simulation ..."
    echo "Number of nodes: $SLURM_NNODES"
    echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
    echo "Total number of processors: $n_processors"
    echo "Datetime: $datetime"

    # Run the metrics script and wait for it to finish
    python pywrdrb_profiling.py

    # Wait for both background processes to finish
    wait
}

submit_job