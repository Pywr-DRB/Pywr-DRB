#!/bin/bash
#SBATCH --job-name=starfit_sensitivity_analysis
#SBATCH --output=starfit_sensitivity_analysis.out
#SBATCH --error=starfit_sensitivity_analysis.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --exclusive

module load python 
source venv/bin/activate

export PYTHONPATH=$PYTHONPATH:/home/fs02/pmr82_0001/ms3654/Research/Pywr-DRB/Pywr-DRB

### Step 1: Create sensitivity analysis samples
#echo "Creating sensitivity analysis samples..."
#time python3 -W ignore ./starfit_sensitivity/make_sensitivity_samples.py

### Step 2: Run sensitivity analysis
echo "Running sensitivity analysis..."
time python3 -W ignore ./starfit_sensitivity/run_starfit_sensitivity.py