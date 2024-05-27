#!/bin/bash
#SBATCH --job-name=pywrWRF
#SBATCH --output=sim.out
#SBATCH --error=sim.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --exclusive

module load python
source venv/bin/activate

### prep inputs from raw data
echo Prepping WRF-Hydro simulation input data...
mpirun -np $(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES)) python -m mpi4py ./pywrdrb/prep_input_data_wrf_hydro.py

### run single-scenario simulations with different data sources
# for inflow_type in nhmv10_withObsScaled nwmv21_withObsScaled nhmv10 nwmv21
# do
# 	echo Running simulation with $inflow_type ...
# 	time python3 -W ignore ./pywrdrb/run_historic_simulation.py $inflow_type
# done

### analyze results, make figures
# echo Analyzing results...
# time python3 -W ignore ./pywrdrb/make_figs_diagnostics_paper.py
