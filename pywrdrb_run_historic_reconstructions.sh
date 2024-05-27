#!/bin/bash
#SBATCH --job-name=SimDRB
#SBATCH --output=SimDRB.out
#SBATCH --error=SimDRB.err
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=30

module load python
source venv/bin/activate

# Execute the Python script
# echo Prepping Obs-PUB reconstruction simulation input data...
mpirun -np $(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES)) python -u ./pywrdrb/prep_input_data_ensemble.py


# echo Running Obs-PUB reconstruction QPPQ aggregate simulations...
# for inflow_type in obs_pub_nhmv10_ObsScaled obs_pub_nwmv21_ObsScaled
# do
# 	echo Running simulation with $inflow_type ...
# 	python -W ignore ./pywrdrb/run_historic_simulation.py $inflow_type 'False'
# done


echo Running Obs-PUB reconstruction ensemble simulations...
# for inflow_type in obs_pub_nhmv10_ObsScaled_ensemble obs_pub_nwmv21_ObsScaled_ensemble
# do
# 	echo Running simulation with $inflow_type ...
# 	mpirun -np $(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES)) python -W ignore ./pywrdrb/run_historic_simulation.py $inflow_type 'True'

#     echo Combining batched output files...
#     python ./pywrdrb/combine_ensemble_results.py $inflow_type 'True'
# done

# Make figures
echo Making figures...
# time python ./pywrdrb/make_figs_historic_reconstruction_paper.py

echo Done!