#!/bin/bash
#SBATCH --job-name=SimDRB
#SBATCH --output=SimDRB.out
#SBATCH --error=SimDRB.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=33


module load python/3.11.5
source venv/bin/activate
export PYTHONPATH=$(dirname "$0")

# Number of processors
np=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

ensemble_datasets=(
	'obs_pub_nhmv10_BC_ObsScaled_ensemble'
)

### Prepare input data
for inflow_type in "${ensemble_datasets[@]}"
do
	echo Prepping simulation input data for $inflow_type...
	mpirun -np $np python3 -u ./pywrdrb/prep_input_data_ensemble.py  $inflow_type
done

# echo Running Obs-PUB reconstruction QPPQ aggregate simulations...
# for inflow_type in obs_pub_nhmv10_ObsScaled obs_pub_nwmv21_ObsScaled
# do
# 	echo Running simulation with $inflow_type ...
# 	python -W ignore ./pywrdrb/run_historic_simulation.py $inflow_type 'False'
# done


echo Running Obs-PUB reconstruction ensemble simulations...
for inflow_type in "${ensemble_datasets[@]}"
do
	echo Running simulation with $inflow_type ...
	mpirun -np $np python3 -W ignore ./pywrdrb/run_historic_simulation.py $inflow_type 'True'

    echo Combining batched output files...
    python3 ./pywrdrb/combine_ensemble_results.py $inflow_type 'True'
done

# Make figures
# echo Making figures...
# time python ./pywrdrb/make_figs_historic_reconstruction_paper.py

echo DONE!
