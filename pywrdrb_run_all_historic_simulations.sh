#!/bin/bash

#### rerun historic pub reconstruction if needed
cd ../DRB-Historic-Reconstruction/
python -W ignore generate_all_reconstructions.py
cd ../Pywr-DRB/

#### prep inputs from raw data
echo Prepping data...
time python3 -W ignore ./pywrdrb/prep_input_data.py

### run ensemble simulations across multiple processors using mpi4py (default: 6 cores)
for inflow_type in obs_pub_nwmv21_NYCScaled_ensemble obs_pub_nhmv10_NYCScaled_ensemble
do
	echo Running simulation with $inflow_type ...
	time mpirun -np 6 python3 -W ignore ./pywrdrb/run_historic_simulation.py $inflow_type True
	time python3 -W ignore ./pywrdrb/combine_ensemble_results.py $inflow_type
done

### run single-scenario simulations with different data sources
for inflow_type in obs_pub_nhmv10_NYCScaled obs_pub_nwmv21_NYCScaled nhmv10 nwmv21 WEAP_29June2023_gridmet
do
	echo Running simulation with $inflow_type ...
	time python3 -W ignore ./pywrdrb/run_historic_simulation.py $inflow_type
done

### analyze results, make figures
#echo Analyzing results...
#time python3 -W ignore ./pywrdrb/make_figs_diagnostics_paper.py
