#!/bin/bash

### prep inputs from raw data
echo Prepping data...
time python3 -W ignore ./pywrdrb/prep_input_data.py

### run simulation using multiple inpu data sources
for inflow_type in obs_pub nhmv10 nwmv21 #WEAP_23Aug2022_gridmet
do
	backup_inflow_type=nhmv10
	echo Running simulation with $inflow_type ...
	time python3 -W ignore ./pywrdrb/drb_run_sim.py $inflow_type $backup_inflow_type
done

### analyze results, make figures
echo Analyzing results...
time python3 -W ignore ./pywrdrb/drb_make_figs.py 
