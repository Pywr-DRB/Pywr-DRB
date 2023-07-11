#!/bin/bash

#### prep inputs from raw data
#echo Prepping data...
#time python3 -W ignore ./pywrdrb/prep_input_data.py

### run simulation using multiple input data sources
for inflow_type in obs_pub_nhmv10_NYCScaled obs_pub_nwmv21_NYCScaled #nhmv10 nwmv21 WEAP_29June2023_gridmet obs_pub_nhmv10_NYCScaled_ensemble obs_pub_nwmv21_NYCScaled_ensemble
do
	echo Running simulation with $inflow_type ...
	time python3 -W ignore ./pywrdrb/run_historic_simulation.py $inflow_type
done

### analyze results, make figures
#echo Analyzing results...
#time python3 -W ignore ./pywrdrb/make_figs_diagnostics_paper.py
