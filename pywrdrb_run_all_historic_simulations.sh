#!/bin/bash

#### prep inputs from raw data
echo Prepping data...
time python3 -W ignore ./pywrdrb/prep_input_data.py

### run simulation using multiple input data sources
for inflow_type in WEAP_29June2023_gridmet nhmv10 nwmv21 obs_pub_nhmv10_NYCScaling obs_pub_nwmv21_NYCScaling
do
	echo Running simulation with $inflow_type ...
	time python3 -W ignore ./pywrdrb/run_historic_simulation.py $inflow_type
done

### analyze results, make figures
#echo Analyzing results...
#time python3 -W ignore ./pywrdrb/make_figs_diagnostics_paper.py
