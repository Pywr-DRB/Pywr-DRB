#!/bin/bash

#### prep inputs from raw data
echo Prepping data...
time python3 -W ignore ./pywrdrb/prep_input_data.py

#### run simulation using multiple inpu data sources
for inflow_type in WEAP_24Apr2023_gridmet obs_pub nhmv10 nwmv21
do
	echo Running simulation with $inflow_type ...
	time python3 -W ignore ./pywrdrb/run_historic_simulation.py $inflow_type
done

### analyze results, make figures
echo Analyzing results...
time python3 -W ignore ./pywrdrb/drb_make_figs.py
