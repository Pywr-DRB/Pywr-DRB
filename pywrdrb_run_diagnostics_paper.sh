#!/bin/bash

#### prep inputs from raw data
echo Prepping data...
time python3 -W ignore ./pywrdrb/prep_input_data.py

### run single-scenario simulations with different data sources
for inflow_type in nhmv10_withObsScaled nwmv21_withObsScaled nhmv10 nwmv21 #WEAP_29June2023_gridmet
do
	echo Running simulation with $inflow_type ...
	time python3 -W ignore ./pywrdrb/run_historic_simulation.py $inflow_type
done

## analyze results, make figures
echo Analyzing results...
time python3 -W ignore ./pywrdrb/make_figs_diagnostics_paper.py
