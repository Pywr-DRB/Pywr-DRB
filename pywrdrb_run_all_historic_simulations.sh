#!/bin/bash

#### prep inputs from raw data
echo Prepping data...
time python3 -W ignore ./pywrdrb/prep_input_data.py

### run simulation using multiple inpu data sources
for inflow_type in WEAP_29June2023_gridmet nhmv10 nwmv21
do
	echo Running simulation with $inflow_type ...
	time python3 -W ignore ./pywrdrb/run_historic_simulation.py $inflow_type
done

### now run historic reconstructions under different assumptions
for fdc_type in nhmv10 nwmv21
do
	for use_NYCScaling in yes #no
	do
		#### Prep inputs from raw data
		echo Prepping data... $fdc_type $use_NYCScaling
		time python3 -W ignore ./pywrdrb/prep_full_reconstruction_input_data.py $fdc_type $use_NYCScaling

		#### Run simulation 
		echo Running full historic reconstruction simulation... $fdc_type $use_NYCScaling
		time python3 -W ignore ./pywrdrb/run_full_reconstruction.py $fdc_type $use_NYCScaling 
	done
done

### analyze results, make figures
#echo Analyzing results...
#time python3 -W ignore ./pywrdrb/drb_make_figs_diagnostics_paper.py
