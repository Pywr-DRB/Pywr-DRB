#!/bin/bash

### loop over two fdc doner types for constructing PUB ensemble
for fdc_doner_type in nhmv10 nwmv21
do
	#### prep inputs from raw data
	echo Prepping data...
	time python3 -W ignore ./pywrdrb/prep_ensemble_input_data.py $fdc_doner_type

	### run simulations using historical PUB ensemble
	echo Running ensemble...
	time python3 -W ignore ./pywrdrb/run_ensemble_simulation.py
done
