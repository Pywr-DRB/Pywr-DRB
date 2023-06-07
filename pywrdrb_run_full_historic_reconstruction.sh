#!/bin/bash

#### Prep inputs from raw data
echo Prepping data...
time python3 -W ignore ./pywrdrb/prep_full_reconstruction_input_data.py

#### Run simulation
echo Running full historic reconstruction simulation...
time python3 -W ignore ./pywrdrb/run_full_reconstruction.py

### Plot figures (TO-DO)