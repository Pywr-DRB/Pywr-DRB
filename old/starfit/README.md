### starfit_experiments

This folder contains various scripts performing different exploratory analyses of the STARFIT and DRB models, so I've written a separate README to explain the contents.

The ```simulate_reservoir_daily``` script contains the most up-to-date copy of the explicit STARFIT simulation.

The ```compare_model_behaviors``` script takes output data from pywr, and compares (plots) simulated storage and releases between the two models. Also, the inflow/outflow are compared to ResOpsUS data, and downstream USGS gage data.

The ```test_reservoir_simulation``` script performs basic tests of the explicit model (simulate_reservoir_daily); plots storage, NOR, and release patterns.


## starfit_sensitivity_analysis

This folder was used with the SALib to conduct a sensitivity analysis of all 19 model parameters.

The ```find_starfit_param_bounds``` script uses the entire ```ISTARF-CONUS.csv``` database to produce distributions of parameter values (starfit_parameter_ranges.png).

The ```reservoir_models_for_SA``` contains the same explicit STARFIT simulation used elsewhere, except the output is now the % of time spent inside the NOR.

The ```starfit_SA``` performs the actual SA.


## /ResOpsUS_data

This folder is used to store ResOpsUS data, which are large when downloaded, they contain:

1. DAILY_AV_INFLOW_CUMECS.csv,
2. DAILY_AV_OUTFLOW_CUMECS.csv,
3.  DAILY_AV_STORAGE_MCM.csv.

  (1) DAILY_AV_INFLOW_CUMECS.csv contains daily records for all the dams (columns) from 1980-01-01 to 2020-12-31. All the records are in cubic meters per second.

  (2) DAILY_AV_OUTFLOW_CUMECS.csv contains daily records for all the dams (columns) from 1980-01-01 to 2020-12-31. All the records are in cubic meters per second.

  (3) DAILY_AV_STORAGE_MCM.csv contains daily records for all the dams (columns) from 1980-01-01 to 2020-12-31. All the records are in million cubic meters.

Download the files, then run the ```get_ResOps_data.py``` to extract data for the reservoirs within the DRB (Blue Marsh and Beltzville).

Once the above is run, output csv files will be ```resops_<reservoir_name>.csv```, and will have ```inflow, outflow, storage``` as columns.


## usgs_data

This folder is used to store and process usgs gage data.

The ```process_usgs_data``` script takes gauge IDs, downloads .txt files, then takes the .txt files and produces cleaned .csv files.  The resulting files have names clean_usgs_<gauge_id>.csv.

This was a modified version of Andrew's original script which did something similar. 
