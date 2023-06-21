"""
This script is used to run a Pywr-DRB simulation with an ensemble of inflow timeseries.

Usage:
python3 drb_run_sim.py <inflow_type> <backup_inflow_type> 

"""
import sys
import numpy as np
from pywr.model import Model
from pywr.recorders import TablesRecorder

from make_ensemble_model import drb_make_model
import custom_parameters.ffmp_parameters
import custom_parameters.starfit_parameter
import custom_parameters.flow_ensemble_parameter
from utils.directories import output_dir, model_data_dir

import time
start_time = time.time()

### specify inflow type from command line args
N_SCENARIOS = 10
inflow_type = 'obs_pub'

start_date = '2001-01-01'
end_date = '2010-12-31'

model_filename = f'{model_data_dir}drb_ensemble_model_full.json'
output_filename = f'{output_dir}drb_ensemble_output_{inflow_type}.hdf5'

inflow_ensemble_indices= [1,2,3,4]

### make model json files
drb_make_model(inflow_type, 'obs_pub', start_date, end_date, 
               inflow_ensemble_indices= inflow_ensemble_indices)

### Load the model
model = Model.load(model_filename)

### Add a storage recorder
TablesRecorder(model, output_filename, parameters=[p for p in model.parameters if p.name])

### Run the model
stats = model.run()
stats_df = stats.to_dataframe()

print("--- %s seconds --------" % (time.time() - start_time))