"""
This script is used to run a Pywr-DRB simulation.

Usage:
python3 drb_run_sim.py <inflow_type>

"""
import sys

from pywr.model import Model
from pywr.recorders import TablesRecorder
import parameters.ffmp
import parameters.starfit
from make_model import make_model
from utils.directories import output_dir, model_data_dir

inflow_type_options = ['obs_pub', 'nhmv10', 'nwmv21', 'WEAP_29June2023_gridmet',
                       'obs_pub_nhmv10_NYCScaled', 'obs_pub_nwmv21_NYCScaled',
                       'obs_pub_nhmv10_NYCScaled_ensemble', 'obs_pub_nwmv21_NYCScaled_ensemble']

### specify inflow type from command line args
inflow_type = sys.argv[1]
assert(inflow_type in inflow_type_options), f'Invalid inflow_type specified. Options: {inflow_type_options}'


### assume we want to run the full range for each dataset
if inflow_type in ('nwmv21', 'nhmv10', 'WEAP_29June2023_gridmet'):
    start_date = '1983-10-01'
    end_date = '2016-12-31'
elif 'obs_pub' in inflow_type:
    start_date = '1950-01-01'
    end_date = '2022-12-31'

### for ensemble mode, list scenario indices we want to run
if 'ensemble' in inflow_type:
    inflow_ensemble_indices = [1, 2, 3, 4, 5]#, 6, 7, 8, 9, 10]
else:
    inflow_ensemble_indices = None

model_filename = f'{model_data_dir}drb_model_full_{inflow_type}.json'
output_filename = f'{output_dir}drb_output_{inflow_type}.hdf5'


### make model json files
make_model(inflow_type, start_date, end_date, inflow_ensemble_indices= inflow_ensemble_indices)

### Load the model
model = Model.load(model_filename)

### Add a storage recorder
TablesRecorder(model, output_filename, parameters=[p for p in model.parameters if p.name])

### Run the model
stats = model.run()
stats_df = stats.to_dataframe()
