"""
This script is used to run a Pywr-DRB simulation.

Usage:
python3 drb_run_sim.py <inflow_type>

"""
import sys

from pywr.model import Model
from pywr.recorders import TablesRecorder
import custom_parameters.ffmp_parameters
import custom_parameters.starfit_parameter
from drb_make_model import drb_make_model
from utils.directories import output_dir, model_data_dir

inflow_type = 'obs_pub'

start_date = '1955-01-01'
end_date = '2022-12-31'

model_filename = f'{model_data_dir}drb_model_full.json'
output_filename = f'{output_dir}drb_output_{inflow_type}.hdf5'

### make model json files
drb_make_model(inflow_type, start_date, end_date)

### Load the model
model = Model.load(model_filename)

### Add a storage recorder
TablesRecorder(model, output_filename, parameters=[p for p in model.parameters if p.name])

### Run the model
print(f'Running historic reconstruction from {start_date} to {end_date}')
stats = model.run()
stats_df = stats.to_dataframe()