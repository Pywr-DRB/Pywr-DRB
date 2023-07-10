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
from drb_make_model import drb_make_model
from utils.directories import output_dir, model_data_dir

obs_pub_donor_fdc = sys.argv[1]            # Options: 'nwmv21', 'nhmv10'
regression_nhm_inflow_scaling = True if sys.argv[2] == 'yes' else 'no'   # If true, Cannonsville and Pep. inflows increase following NHM-based regression to estimate HRU inflows

inflow_type = f'obs_pub_{obs_pub_donor_fdc}_NYCScaling{regression_nhm_inflow_scaling}'

start_date = '1950-01-01'
end_date = '2022-12-31'

obs_pub_donor_fdc = sys.argv[1]            # Options: 'nwmv21', 'nhmv10'
regression_nhm_inflow_scaling = True if sys.argv[2] == 'yes' else 'no'   # If true, Cannonsville and Pep. inflows increase follo

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
