"""
This script is used to run a Pywr-DRB simulation.

Usage:
python3 drb_run_sim.py <inflow_type> <backup_inflow_type> 

"""
import sys

from pywr.model import Model
from pywr.recorders import TablesRecorder

from drb_make_model import drb_make_model
import custom_parameters.ffmp_parameters
import custom_parameters.starfit_parameter
from utils.directories import output_dir, model_data_dir

### specify inflow type from command line args
inflow_type = sys.argv[1]
if len(sys.argv) >=2:
    backup_inflow_type = sys.argv[2]
else:
    backup_inflow_type = 'nhmv10'

# inflow_type = 'nhmv10'  ### nhmv10, nwmv21, nwmv21_withLakes, obs, obs_pub, WEAP_23Aug2022_gridmet
# backup_inflow_type = 'obs-pub'  ## for WEAP inflow type, we dont have all reservoirs. use this secondary type for missing.

### assume we want to run the full range for each dataset
if inflow_type in ('nwmv21', 'nwmv21_withLakes', 'nhmv10', 'obs_pub'):
    start_date = '1999-10-01' #'1983-10-01'
    end_date = '2010-12-31'#'2016-12-31'
elif 'WEAP' in inflow_type:
    start_date = '1999-06-01'
    end_date = '2010-05-31'

# start_date = '1999-06-01'
# end_date = '2010-05-31'

model_filename = f'{model_data_dir}drb_model_full.json'
if 'WEAP' in inflow_type:
    output_filename = f'{output_dir}drb_output_{inflow_type}_{backup_inflow_type}.hdf5'
else:
    output_filename = f'{output_dir}drb_output_{inflow_type}.hdf5'


### make model json files
drb_make_model(inflow_type, backup_inflow_type, start_date, end_date)

### Load the model
model = Model.load(model_filename)

### Add a storage recorder
TablesRecorder(model, output_filename, parameters=[p for p in model.parameters if p.name])

### Run the model
stats = model.run()
stats_df = stats.to_dataframe()
