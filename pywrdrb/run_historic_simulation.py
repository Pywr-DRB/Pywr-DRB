"""
This script is used to run a Pywr-DRB simulation.

Usage:
python3 drb_run_sim.py <inflow_type>

"""
import sys
import os
import math

from pywr.model import Model
from pywr.recorders import TablesRecorder
import parameters.ffmp
import parameters.starfit
from make_model import make_model
from utils.directories import output_dir, model_data_dir, input_dir
from utils.hdf5 import get_hdf5_realization_numbers, combine_batched_hdf5_outputs

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

# Set the filename based on inflow type
model_filename = f'{model_data_dir}drb_model_full_{inflow_type}.json'


## Run single trace for given inflow type
if 'ensemble' not in inflow_type:

    inflow_ensemble_indices = None
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

## Run ensemble reconstruction in batches
elif 'ensemble' in inflow_type:
    # How many inflow scenarios to run internally per sim using Pywr parallel
    batch_size= 10
    inflow_subtype= inflow_type.split('obs_pub_')[1]
    
    # Get the IDs for the realizations
    ensemble_input_filename= f'{input_dir}/historic_ensembles/historic_reconstruction_daily_{inflow_subtype}_mgd.hdf5'
    realization_ids= get_hdf5_realization_numbers(ensemble_input_filename)
    
    # Split the realizations into batches
    n_realizations=len(realization_ids)
    n_batches= math.ceil(n_realizations/batch_size)
    batched_indices={f'batch_{i}': realization_ids[(i*batch_size):min((i*batch_size + batch_size), 
                                                                      n_realizations)] for i in range(n_batches)}
    batched_filenames=[]
    
    # Run individual batches
    for batch, indices in batched_indices.items():
        
        print(f'Running {inflow_type} {batch} with inflow scenarios {indices}.')
        model_filename = f'{model_data_dir}drb_model_full_{inflow_type}.json'
        output_filename = f'{output_dir}drb_output_{inflow_type}_{batch}.hdf5'
        batched_filenames.append(output_filename)
        
        ### make model json files
        make_model(inflow_type, start_date, end_date, 
                   inflow_ensemble_indices= indices)

        ### Load the model
        model = Model.load(model_filename)

        ### Add a storage recorder
        TablesRecorder(model, output_filename, parameters=[p for p in model.parameters if p.name])

        ### Run the model
        stats = model.run()
        stats_df = stats.to_dataframe()

    # Combine outputs into single HDF5
    print(f'Combining {len(batched_filenames)} batched output files to single HDF5 file.')
    combined_output_filename = f'{output_dir}drb_output_{inflow_type}.hdf5'
    combine_batched_hdf5_outputs(batch_files=batched_filenames,
                                 combined_output_file=combined_output_filename)
    
    # Delete batched files
    print('Deleting batched file outputs')
    for file in batched_filenames:
        os.remove(file)