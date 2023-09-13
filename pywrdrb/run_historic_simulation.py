"""
This script is used to run a Pywr-DRB simulation.

Usage:
python3 drb_run_sim.py <inflow_type>

"""
import sys
import os
import math
import time
from pywr.model import Model
from pywr.recorders import TablesRecorder

sys.path.insert(0, os.path.abspath('./'))
sys.path.insert(0, os.path.abspath('../'))

import parameters.ffmp
import parameters.starfit
from make_model import make_model
from utils.directories import output_dir, model_data_dir, input_dir
from utils.hdf5 import get_hdf5_realization_numbers, combine_batched_hdf5_outputs

t0 = time.time()

### specify inflow type from command line args
inflow_type_options = ['obs_pub', 'nhmv10', 'nwmv21', 'WEAP_29June2023_gridmet',
                        'nhmv10_withObsScaled', 'nwmv21_withObsScaled',
                       'obs_pub_nhmv10_ObsScaled', 'obs_pub_nwmv21_ObsScaled',
                       'obs_pub_nhmv10', 'obs_pub_nwmv21',
                       'obs_pub_nhmv10_ObsScaled_ensemble', 'obs_pub_nwmv21_ObsScaled_ensemble']
inflow_type = sys.argv[1]
assert(inflow_type in inflow_type_options), f'Invalid inflow_type specified. Options: {inflow_type_options}'

### specify whether to use MPI or not. This only matters for ensemble mode.
if len(sys.argv) > 2:
    use_mpi_options = [None,'','True','False']
    use_mpi = sys.argv[2]
    assert(use_mpi in use_mpi_options), f'Invalid use_mpi specified. Options: {use_mpi_options}'
    if use_mpi == 'True':
        use_mpi = True
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        use_mpi = False
else:
    use_mpi = False

### assume we want to run the full range for each dataset
if inflow_type in ('nwmv21', 'nhmv10', 'WEAP_29June2023_gridmet') or 'withObsScaled' in inflow_type:
    start_date = '1983-10-01'
    end_date = '2016-12-31'
elif 'obs_pub' in inflow_type:
    start_date = '1952-01-01'
    end_date = '2022-12-31'

# Set the filename based on inflow type
model_filename = f'{model_data_dir}drb_model_full_{inflow_type}.json'


## Run single trace for given inflow type
if 'ensemble' not in inflow_type:

    output_filename = f'{output_dir}drb_output_{inflow_type}.hdf5'
    model_filename = f'{model_data_dir}drb_model_full_{inflow_type}.json'

    ### make model json files
    make_model(inflow_type, model_filename, start_date, end_date)

    ### Load the model
    model = Model.load(model_filename)

    ### Add a storage recorder
    TablesRecorder(model, output_filename, parameters=[p for p in model.parameters if p.name])

    ### Run the model
    stats = model.run()
    stats_df = stats.to_dataframe()

## Run ensemble reconstruction in batches
elif 'ensemble' in inflow_type:
    # Get the IDs for the realizations
    ensemble_input_filename= f'{input_dir}/historic_ensembles/catchment_inflow_{inflow_type}.hdf5'
    realization_ids= get_hdf5_realization_numbers(ensemble_input_filename)
    n_realizations=len(realization_ids)

    # How many inflow scenarios to run internally per sim using Pywr parallel
    if use_mpi:
        batch_size = 5
        ### get subset of realizations assigned to this rank
        rank_realization_ids = []
        count = 0
        for i,r in enumerate(realization_ids):
            if count == rank:
                rank_realization_ids.append(realization_ids[i])
            count += 1
            if count == size:
                count = 0
        print(f'hello from rank {rank} out of {size}. realizations: {rank_realization_ids}')
        realization_ids = rank_realization_ids
        n_realizations = len(realization_ids)

    else:
        batch_size= 10

    # Split the realizations into batches
    n_batches= math.ceil(n_realizations/batch_size)
    batched_indices={i: realization_ids[(i*batch_size):min((i*batch_size + batch_size), n_realizations)]
                     for i in range(n_batches)}
    batched_filenames=[]

    # Run individual batches
    for batch, indices in batched_indices.items():

        print(f'Running {inflow_type} {batch} with inflow scenarios {indices}, {time.time()-t0}')
        sys.stdout.flush()

        if use_mpi:
            model_filename = f'{model_data_dir}drb_model_full_{inflow_type}_rank{rank}.json'
            output_filename = f'{output_dir}drb_output_{inflow_type}_rank{rank}_batch{batch}.hdf5'
        else:
            model_filename = f'{model_data_dir}drb_model_full_{inflow_type}.json'
            output_filename = f'{output_dir}drb_output_{inflow_type}_batch{batch}.hdf5'
        batched_filenames.append(output_filename)

        ### make model json files
        make_model(inflow_type, model_filename, start_date, end_date, inflow_ensemble_indices=indices)

        ### Load the model
        model = Model.load(model_filename)

        ### Add a storage recorder
        TablesRecorder(model, output_filename, parameters=[p for p in model.parameters if p.name])

        ### Run the model
        stats = model.run()
        stats_df = stats.to_dataframe()


