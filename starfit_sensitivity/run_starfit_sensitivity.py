############################################################################################################
"""
SUMMARY:
Run Starfit sensitivity analysis on the Pywr-DRB model using the Latin Hypercube Sampling (LHS) method,
"""
############################################################################################################

from SALib.sample import latin
import numpy as np
import pandas as pd
import math
import traceback
import logging 
import os

import sys

path_to_pywrdrb = '../'
sys.path.append(path_to_pywrdrb)

from pywr.model import Model 
from pywrdrb.parameters import *
from pywrdrb.make_model import make_model
from pywrdrb.utils.directories import output_dir

from pywr.recorders import TablesRecorder
import warnings
warnings.filterwarnings("ignore")

import h5py

# Path to the samples
#hdf5_file_path = h5py.File(f"{model_data_dir}scenarios_data.h5", "r")

N_SAMPLES = 2001 #including default scenario
SAMPLE_IDS = np.arange(N_SAMPLES).tolist() 
BATCH_SIZE = 10

#### RUN SIMULATIONS

### SPECIFICATIONS
#inflow_type = "nhmv10_withObsScaled"
# Date ranges for #inflow_type = "nhmv10_withObsScaled"
#start_date = '1983-10-01'
#end_date = '2016-09-30'

inflow_type = "obs_pub_nhmv10_ObsScaled"
# Date ranges for #inflow_type = "obs_pub_nhmv10_ObsScaled"
start_date = '1945-01-01'
end_date = '2022-12-31'
# Simulation start and end dates
#from pywrdrb.utils.dates import model_date_ranges
#start_date, end_date = model_date_ranges[inflow_type]
# Historic reconstructions
#    model_date_ranges[f"obs_pub_{nxm}_ObsScaled"] = ("1945-01-01", "2022-12-31")
#    model_date_ranges[f"obs_pub_{nxm}_ObsScaled_ensemble"] = (
#        "1945-01-01",
#        "2022-12-31",
#    )


# Split the realizations into batches
n_batches= math.ceil(N_SAMPLES/BATCH_SIZE)

batched_indices={i: SAMPLE_IDS[(i*BATCH_SIZE):min((i*BATCH_SIZE + BATCH_SIZE), N_SAMPLES)]
                    for i in range(n_batches)}
batched_filenames=[]

# Run individual batches
for batch, indices in batched_indices.items():
    indices = list(indices)

    print(f'Running {inflow_type} batch {batch} with scenarios {indices}')
    sys.stdout.flush()

    model_filename = f'{model_data_dir}drb_model_full_{inflow_type}.json'
    output_filename = f'{output_dir}drb_output_{inflow_type}_batch{batch}.hdf5'
    batched_filenames.append(output_filename)

    # Debugging for make_model function
    print(f"Generating model file: {model_filename}")
    ### make model json files
    make_model(inflow_type, model_filename, start_date, end_date, 
            sensitivity_analysis_scenarios=indices)

    ### Load the model
    #model = Model.load(model_filename)
    
    try:
        print(f"Loading model from: {model_filename}")
        model = Model.load(model_filename)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Check N_SCENARIOS
    n_scenarios = len(model.scenarios.get_combinations())
    print(f'Model built with {n_scenarios} scenarios')

    ### Add a storage recorder
    TablesRecorder(model, output_filename, 
                   parameters=[p for p in model.parameters if p.name])

    ### Run the model
    #stats = model.run()
    #stats_df = stats.to_dataframe()

    # Set up logging
    #logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Set up logging to write to a file
    logging.basicConfig(filename='model_run.log', 
                    filemode='w',  # 'a' for append, 'w' for overwrite
                    level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
    # Run the model
    try:
        #print(f"Running model for batch {batch}...")
        logging.info(f"Running model for batch {batch}...")
        stats = model.run()
        stats_df = stats.to_dataframe()
        #print(f"Model run completed for batch {batch}.")
        logging.info(f"Model run completed for batch {batch}.")
    except Exception as e:
        #print(f"Error during model run for batch {batch}: {e}")
        #print("Detailed traceback:")
        #traceback.print_exc()  # Print the full traceback
        logging.error(f"Error during model run for batch {batch}: {e}")
        logging.debug("Detailed traceback:", exc_info=True)  # This will include the stack trace
        sys.exit(1)


### Combine output files
from pywrdrb.utils.hdf5 import combine_batched_hdf5_outputs

final_output_filename = f'{output_dir}drb_output_{inflow_type}.hdf5'
print("Combining batched output files...")

combine_batched_hdf5_outputs(batch_files=batched_filenames,
                             combined_output_file=final_output_filename)

print('Deleting individual batch results files')
for file in batched_filenames:
    os.remove(file)

print('Sensitivity analysis completed! Woop!')