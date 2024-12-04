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

sys.path.insert(0, os.path.abspath("./"))
sys.path.insert(0, os.path.abspath("../"))

from pywrdrb import ModelBuilder

import pywrdrb.parameters.general
import pywrdrb.parameters.ffmp
import pywrdrb.parameters.starfit
import pywrdrb.parameters.lower_basin_ffmp
from pywrdrb.utils.dates import model_date_ranges

# from pywrdrb.make_model import make_model
from pywrdrb.utils.directories import output_dir, model_data_dir, input_dir
from pywrdrb.utils.hdf5 import (
    get_hdf5_realization_numbers,
    combine_batched_hdf5_outputs,
)
from pywrdrb.utils.options import inflow_type_options

t0 = time.time()

### specify inflow type from command line args
inflow_type = sys.argv[1]
assert (
    inflow_type in inflow_type_options
), f"Invalid inflow_type specified. Options: {inflow_type_options}"

# modify input directory for ensemble sets
if "syn" in inflow_type:
    input_dir = f"{input_dir}/synthetic_ensembles/"
elif ("pub" in inflow_type) and ("ensemble" in inflow_type):
    input_dir = f"{input_dir}/historic_ensembles/"


### specify whether to use MPI or not. This only matters for ensemble mode.
if len(sys.argv) > 2:
    use_mpi_options = [None, "", "True", "False"]
    use_mpi = sys.argv[2]
    assert (
        use_mpi in use_mpi_options
    ), f"Invalid use_mpi specified. Options: {use_mpi_options}"
    if use_mpi == "True":
        use_mpi = True
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        use_mpi = False
else:
    use_mpi = False


# Simulation start and end dates
# assume we want to run the full range for each dataset
start_date, end_date = model_date_ranges[inflow_type]


# Set the filename based on inflow type
model_filename = f"{model_data_dir}drb_model_full_{inflow_type}.json"


## Run single trace for given inflow type
if "ensemble" not in inflow_type:
    output_filename = f"{output_dir}drb_output_{inflow_type}.hdf5"
    model_filename = f"{model_data_dir}drb_model_full_{inflow_type}.json"

    ### make model json files
    print("Making model...")
    mb = ModelBuilder(
        inflow_type, start_date, end_date
    )  # Optional "options" argument is available
    mb.make_model()
    mb.write_model(model_filename)

    ### Load the model
    print("Loading model...")
    model = Model.load(model_filename)

    ### Add a storage recorder
    TablesRecorder(
        model, output_filename, parameters=[p for p in model.parameters if p.name]
    )

    ### Run the model
    print("Starting simulation...")
    stats = model.run()


## Run ensemble reconstruction in batches
elif "ensemble" in inflow_type:
    # Get the IDs for the realizations
    ensemble_input_filename = f"{input_dir}/catchment_inflow_{inflow_type}.hdf5"
    realization_ids = get_hdf5_realization_numbers(ensemble_input_filename)
    n_realizations = len(realization_ids)

    # How many inflow scenarios to run internally per sim using Pywr parallel
    if use_mpi:
        batch_size = 5
        ### get subset of realizations assigned to this rank
        rank_realization_ids = []
        count = 0
        for i, r in enumerate(realization_ids):
            if count == rank:
                rank_realization_ids.append(realization_ids[i])
            count += 1
            if count == size:
                count = 0
        realization_ids = rank_realization_ids
        n_realizations = len(realization_ids)

    else:
        batch_size = 10

    # Split the realizations into batches
    n_batches = math.ceil(n_realizations / batch_size)
    batched_indices = {
        i: realization_ids[
            (i * batch_size) : min((i * batch_size + batch_size), n_realizations)
        ]
        for i in range(n_batches)
    }
    batched_filenames = []

    # Run individual batches
    for batch, indices in batched_indices.items():
        print(f"Running {inflow_type} {batch} with inflow scenarios {indices}")
        sys.stdout.flush()

        if use_mpi:
            model_filename = (
                f"{model_data_dir}drb_model_full_{inflow_type}_rank{rank}.json"
            )
            output_filename = (
                f"{output_dir}drb_output_{inflow_type}_rank{rank}_batch{batch}.hdf5"
            )
        else:
            model_filename = f"{model_data_dir}drb_model_full_{inflow_type}.json"
            output_filename = f"{output_dir}drb_output_{inflow_type}_batch{batch}.hdf5"
        batched_filenames.append(output_filename)

        ### make model json files
        print("Making model...")

        options = {"inflow_ensemble_indices": indices}

        mb = ModelBuilder(
            inflow_type, start_date, end_date, options=options
        )  # Optional "options" argument is available
        mb.make_model()
        mb.write_model(model_filename)

        ### Load the model
        model = Model.load(model_filename)

        ### Add a storage recorder
        TablesRecorder(
            model, output_filename, parameters=[p for p in model.parameters if p.name]
        )

        ### Run the model
        stats = model.run()
