"""
Organize data records into appropriate format for Pywr-DRB.

This script handles ensemble inflow inflow_types.

Predictions of inflows and diversions are made individually for each ensemble member.
The predictions are parallelized using MPI.

This script can be used to prepare inputs from the following inflow_types:
- PUB reconstruction ensembles (obs_pub_*_ensemble)
- Syntehtically generated ensembles (syn_*_ensemble)

"""
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, "../")

from utils.hdf5 import (
    get_hdf5_realization_numbers,
    export_ensemble_to_hdf5,
    extract_realization_from_hdf5,
)
from utils.directories import input_dir
from pre.disaggregate_DRBC_demands import disaggregate_DRBC_demands
from pre.extrapolate_NYC_NJ_diversions import (
    extrapolate_NYC_NJ_diversions,
    download_USGS_data_NYC_NJ_diversions,
)
from pre.predict_inflows_diversions import predict_inflows_diversions
from utils.options import inflow_type_options

from mpi4py import MPI

### specify inflow type from command line args
inflow_type = sys.argv[1]
assert (
    inflow_type in inflow_type_options
), f"Invalid inflow_type specified. Options: {inflow_type_options}"


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()  # Total number of processes
    rank = comm.Get_rank()  # The rank of this process

    ## Optional data processing steps
    # Generally, these only need to be run once.  Assuming already run, set to False.
    download_flow_data_for_diversion_extrapolation = True
    extrapolate_diversions = True
    disaggregate_demands = False

    # Rank 0 handles the optional data processing steps
    if rank == 0:
        ## Get NYC & NJ diversions.
        # For time periods w/ no historical record, extrapolate by seasonal relationship to flow.
        if download_flow_data_for_diversion_extrapolation:
            download_USGS_data_NYC_NJ_diversions(dates=("1945-01-01", "2022-12-31"))
        if extrapolate_diversions:
            extrapolate_NYC_NJ_diversions("nyc", make_figs=True)
            extrapolate_NYC_NJ_diversions("nj", make_figs=True)

        ## Spatially disaggregate DRBC demands to match PywrDRB node catchments
        # Source data is from DRBC report
        if disaggregate_demands:
            sw_demand = disaggregate_DRBC_demands()
    comm.Barrier()

    ## Predict future Montague & Trenton inflows & NJ diversions based on lagged regressions
    # used to inform NYC releases on day t needed to meet flow targets on days t+n (n=1,..4)
    if rank == 0:
        print(f"Starting inflow & diversion predictions for {inflow_type}")

        if "syn" in inflow_type:
            ensemble_input_dir = input_dir + "synthetic_ensembles/"
        elif "obs_pub" in inflow_type:
            ensemble_input_dir = input_dir + "historic_ensembles/"

        ensemble_filename = f"{ensemble_input_dir}catchment_inflow_{inflow_type}.hdf5"

        # Get realization numbers
        realization_numbers = np.array(get_hdf5_realization_numbers(ensemble_filename))

        # Load 1 realization to verify start and end date
        test_realization = extract_realization_from_hdf5(
            ensemble_filename, realization_numbers[0], stored_by_node=True
        )
        start_date = test_realization.index[0]
        end_date = test_realization.index[-1]

    else:
        realization_numbers = None
        start_date, end_date = None, None
    comm.Barrier()

    # Divide work among MPI processes
    # Broadcast the realization numbers and dates to all processes
    realization_numbers = comm.bcast(realization_numbers, root=0)
    start_date = comm.bcast(start_date, root=0)
    end_date = comm.bcast(end_date, root=0)

    # Divide work among MPI processes
    n_realizations = len(realization_numbers)
    proc_realizations = np.array_split(realization_numbers, size)[rank]
    n_realizations_per_proc = len(proc_realizations)

    # Predict inflows and diversions for each realization
    ensemble_predictions = np.array([])
    for i, real in enumerate(proc_realizations):
        # Make predictions for local subset of realizations
        df_predictions = predict_inflows_diversions(
            inflow_type,
            start_date,
            end_date,
            use_log=True,
            remove_zeros=False,
            use_const=False,
            ensemble_inflows=True,
            realization=str(real),
            save_predictions=False,
            return_predictions=True,
            make_figs=False,
            catchment_inflows=None,
        )

        df_predictions = df_predictions.drop(
            columns=[dt for dt in ["date", "datetime"] if dt in df_predictions.columns]
        )

        # combine in 3d np array
        if ensemble_predictions.size == 0:
            ensemble_predictions = df_predictions.values[np.newaxis, :, :]
            df_shape = df_predictions.shape
            df_columns = df_predictions.columns
        else:
            ensemble_predictions = np.concatenate(
                (ensemble_predictions, df_predictions.values[np.newaxis, :, :]), axis=0
            )

    # Make float64 and contiguous
    ensemble_predictions = ensemble_predictions.astype(np.float64)
    ensemble_predictions = np.ascontiguousarray(ensemble_predictions)

    # Gather sizes of local predictions - helps handle different n_realizations_per_proc
    local_size = np.array(ensemble_predictions.size, dtype=int)
    sizes = None
    if rank == 0:
        sizes = np.empty(comm.size, dtype=int)
    comm.Gather(local_size, sizes, root=0)

    # Gather all predictions to root process
    if rank == 0:
        # Calculate displacements for gatherv: accomodate different sizes on ranks
        displacements = np.zeros(comm.size, dtype=int)
        displacements[1:] = np.cumsum(sizes[:-1])

        # Set up array to gather all predictions
        gathered_data = np.empty(sum(sizes), dtype=np.float64)
    else:
        gathered_data = None
        displacements = None

    comm.Gatherv(
        ensemble_predictions, (gathered_data, sizes, displacements, MPI.DOUBLE), root=0
    )

    ## Reformat & export to HDF5 by root process
    if rank == 0:
        print(f"Gathered predictions with shape {gathered_data.shape}")

        # setup datetime index
        datetime = pd.date_range(start=start_date, end=end_date, freq="D")
        datetime = datetime.astype(str).tolist()

        # get number of realizations on each rank
        proc_realizations = np.array_split(realization_numbers, size)
        n_realizations_per_proc = [len(proc) for proc in proc_realizations]

        # setup 3d array
        ensemble_predictions_mat = np.empty(
            (n_realizations, len(datetime), len(df_columns))
        )

        # Place in 3d arrary; ranks may have different n_realizations_per_proc
        for rank_i in range(comm.size):
            start_idx = displacements[rank_i]
            n_reals_on_ri = n_realizations_per_proc[rank_i]

            if n_reals_on_ri == 0:
                continue

            for rank_real_ij in range(n_reals_on_ri):
                if rank_i == 0:
                    real_idx = rank_real_ij
                else:
                    real_idx = (
                        np.cumsum(n_realizations_per_proc[:rank_i])[-1] + rank_real_ij
                    )

                end_idx = start_idx + (sizes[rank_i] / n_reals_on_ri)

                start_idx = int(start_idx)
                end_idx = int(end_idx)

                ensemble_predictions_mat[real_idx, :, :] = gathered_data[
                    start_idx:end_idx
                ].reshape(len(datetime), len(df_columns))
                start_idx = end_idx

        # Convert 3d array to dict
        combined_ensemble_pred = {}
        for i, realization in enumerate(realization_numbers):
            realization_data = ensemble_predictions_mat[i, :, :]
            realization_df = pd.DataFrame(realization_data, columns=df_columns)
            combined_ensemble_pred[f"{realization}"] = realization_df

        # Add datetime index
        for key in combined_ensemble_pred.keys():
            combined_ensemble_pred[key]["datetime"] = datetime

        output_filename = f"{input_dir}predicted_inflows_diversions_{inflow_type}.hdf5"
        print(f"Exporting {inflow_type} predicted inflows/diversions to hdf5.")
        export_ensemble_to_hdf5(combined_ensemble_pred, output_filename)

        print(f"Done preparing ensemble data for {inflow_type}!")
