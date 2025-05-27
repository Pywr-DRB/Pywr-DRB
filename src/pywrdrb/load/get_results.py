"""
Functions for retrieving and organizing simulation results from various models.

Overview:
This module contains functions to load and process data from:
- pywrdrb output files
- observational data files
- nhm, nwm, and other internally available datasets

Technical Notes:
- Data is consistently organized based on results_set
- All functions should support conversion to units: MGD, MCM
- #TODO:
    - Rethink the use of `get_base_results()` function. See Notes in docstring. 
    - Come up with consistent naming for 'model' as used in get_base_results().

Links:
- See results_set options in the docs: https://pywr-drb.github.io/Pywr-DRB/results_set_options.html

Change Log:
TJA, 2025-05-02, Added consistent docstrings. Deleted old functions.
"""

import os
import numpy as np
import pandas as pd
import h5py
import warnings

from pywrdrb.utils.lists import (
    reservoir_list,
    reservoir_list_nyc,
    majorflow_list,
    reservoir_link_pairs,
)
from pywrdrb.utils.lists import drbc_lower_basin_reservoirs
from pywrdrb.utils.constants import cfs_to_mgd, mg_to_mcm
from pywrdrb.utils.results_sets import pywrdrb_results_set_opts


def get_keys_and_column_names_for_results_set(keys, results_set):
    """
    For given results_set, identify hdf5 key subset and new variable names.
    
    The pywrdrb output file contains a large number of variables which each
    have a different key in the HDF5 file. When loading results, we want to 
    extract a subset of these variables corresponding to a unique results_set.
    E.g., for results_set = "res_storage" we want to keep only keys for the 
    stroage variables. 
    
    Also, we want to rename the keys to be user-friendly, often we rename the 
    variable as simply the node name. These col_names are used to rename columns
    for the loaded pd.DataFrame.
    
    Parameters
    ----------
    keys : list[str]
        The full list of HDF5 keys stored in the pywrdrb output file.
    results_set : str
        The type of results to retrieve.
        
    Returns
    -------
    tuple
        (keys, col_names) where keys is a subset of all output hdf5 keys to extract 
        for the given results_set and col_names are the corresponding column names 
        for the final output DataFrame, used to rename the keys.
    """
    if results_set == "all":
        keys = keys
        col_names = [k for k in keys]
        
    elif results_set == "reservoir_downstream_gage":
        ## Need to pull flow data for link_ downstream of reservoirs instead of simulated outflows
        keys_with_link = [
            k
            for k in keys
            if k.split("_")[0] == "link"
            and k.split("_")[1] in reservoir_link_pairs.values()
        ]
        col_names = []
        for k in keys_with_link:
            col_names.append(
                [
                    res
                    for res, link in reservoir_link_pairs.items()
                    if link == k.split("_")[1]
                ][0]
            )

        # Now pull simulated relases from un-observed reservoirs
        keys_without_link = [
            k
            for k in keys
            if k.split("_")[0] == "outflow"
            and k.split("_")[1] in reservoir_list
            and k.split("_")[1] not in reservoir_link_pairs.keys()
        ]
        for k in keys_without_link:
            col_names.append(k.split("_")[1])
        keys = keys_with_link + keys_without_link

    elif results_set == "res_storage":
        keys = [
            k
            for k in keys
            if k.split("_")[0] == "reservoir" and k.split("_")[1] in reservoir_list
        ]
        col_names = [k.split("_")[1] for k in keys]
        
        # print(f"Reservoir storage keys: {keys}")
        # print(f"Reservoir storage column names: {col_names}")
        
    elif results_set == "major_flow":
        keys = [
            k
            for k in keys
            if k.split("_")[0] == "link" and k.split("_")[1] in majorflow_list
        ]
        col_names = [k.split("_")[1] for k in keys]
    elif results_set == "res_release":
        ### reservoir releases are "outflow" plus "spill".
        # These are summed later in this function.
        # Not all reservoirs have spill.
        keys_outflow = [f"outflow_{r}" for r in reservoir_list]
        keys_spill = [f"spill_{r}" for r in reservoir_list]
        keys = keys_outflow + keys_spill
        col_names = keys

    elif results_set == "downstream_release_target":
        keys = [f"{results_set}_{reservoir}" for reservoir in reservoir_list_nyc]
        col_names = reservoir_list_nyc

    elif results_set == "inflow":
        keys = [k for k in keys if k.split("_")[0] == "catchment"]
        col_names = [k.split("_")[1] for k in keys]
    elif results_set == "catchment_withdrawal":
        keys = [k for k in keys if k.split("_")[0] == "catchmentWithdrawal"]
        col_names = [k.split("_")[1] for k in keys]
    elif results_set == "catchment_consumption":
        keys = [k for k in keys if k.split("_")[0] == "catchmentConsumption"]
        col_names = [k.split("_")[1] for k in keys]

    elif results_set in (
        "prev_flow_catchmentWithdrawal",
        "max_flow_catchmentWithdrawal",
        "max_flow_catchmentConsumption",
    ):
        keys = [k for k in keys if results_set in k]
        col_names = [k.split("_")[-1] for k in keys]

    elif results_set in ("res_level"):
        keys = [k for k in keys if "drought_level" in k]
        col_names = [k.split("_")[-1] for k in keys]

    elif results_set == "ffmp_level_boundaries":
        keys = [f"level{l}" for l in ["1b", "1c", "2", "3", "4", "5"]]
        col_names = [k for k in keys]
    elif results_set == "mrf_target":
        keys = [k for k in keys if results_set in k]
        col_names = [k.split("mrf_target_")[1] for k in keys]

    elif results_set == "nyc_release_components":
        keys = (
            [
                f"mrf_target_individual_{reservoir}"
                for reservoir in reservoir_list_nyc
            ]
            + [f"flood_release_{reservoir}" for reservoir in reservoir_list_nyc]
            + [
                f"mrf_montagueTrenton_{reservoir}"
                for reservoir in reservoir_list_nyc
            ]
            + [f"spill_{reservoir}" for reservoir in reservoir_list_nyc]
        )
        col_names = [k for k in keys]
    elif results_set == "lower_basin_mrf_contributions":
        keys = [
            f"mrf_trenton_{reservoir}" for reservoir in drbc_lower_basin_reservoirs
        ]
        col_names = [k for k in keys]
    elif results_set == "ibt_demands":
        keys = ["demand_nyc", "demand_nj"]
        col_names = [k for k in keys]
    elif results_set == "ibt_diversions":
        keys = ["delivery_nyc", "delivery_nj"]
        col_names = [k for k in keys]
    elif results_set == "mrf_targets":
        keys = ["mrf_target_delMontague", "mrf_target_delTrenton"]
        col_names = [k for k in keys]
    elif results_set == "all_mrf":
        keys = [k for k in keys if "mrf" in k]
        col_names = [k for k in keys]
    elif results_set == "temperature":
        keys = [k for k in keys if "temperature" in k] \
            + [k for k in keys if "thermal" in k] \
            + ['estimated_Q_i', 'estimated_Q_C']
        col_names = [k for k in keys]
    elif results_set == "salinity":
        keys = [k for k in keys if "salinity" in k] \
            + [k for k in keys if "salt_front" in k]
        col_names = [k for k in keys]
    # resulst_set may be a specific key in the model
    elif results_set in keys:
        keys = [results_set]
        col_names = [results_set]
    else:
        #TODO: raise value error
        pass
    return keys, col_names


def get_pywrdrb_results(
    output_filename, 
    results_set="all", 
    scenarios=[0], 
    datetime_index=None, 
    units=None,
    ):
    """
    Extract simulation results from pywrdrb model output file.
    
    Retrieves specified results for specific scenarios from an HDF5 output file
    and organizes them into a dictionary of pandas DataFrames.
    
    Parameters
    ----------
    output_filename : str
        The full output filename from pywrdrb simulation (e.g., "<path>/drb_output_nhmv10.hdf5").
    results_set : str, optional
        The results set to return. Options include:
        - "all": All results.
        - "reservoir_downstream_gage": Downstream gage flow below reservoir.
        - "res_storage": Reservoir storages.
        - "major_flow": Flow at major flow points of interest.
        - "inflow": Inflow at each catchment.
    scenarios : list[int], optional
        The scenario index numbers. Only needed for ensemble simulation results. Default: [0]
    datetime_index : pd.DatetimeIndex, optional
        Existing datetime index to reuse. Creating dates is slow, so reusing is efficient.
    units : str, optional
        Units to convert flow data to. Options: "MG", "MCM"

    Returns
    -------
    tuple
        (dict, pd.DatetimeIndex) where dict maps scenario indices to DataFrames
        of results, and pd.DatetimeIndex is the datetime index used.
    """
    # Validate output file
    output_filename = output_filename if ".hdf5" in output_filename else f"{output_filename}.hdf5"

    # Validate results_set
    if results_set not in pywrdrb_results_set_opts:
        err_msg = f"Invalid results_set specified for get_pywrdrb_results().\n"
        err_msg += f" Valid results_set options: {pywrdrb_results_set_opts}"
        raise ValueError(err_msg)

    # Get result data from HDF5 output file
    with h5py.File(output_filename, "r") as f:
        all_keys = list(f.keys())
        keys, col_names = get_keys_and_column_names_for_results_set(keys=all_keys, 
                                                                    results_set=results_set)

        data = []
        # Now pull the data using keys
        for k in keys:
            data.append(f[k][:, scenarios])

        # Convert data to 3D array
        data = np.stack(data, axis=2)

        if units is not None:
            if units == "MG":
                pass
            elif units == "MCM":
                data *= mg_to_mcm

        if datetime_index is not None:
            if len(datetime_index) == len(f["time"]):
                reuse_datetime_index = True
            else:
                reuse_datetime_index = False
        else:
            reuse_datetime_index = False

        if not reuse_datetime_index:
            # Format datetime index
            time = f["time"][:]
            
            ### custom OutputRecorder requires this:
            if type(time[0]) == bytes:
                datetime_index = pd.to_datetime([t.decode('utf-8') for t in f['time'][:]])

            ### TablesRecorder requires this:
            else:                
                day = [f["time"][i][0] for i in range(len(f["time"]))]
                month = [f["time"][i][2] for i in range(len(f["time"]))]
                year = [f["time"][i][3] for i in range(len(f["time"]))]
                date = [f"{y}-{m}-{d}" for y, m, d in zip(year, month, day)]
                datetime_index = pd.to_datetime(date)


        # Now store each scenario as individual pd.DataFrames in the dict
        results_dict = {}
        for s in scenarios:
            results_dict[s] = pd.DataFrame(
                data[:, s, :], columns=col_names, index=datetime_index
            )

        # If results_set is 'res_release', sum the outflow and spill data,
        # columns with the same reservoir name
        if results_set == "res_release":
            for s in scenarios:
                for r in reservoir_list:
                    release_cols = [c for c in results_dict[s].columns if r in c]

                    # sum
                    if len(release_cols) > 1:
                        results_dict[s][r] = results_dict[s][release_cols].sum(axis=1)
                        results_dict[s] = results_dict[s].drop(release_cols, axis=1)
                    else:
                        results_dict[s] = results_dict[s].rename(
                            columns={release_cols[0]: r}
                        )
        return results_dict, datetime_index


def get_base_results(
    input_dir,
    model,
    datetime_index=None,
    results_set="all",
    ensemble_scenario=None,
    units=None,
    ):
    """
    Retrieve results from gage_flow_mgd.csv or gage_flow_mgd.hdf5 fils.
    
    These 'base' results include flows from different sources,
    which are _not_ the pywrdrb model. This function is designed to be used for the 
    internally available datasets, including "obs", "nwmv21", 
    "nhmv10", "nwmv21_withObsScaled", etc.
    
    Parameters
    ----------
    input_dir : str
        Directory containing input data files.
    model : str
        Model name. Options:
        - "obs": Observed data.
        - "nwmv21": NWM v2.1 data.
        - "nhmv10": NHM v1.0 data.
        - "nwmv21_withObsScaled": NWM v2.1 data with scaled inflow observations.
        - "nhmv10_withObsScaled": NHM v1.0 data with scaled inflow observations.
    datetime_index : pd.DatetimeIndex, optional
        Existing datetime index to reuse. Creating dates is slow, so reusing is efficient.
    results_set : str, optional
        Results set to return. Options:
        - "all": All results.
        - "reservoir_downstream_gage": Downstream gage flow below reservoir.
        - "major_flow": Flow at major flow points of interest.
    ensemble_scenario : int, optional
        Ensemble scenario index. If provided, load data from HDF5 file 
        instead of CSV.
    units : str, optional
        Units to convert flow data to. Options: "MG", "MCM"

    Returns
    -------
    tuple
        (dict, pd.DatetimeIndex) where dict maps scenario indices to DataFrames
        of results, and pd.DatetimeIndex is the datetime index used.
        
    Notes
    -----
    (TJA) It would be nice to rethink this function. The term "base result" is not clear, 
    and not appropriate. Base originally referred to natural flows, but observed flows are also
    included which are non-natural. For now, this is important for loading the internal datasets. 
    """
    if ensemble_scenario is None:
        gage_flow = pd.read_csv(f"{input_dir}/gage_flow_mgd.csv")
        gage_flow.index = pd.DatetimeIndex(gage_flow["datetime"])
        gage_flow = gage_flow.drop("datetime", axis=1)
    else:
        with h5py.File(f"{input_dir}/gage_flow_mgd.hdf5", "r") as f:
            nodes = list(f.keys())
            gage_flow = pd.DataFrame()
            for node in nodes:
                gage_flow[node] = f[f"{node}/realization_{ensemble_scenario}"]

            if datetime_index is not None:
                if len(datetime_index) == len(f[nodes[0]]["date"]):
                    gage_flow.index = datetime_index
                    reuse_datetime_index = True
                else:
                    reuse_datetime_index = False
            else:
                reuse_datetime_index = False

            if not reuse_datetime_index:
                datetime = [str(d, "utf-8") for d in f[nodes[0]]["date"]]
                datetime_index = pd.to_datetime(datetime)
                gage_flow.index = datetime_index

        data = gage_flow.copy()

    if results_set == "reservoir_downstream_gage":
        available_release_data = gage_flow.columns.intersection(
            reservoir_link_pairs.values()
        )
        reservoirs_with_data = [
            list(
                filter(lambda x: reservoir_link_pairs[x] == site, reservoir_link_pairs)
            )[0]
            for site in available_release_data
        ]
        gage_flow = gage_flow.loc[:, available_release_data]
        gage_flow.columns = reservoirs_with_data

        data = gage_flow.copy()

    elif results_set == "major_flow":
        for c in gage_flow.columns:
            if c not in majorflow_list:
                gage_flow = gage_flow.drop(c, axis=1)
        data = gage_flow.copy()

    elif results_set == "res_storage" and model == "obs":
        observed_storage_path = (
            f"{input_dir}/reservoir_storage_mg.csv"
        )
        try:
            observed_storage = pd.read_csv(observed_storage_path)
            observed_storage.index = pd.DatetimeIndex(observed_storage["datetime"])
            observed_storage = observed_storage.drop("datetime", axis=1)
            data = observed_storage.copy()
        except FileNotFoundError:
            print(f"Observed storage CSV file not found at {observed_storage_path}.")
            return None, datetime_index
        except KeyError:
            print(
                'The observed storage data does not contain the expected "datetime" column.'
            )
            return None, datetime_index
    elif results_set == "res_storage" and model != "obs":
        raise ValueError(
            f"Reservoir storage data is not available for model={model}. Only available for model='obs'."
        )

    else:
        raise ValueError("Invalid results_set specified for get_base_results().")

    if units is not None:
        if units == "MG":
            pass
        elif units == "MCM":
            data *= mg_to_mcm

    ## Re-organize as dict for consistency with pywrdrb results
    results_dict = {0: data}
    return results_dict, datetime_index