"""
Contains functions for retrieving and organizing
simulation results from pywrdrb model runs.
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
from pywrdrb.utils.hdf5 import get_hdf5_realization_numbers
from pywrdrb.utils.directories import input_dir
from pywrdrb.utils.results_sets import pywrdrb_results_set_opts


def get_keys_and_column_names_for_results_set(keys, results_set):
    """
    Return

    Args:
        keys (list[str]): The full list of HDF5 keys stored in the pywrdrb output file.
        results_set (str): The type of results that should be retrieved.
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
        keys = [k for k in keys if "temperature" in k]
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
    units=None
):
    """
    Gathers simulation results from pywrdrb model run 
    and returns a dict of pd.DataFrames.
    This can handle retrieve multiple scenarios.
    Each key in the dict corresponds to a scenario.

    Args:
        output_filename (str): The output filename from pywrdrb simulation (e.g., "<path>/drb_output_nhmv10.hdf5").
        results_set (str, optional): The results set to return. Can be one of the following:
            - "all": Return all results.
            - "reservoir_downstream_gage": Return downstream gage flow below reservoir.
            - "res_storage": Return reservoir storages.
            - "major_flow": Return flow at major flow points of interest.
            - "inflow": Return the inflow at each catchment.
            (Default: 'all')
        scenarios (list(int), optional): The scenario index numbers. Default: [0]
        datetime_index (Pandas datetime_index): Creating the dates are slow: if this isn't our first data retrieval, we can provide the dates from a previous results dataframe.
        units (str, optional): The units to convert the flow data to. Options: "MG", "MCM"

    Returns:
        dict(pd.DataFrame): Dictionary containing simulation results for each scenario.
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


### load flow estimates from raw input datasets
def get_base_results(
    input_dir,
    model,
    datetime_index=None,
    results_set="all",
    ensemble_scenario=None,
    units=None,
):
    """
    Function for retrieving and organizing results from non-pywr streamflows (NHM, NWM, WEAP).

    Args:
        input_dir (str): The input data directory.
        model (str): The model datatype name (e.g., "nhmv10").
        datetime_index: The datetime index.
        results_set (str, optional): The results set to return. Can be one of the following:
            - "all": Return all results.
            - "reservoir_downstream_gage": Return downstream gage flow below reservoir.
            - "major_flow": Return flow at major flow points of interest.
            (Default: 'all')
        ensemble_scenario (int, optional): The ensemble scenario index number. If not None, load data from HDF5. Else look for CSV. (Default: None)
        units (str, optional): The units to convert the flow data to. Options: "MG", "MCM"

    Returns:
        pd.DataFrame: The retrieved and organized results with datetime index.
    """
    if ensemble_scenario is None:
        gage_flow = pd.read_csv(f"{input_dir}gage_flow_{model}.csv")
        gage_flow.index = pd.DatetimeIndex(gage_flow["datetime"])
        gage_flow = gage_flow.drop("datetime", axis=1)
    else:
        with h5py.File(f"{input_dir}gage_flow_{model}.hdf5", "r") as f:
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
            f"{input_dir}/historic_reservoir_ops/observed_storage_data.csv"
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


def get_all_historic_reconstruction_pywr_results(
    output_dir, model_list, results_set, start_date, end_date, units="MG"
):
    """
    Function for retrieving and organizing results from multiple pywr models.

    """
    datetime_index = pd.date_range(start=start_date, end=end_date, freq="D")
    results_dict = {}
    for model in model_list:
        # For ensembles, get realization/scenario numbers
        if "ensemble" in model:
            realizations = get_hdf5_realization_numbers(
                f"{input_dir}/historic_ensembles/catchment_inflow_{model}.hdf5"
            )
            scenarios = list(range(len(realizations)))
        else:
            scenarios = [0]

        results_dict[f"pywr_{model}"], datetime_index = get_pywrdrb_results(
            output_dir,
            model,
            results_set=results_set,
            scenarios=scenarios,
            datetime_index=datetime_index,
            units=units,
        )

    return results_dict


### WARNING: Depreciated due to poor handling of many-scenario simulations.
### Use get_pywrdrb_results instead.


def get_pywr_results(
    output_dir, model, results_set="all", scenario=0, datetime_index=None, units=None
):
    """
    Gathers simulation results from Pywr model run and returns a pd.DataFrame.

    WARNING: Depreciated due to poor handling of many-scenario simulations. Use get_pywrdrb_results instead.

    Args:
        output_dir (str): The output directory.
        model (str): The model datatype name (e.g., "nhmv10").
        results_set (str, optional): The results set to return. Can be one of the following:
            - "all": Return all results.
            - "reservoir_downstream_gage": Return downstream gage flow below reservoir.
            - "res_storage": Return reservoir storages.
            - "major_flow": Return flow at major flow points of interest.
            - "inflow": Return the inflow at each catchment.
            (Default: 'all')
        scenario (int, optional): The scenario index number. (Default: 0)
        datetime_index (Pandas datetime_index): Creating the dates are slow: if this isn't our first data retrieval, we can provide the dates from a previous results dataframe.

    Returns:
        pd.DataFrame: The simulation results with datetime index.
    """

    # Raise depreciation warning
    warnings.warn(
        "The get_pywr_results() function is depreciated. Use get_pywrdrb_results instead.",
        DeprecationWarning,
    )

    with h5py.File(f"{output_dir}drb_output_{model}.hdf5", "r") as f:
        keys = list(f.keys())
        results = pd.DataFrame()
        if results_set == "all":
            for k in keys:
                results[k] = f[k][:, scenario]
        elif results_set == "reservoir_downstream_gage":
            ## Need to pull flow data for link_ downstream of reservoirs instead of simulated outflows
            keys_with_link = [
                k
                for k in keys
                if k.split("_")[0] == "link"
                and k.split("_")[1] in reservoir_link_pairs.values()
            ]
            # print(keys_with_link)
            for k in keys_with_link:
                res_name = [
                    res
                    for res, link in reservoir_link_pairs.items()
                    if link == k.split("_")[1]
                ][0]
                results[res_name] = f[k][:, scenario]
            # Now pull simulated relases from un-observed reservoirs
            keys_without_link = [
                k
                for k in keys
                if k.split("_")[0] == "outflow"
                and k.split("_")[1] in reservoir_list
                and k.split("_")[1] not in reservoir_link_pairs.keys()
            ]
            for k in keys_without_link:
                results[k.split("_")[1]] = f[k][:, scenario]
        elif results_set == "res_storage":
            keys = [
                k
                for k in keys
                if k.split("_")[0] == "reservoir" and k.split("_")[1] in reservoir_list
            ]
            for k in keys:
                results[k.split("_")[1]] = f[k][:, scenario]
        elif results_set == "major_flow":
            keys = [
                k
                for k in keys
                if k.split("_")[0] == "link" and k.split("_")[1] in majorflow_list
            ]
            for k in keys:
                results[k.split("_")[1]] = f[k][:, scenario]
        elif results_set == "res_release":
            ### reservoir releases are "outflow" plus "spill". Not all reservoirs have spill.
            keys_outflow = [f"outflow_{r}" for r in reservoir_list]
            for k in keys_outflow:
                results[k.split("_")[1]] = f[k][:, scenario]
            keys_spill = [f"spill_{r}" for r in reservoir_list]
            for k in keys_spill:
                results[k.split("_")[1]] += f[k][:, scenario]
        elif results_set == "downstream_release_target":
            for reservoir in reservoir_list_nyc:
                results[reservoir] = f[f"{results_set}_{reservoir}"][:, scenario]
        elif results_set == "inflow":
            keys = [k for k in keys if k.split("_")[0] == "catchment"]
            for k in keys:
                results[k.split("_")[1]] = f[k][:, scenario]
        elif results_set == "catchment_withdrawal":
            keys = [k for k in keys if k.split("_")[0] == "catchmentWithdrawal"]
            for k in keys:
                results[k.split("_")[1]] = f[k][:, scenario]
        elif results_set == "catchment_consumption":
            keys = [k for k in keys if k.split("_")[0] == "catchmentConsumption"]
            for k in keys:
                results[k.split("_")[1]] = f[k][:, scenario]
        elif results_set in (
            "prev_flow_catchmentWithdrawal",
            "max_flow_catchmentWithdrawal",
            "max_flow_catchmentConsumption",
        ):
            keys = [k for k in keys if results_set in k]
            for k in keys:
                results[k.split("_")[-1]] = f[k][:, scenario]
        elif results_set in ("res_level"):
            keys = [k for k in keys if "drought_level" in k]
            for k in keys:
                results[k.split("_")[-1]] = f[k][:, scenario]
        elif results_set == "ffmp_level_boundaries":
            keys = [f"level{l}" for l in ["1b", "1c", "2", "3", "4", "5"]]
            for k in keys:
                results[k] = f[k][:, scenario]
        elif results_set == "mrf_target":
            keys = [k for k in keys if results_set in k]
            for k in keys:
                results[k.split("mrf_target_")[1]] = f[k][:, scenario]
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
            for k in keys:
                results[k] = f[k][:, scenario]
        elif results_set == "lower_basin_mrf_contributions":
            keys = [
                f"mrf_trenton_{reservoir}" for reservoir in drbc_lower_basin_reservoirs
            ]
            for k in keys:
                results[k] = f[k][:, scenario]
        elif results_set == "ibt_demands":
            keys = ["demand_nyc", "demand_nj"]
            for k in keys:
                results[k] = f[k][:, scenario]
        elif results_set == "ibt_diversions":
            keys = ["delivery_nyc", "delivery_nj"]
            for k in keys:
                results[k] = f[k][:, scenario]
        elif results_set == "mrf_targets":
            keys = ["mrf_target_delMontague", "mrf_target_delTrenton"]
            for k in keys:
                results[k] = f[k][:, scenario]
        elif results_set == "all_mrf":
            keys = [k for k in keys if "mrf" in k]
            for k in keys:
                results[k] = f[k][:, scenario]

        else:
            print("Invalid results_set specified.")
            return

        if datetime_index is not None:
            if len(datetime_index) == len(f["time"]):
                results.index = datetime_index
                reuse_datetime_index = True
            else:
                reuse_datetime_index = False
        else:
            reuse_datetime_index = False

        if not reuse_datetime_index:
            # Format datetime index
            day = [f["time"][i][0] for i in range(len(f["time"]))]
            month = [f["time"][i][2] for i in range(len(f["time"]))]
            year = [f["time"][i][3] for i in range(len(f["time"]))]
            date = [f"{y}-{m}-{d}" for y, m, d in zip(year, month, day)]
            datetime_index = pd.to_datetime(date)
            results.index = datetime_index

        if units is not None:
            if units == "MG":
                pass
            elif units == "MCM":
                results *= mg_to_mcm

        return results, datetime_index
