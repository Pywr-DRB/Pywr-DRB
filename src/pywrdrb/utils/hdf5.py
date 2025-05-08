"""
Contains functions for working with HDF5 files in pywrdrb context.

Overview:
Simple functions for reading and working with HDF5 files. 
Some functions are deisgned for pywrdrb output files while others are for ensemble input files.

Technical Notes: 
- In the future, we should consider improving this to be a more standard class, but for now these are simple functions.

Links: 
- NA
 
Change Log:
TJA, 2025-05-06, Add docs.
"""

import h5py
import pandas as pd
import numpy as np

from pywrdrb.pywr_drb_node_data import obs_site_matches

pywrdrb_all_nodes = list(obs_site_matches.keys())

def get_n_scenarios_from_pywrdrb_output_file(file_path):
    """
    Determine the number of scenarios (n_scenarios) from datasets in an HDF5 file.

    This function assumes that datasets have the shape (len(datetime), len(n_scenarios)).
    It skips a specific dataset named 'time' and ignores non-dataset objects.
    This is used by pywrdrb.load.Output() class to determine the number of scenarios
    in a pywrdrb output file.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.

    Returns
    -------
    int
        The number of scenarios (n_scenarios), determined from the second dimension
        of the first relevant dataset.

    Raises:
    ------
    ValueError
        If no valid dataset is found in the file.
    """
    with h5py.File(file_path, 'r') as hdf_file:
        for name, obj in hdf_file.items():
            if name == 'time':  # Skip the 'time' dataset since it is always 1d
                continue
            if isinstance(obj, h5py.Dataset):  # Only consider datasets
                return obj.shape[1]  # Return n_scenarios from the second dimension
    raise ValueError("No valid datasets found in the HDF5 file.")



def combine_batched_hdf5_outputs(batch_files, combined_output_file):
    """
    Aggregate multiple pywrdrb output (hdf5) files into a single HDF5 file.

    Parameters
    ----------
    batch_files : list of str
        List of full paths to the HDF5 files to be combined. Must be full filename.
    combined_output_file : str
        Path to the output HDF5 file where the combined data will be stored. Must be full filename.
    
    Returns
    -------
    None
    """
    if not batch_files:
        raise ValueError("No batch files provided.")

    # Open all input files once and store their references
    hdf5_files = [h5py.File(file, "r") for file in batch_files]

    with h5py.File(combined_output_file, "w") as hf_out:
        # Extract keys and time array from the first file
        first_file = hdf5_files[0]
        keys = list(first_file.keys())

        datetime_key_opts = ["time", "date", "datetime"]
        time_key = next(
            (dt_key for dt_key in datetime_key_opts if dt_key in keys), None
        )
        if time_key is None:
            err_msg = f"No time key found in HDF5 file {batch_files[0]}."
            err_msg += f" Expected keys: {datetime_key_opts}"
            raise ValueError(err_msg)

        time_array = first_file[time_key][:]

        # Process each key except datetime keys and scenarios
        for key in keys:
            if key in datetime_key_opts + ["scenarios"]:
                continue

            # Collect data for the current key from all files
            data_for_key = []
            for hf_in in hdf5_files:
                if key in hf_in:
                    data_for_key.append(hf_in[key][:])

            if data_for_key:
                # Concatenate data along the second axis (axis=1)
                combined_data = np.concatenate(data_for_key, axis=1)
                hf_out.create_dataset(key, data=combined_data)

        # Write the time array to the output file
        hf_out.create_dataset(time_key, data=time_array)

    # Close all input files
    for hf in hdf5_files:
        hf.close()

    return


def get_hdf5_realization_numbers(filename):
    """
    Checks the contents of hdf5 and return a list of the realization IDs.
     
    Parameters
    ----------
    flename : str
        The filename for the hdf5 file.
    
    Returns
    -------
    realization_numbers
        list of int or str corresponding to the realization IDs in the hdf5 file.
    """
    realization_numbers = []
    with h5py.File(filename, "r") as file:
        # Get the keys in the HDF5 file
        keys = list(file.keys())

        # Get the df using a specific node key
        node_data = file[keys[0]]
        column_labels = node_data.attrs["column_labels"]

        # Iterate over the columns and extract the realization numbers
        for col in column_labels:
            # handle different types of column labels
            if type(col) == str:
                if col.startswith("realization_"):
                    # Extract the realization number from the key
                    realization_numbers.append(int(col.split("_")[1]))
                else:
                    realization_numbers.append(col)
            elif type(col) == int:
                realization_numbers.append(col)
            else:
                err_msg = f"Unexpected type {type(col)} for column label {col}."
                err_msg += f"in HDF5 file {filename}"
                raise ValueError(err_msg)
    return realization_numbers


def extract_realization_from_hdf5(hdf5_file, realization, stored_by_node=False):
    """
    Pull a single inflow realization from an HDF5 file of inflows.

    Parameters
    ----------
    hdf5_file : str
            Path to the HDF5 file.
    realization : str or int
            The realization number or name to extract.
    stored_by_node : bool, optional
                If True, assumes that the data keys are node names. If False, keys are realizations. Default is False.
                
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the extracted realization data, with datetime as the index.
    """

    with h5py.File(hdf5_file, "r") as f:
        if stored_by_node:
            # Extract timeseries data from realization for each node
            data = {}

            for node in pywrdrb_all_nodes:
                node_data = f[node]
                column_labels = node_data.attrs["column_labels"]

                err_msg = f"The specified realization {realization} is not available in the HDF file."
                assert realization in column_labels, (
                    err_msg + f" Realizations available: {column_labels}"
                )
                data[node] = node_data[realization][:]

            dates = node_data["date"][:].tolist()

        else:
            realization_group = f[realization]

            # Extract column labels
            column_labels = realization_group.attrs["column_labels"]
            # Extract timeseries data for each location
            data = {}
            for label in column_labels:
                dataset = realization_group[label]
                data[label] = dataset[:]

            # Get date indices
            dates = realization_group["date"][:].tolist()
        data["datetime"] = dates

    # Combine into dataframe
    df = pd.DataFrame(data, index=dates)
    df.index = pd.to_datetime(df.index.astype(str))
    return df
