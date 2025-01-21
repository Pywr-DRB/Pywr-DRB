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

    Parameters:
    ----------
    file_path : str
        Path to the HDF5 file.

    Returns:
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
    Aggregate multiple HDF5 files into a single HDF5 file.

    Args:
        batch_files (list): List of HDF5 files to combine.
        combined_output_file (str): Full output file path & name to write combined HDF5.

    Returns:
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


def export_ensemble_to_hdf5(dict, output_file):
    """
    Export a dictionary of ensemble data to an HDF5 file.
    Data is stored in the dictionary as {realization number (int): pd.DataFrame}.

    Args:
        dict (dict): A dictionary of ensemble data.
        output_file (str): Full output file path & name to write HDF5.

    Returns:
        None
    """

    dict_keys = list(dict.keys())
    column_labels = dict[dict_keys[0]].columns.to_list()

    with h5py.File(output_file, "w") as f:
        for key in dict_keys:
            data = dict[key]
            datetime = data.index.to_numpy().astype(
                "S"
            )  # Directly use numpy array for dates

            grp = f.create_group(key)

            # Store column labels as an attribute
            grp.attrs["column_labels"] = column_labels

            # Create dataset for dates
            grp.create_dataset("date", data=datetime)

            # Create datasets for each array subset from the group
            for col in column_labels:
                grp.create_dataset(
                    col, data=data[col].to_numpy()
                )  # Directly use numpy array
    return


def get_hdf5_realization_numbers(filename):
    """
    Checks the contents of an hdf5 file, and returns a list
    of the realization ID numbers contained.
    Realizations have key 'realization_i' in the HDF5.

    Args:
        filename (str): The HDF5 file of interest

    Returns:
        list: Containing realizations ID numbers; realizations have key 'realization_i' in the HDF5.
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

    Args:
        hdf5_file (str): The filename for the hdf5 file
        realization (int): Integer realization index
        stored_by_node (bool): Whether the data is stored with node name as key.

    Returns:
        pandas.DataFrame: A DataFrame containing the realization
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
