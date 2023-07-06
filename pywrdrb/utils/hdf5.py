import h5py
import pandas as pd



def export_ensemble_to_hdf5(dict, output_file):
    
    dict_keys = list(dict.keys())
    N = len(dict)
    T, M = dict[dict_keys[0]].shape
    column_labels = dict[dict_keys[0]].columns.to_list()
    
    with h5py.File(output_file, 'w') as f:
        for key in dict_keys:
            data = dict[key]
            datetime = data.index.astype(str).tolist() #.strftime('%Y-%m-%d').tolist()
            
            grp = f.create_group(key)
                    
            # Store column labels as an attribute
            grp.attrs['column_labels'] = column_labels

            # Create dataset for dates
            grp.create_dataset('date', data=datetime)
            
            # Create datasets for each array subset from the group
            for j in range(M):
                dataset = grp.create_dataset(column_labels[j], 
                                             data=data[column_labels[j]].to_list())

    return


def get_hdf5_realization_numbers(filename):
    """Checks the contents of an hdf5 file, and returns a list 
    of the realization ID numbers contained.
    Realizations have key 'realization_i' in the HDF5.

    Args:
        filename (str): The HDF5 file of interest

    Returns:
        list: Containing realizations ID numbers; realizations have key 'realization_i' in the HDF5.
    """
    realization_numbers = []
    with h5py.File(filename, 'r') as file:
        # Get the keys (dataset names) in the HDF5 file
        keys = list(file.keys())

        # Iterate over the keys and extract the realization numbers
        for key in keys:
            if key.startswith('realization_'):
                # Extract the realization number from the key
                realization_number = int(key.split('_')[1])
                realization_numbers.append(realization_number)

    return realization_numbers


def extract_realization_from_hdf5(hdf5_file, realization):
    """_summary_

    Args:
        hdf5_file (str): The filename for the hdf5 file
        realization (_type_): Integer realization index

    Returns:
        pandas.DataFrame: A DataFrame containing the realization
    """
    with h5py.File(hdf5_file, 'r') as f:
        realization_group = f[f"realization_{realization}"]
        
        # Extract column labels
        column_labels = realization_group.attrs['column_labels']
        
        # Extract timeseries data for each location
        data = {}
        for label in column_labels:
            dataset = realization_group[label]
            data[label] = dataset[:]
        
        # Get date indices
        dates = realization_group['date'][:].tolist()
        # dates = pd.to_datetime([d[1:] for d in dates])
        
    # Combine into dataframe
    df = pd.DataFrame(data, index = dates)
    df.index = pd.to_datetime(df.index.astype(str))
    return df

def load_all_node_inflow_realizations_hdf5(hdf5_file):
    """_summary_

    Args:
        hdf5_file (str): The filename for the hdf5 file
    
    Returns:
        pandas.DataFrame: A DataFrame containing the realization
    """
    with h5py.File(hdf5_file, 'r') as f:
        realization_group = f[f"realization_{realization}"]
        
        # Extract column labels
        column_labels = realization_group.attrs['column_labels']
        
        # Extract timeseries data for each location
        data = {}
        for label in column_labels:
            dataset = realization_group[label]
            data[label] = dataset[:]
        
        # Get date indices
        dates = realization_group['date'][:].tolist()
        # dates = pd.to_datetime([d[1:] for d in dates])
        
    # Combine into dataframe
    df = pd.DataFrame(data, index = dates)
    df.index = pd.to_datetime(df.index.astype(str))
    return df