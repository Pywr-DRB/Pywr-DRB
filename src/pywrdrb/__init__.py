# Import pywr modules to be accessed through pywrdrb
from pywr.model import Model
from pywr.recorders import TablesRecorder


import os
import copy
from dataclasses import dataclass, field
import pathnavigator

##### Set directory config using pathnavigator V0.4.2
# Get the root directory
root_dir = os.path.realpath(os.path.dirname(__file__))

# Ensure root_dir points to the package directory
if not os.path.basename(root_dir) == "pywrdrb":
    root_dir = os.path.join(root_dir, "pywrdrb")

# Create a new global pathnavigator instance
global pn
pn = pathnavigator.create(root_dir)

def reset_pn():
    """
    Resets the pathnavigator object to the default configuration.
    Note: we only want to add shortcuts that matter to the user.
    Others, we should retrieve from pn directly.
    If we use shortcuts for certain files or folders, we will need to use that shortcut 
    throughout the program to make it consistent.
    """
    global pn  # Ensure pn is modified globally
    # Add folder directories as shortcuts (can be accessed as pn.sc.get("folder name"))
    re_pattern = r"^_"  # Ignore folders/files starting with "_"
    
    # Now we only allow users to add customized flows and diversions subfolders 
    # (i.e., flows/customized_flow_type and diversions/customized_diversion_type)
    pn.data.flows.set_all_to_sc(prefix="flows/", mode="folders", overwrite=True, exclude=re_pattern)
    pn.data.diversions.set_all_to_sc(prefix="diversions/", mode="folders", overwrite=True, exclude=re_pattern)

def get_pn_object(copy=False):
    """
    Returns the pathnavigator object.
    
    Returns
    -------
    pathnavigator.PathNavigator
        The directories object.
    """
    global pn  # Ensure pn is modified globally
    if copy:
        return copy.deepcopy(pn)
    else:
        return pn

def get_pn_config(filename=""):
    """
    Returns the pathnavigator configuration.
    
    Parameters
    ----------
    filename : str, optional
        If a filename is provided, saves the configuration to the file. The allowed file
        extensions are ".json" and ".yml".
    
    Returns
    -------
    dict or None
        If no filename is provided, returns the directories configuration as a dictionary.
        If a filename is provided, saves the configuration to the file and returns None.
    """
    global pn  # Ensure pn is modified globally
    if ".json" in filename:
        pn.sc.to_json(filename)
        print(f"Directories configuration saved to {filename}")
        return None
    elif ".yml" in filename:
        pn.sc.to_yaml(filename)
        print(f"Directories configuration saved to {filename}")
        return None
    else:
        return pn.sc.to_dict(to_str=True)

def load_pn_config(pn_config):
    """
    Loads the directories configuration from a file. Overwrites the existing directories
    with the given configuration. The allowed file extensions are ".json" and ".yml". 
    User may also provide a dictionary as input. Partial configurations can be provided.
    
    If users want to create a model with customized inflow files, they need to add the new
    folder to the directories configuration before creating the model with the customize
    flow_type. Otherwise, the model will not be able to find the inflow files.
    
    Alternatively, users can change the directories under the current directories 
    configuration. That way, the model will be created under the default flow_type while  
    using customized files in the given directories.
    
    Parameters
    ----------
    pn_config : str or dict
        The file to load the directories configuration from.
    """
    global pn  # Ensure pn is modified globally
    if ".json" in pn_config:
        pn.sc.load_json(pn_config, overwrite=True)
    elif ".yml" in pn_config:
        pn.sc.load_yaml(pn_config, overwrite=True)
    else:
        pn.sc.load_dict(pn_config, overwrite=True)

reset_pn()




# Import pywr modules
from pywr.model import Model
from pywr.recorders import NumpyArrayParameterRecorder, TablesRecorder


# Core pywrdrb functionality
from .recorders.output_recorder import OutputRecorder
from .model_builder import ModelBuilder
from .load.data_loader import Data

from .parameters.ffmp import *
VolBalanceNYCDemand.register()


from .pre.predict_inflows import PredictedInflowPreprocessor
from .pre.predict_diversions import PredictedDiversionPreprocessor

# CL's temporary output parser
import h5py
def hdf5_to_dict(file_path):
    def recursive_dict(group):
        d = {}
        for key, item in group.items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                d[key] = item[()]
            elif isinstance(item, h5py._hl.group.Group):
                d[key] = recursive_dict(item)
        return d

    with h5py.File(file_path, 'r') as f:
        data_dict = recursive_dict(f)

    return data_dict

# Create a new HDF5 file
def dict_to_hdf5(recorder_dict, filename):
    output_dict = {}
    for name, recorder in recorder_dict.items():
        #print(f"Data for {name}:")
        df = recorder.to_dataframe()
        d = df.reset_index(drop=True)
        d.columns = [int(col) for col in d.columns.get_level_values(0)]    
        d = d.to_dict(orient="list")
        output_dict[name] = d

    # Create a new HDF5 file
    with h5py.File(filename, 'w') as hdf:
        # Loop through each variable in the dictionary
        for var, indices in output_dict.items():
            # Create a group for each variable
            group = hdf.create_group(var)
            # Loop through each index in the variable
            for idx, values in indices.items():
                # Convert index to string to use as dataset name
                dataset_name = str(idx)
                # Create a dataset for each index
                group.create_dataset(dataset_name, data=values)
    
    print(f"Outputs saved to {filename}")