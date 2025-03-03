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

def get_pn_config(filename=None):
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
        return pn.sc.to_dict()

def load_dirs_config(config):
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
    config : str or dict
        The file to load the directories configuration from.
    """
    global pn  # Ensure pn is modified globally
    if ".json" in config:
        pn.sc.load_json(config, overwrite=True)
    elif ".yml" in config:
        pn.sc.load_yaml(config, overwrite=True)
    else:
        pn.sc.load_dict(config, overwrite=True)

reset_pn()


#### The following are pending to be removed.


# Set a global directory instance
# Has to be done before importing other modules
# https://chatgpt.com/share/674673b5-607c-8007-ab64-d845d032cb10
@dataclass
class Directories:
    root_dir: str = field(init=False)
    input_dir: str = field(init=False)
    model_data_dir: str = field(init=False)

    def __post_init__(self):
        """Ensures the correct root directory and initializes paths."""
        
        # Get the root directory
        self.root_dir = os.path.realpath(os.path.dirname(__file__))
        
        # Ensure root_dir points to the package directory
        if not os.path.basename(self.root_dir) == "pywrdrb":
            self.root_dir = os.path.join(self.root_dir, "pywrdrb")

        # Set input_dir correctly
        self.input_dir = os.path.join(self.root_dir, "input_data") + os.sep
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(f"input_data folder not found at {self.input_dir}")

        # Set model_data_dir correctly
        self.model_data_dir = os.path.join(self.root_dir, "model_data") + os.sep
        if not os.path.exists(self.model_data_dir):
            raise FileNotFoundError(f"model_data folder not found at {self.model_data_dir}")


    def list(self):
        """Prints the directories."""
        for attribute, value in self.__dict__.items():
            print(f"{attribute}: {value}")

# Create a global instance of Directory
_directory_instance = Directories()

def get_directory() -> Directories:
    """
    Returns the global instance of the Directory.
    """
    return _directory_instance

def set_directory(**kwargs):
    """
    Updates the global Directory instance with the provided keyword arguments.
    """
    for key, value in kwargs.items():
        if hasattr(_directory_instance, key):
            if not os.path.isdir(value):  # Ensure it's a valid directory
                raise ValueError(f"Invalid directory path: {value}")
            setattr(_directory_instance, key, value)
        else:
            raise AttributeError(f"Invalid directory attribute: {key}")



# Import pywr modules
from pywr.model import Model
from pywr.recorders import NumpyArrayParameterRecorder, TablesRecorder


from .model_builder import ModelBuilder
from .load.data_loader import Data

# Not sure why this is needed, but it is.
from .parameters.ffmp import *
VolBalanceNYCDemand.register()

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