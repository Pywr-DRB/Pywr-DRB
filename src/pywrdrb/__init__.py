import os
from dataclasses import dataclass, field

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
