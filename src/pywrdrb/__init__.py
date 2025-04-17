# Import pywr modules to be accessed through pywrdrb
from pywr.model import Model

# Run path_manager first!!! 
from .path_manager import *
reset_pn() # Create a new global pathnavigator instance

from .recorder import *
from .model_builder import *
from . import pre
from .load.data_loader import Data  

# All "parameters" need to be registered such that they can be accessed in pywr when loading a model file.
# We register the parameter classes right after they are defined. But some time it they are not correctly registered.
# In that case, we need to register them again here. To ensure that all parameters are registered whenever pywrdrb is imported.

# CL: I think the better way is to register them here. But the code work for now, so I will leave it as is.
# If we encounter issues with the parameters, then we do the structural change.
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