"""
Defines abstract base used for data loaders.

Overview:
This module provides an abstract class that defines the common interface
for different data loaders. Data loaders are used to load different datasets 
while maintaining a consistent data format.

Technical Notes:
- Data loaders that are built from this class:
    - `Observations`
    - `Output`
    ` `Data` which combines all the above
- Provides helper methods for argument parsing, validation, and data storage.
- Works with PathNavigator (pn) for directory management. Default is the global pn.

Links:
- NA

Change Log:
TJA, Spring 2025, Initially created the data loader functionality.
TJA, 2025-05-02, Setup documentation.
"""
import os
from abc import ABC, abstractmethod

from pywrdrb.path_manager import get_pn_object
pn = get_pn_object()

# Default kwargs
default_kwargs = {
    "pn": pn,                   # used by all 
    "results_sets": [],         # used by all
    "output_filenames": [],     # used for Output
    "flowtypes": [],            # used for HydrologicModelFlow
    "output_labels": [],        # used for Output
    "units": "MG",              # used by all
    "print_status": False,      # used by all
}

class AbstractDataLoader(ABC):
    """
    Abstract base class for all data loaders.
    
    Defines common interface and functionality including argument 
    parsing, validation, and data storage methods.
    
    Methods
    -------
    set_data(data, name)
        Store or update data stored in the object attributes, using a dictionary of new data.
    __parse_kwargs__(default_kwargs, **kwargs)
        Parse and set keyword arguments as attributes.
    __validate_results_sets__(valid_results_set_opts)
        Validate the provided results_sets list against valid options.
    __verify_files_exist__(files)
        Verify that all files in a list exist.
    
    Attributes
    ----------
    results_sets : list
        List of result sets to load.
    units : str
        Units for the results (default 'MG'). Options: 'MG' or 'MCM'.
    print_status : bool
        Whether to print status updates.
    """
    
    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the data loader.
        
        This method must be implemented uniqeuly by all subclasses.
        
        Parameters
        ----------
        **kwargs
            Arbitrary keyword arguments, to override default values.
        """
        pass        
    
    
    def __parse_kwargs__(self, 
                         default_kwargs, 
                         **kwargs):
        """
        Parse and set keyword arguments as attributes.
        
        Uses provided kwargs, existing attributes, or default values in that order.

        Parameters
        ----------
        default_kwargs : dict
            Default keyword arguments with default values.
        **kwargs
            User-provided keyword arguments to override defaults.
        """
        # check for invalid kwargs
        for key in kwargs.keys():
            if key not in default_kwargs.keys():
                raise ValueError(f"Invalid keyword argument: {key}")
        
        # Set attribute based on order of precedence:
        # kwargs > existing attribute > default value
        for key, default_value in default_kwargs.items():
            setattr(self, key, kwargs.get(key, getattr(self, key, default_value)))


    def __validate_results_sets__(self, 
                               valid_results_set_opts):
        """
        Validate the provided results_sets list against valid options.

        Parameters
        ----------
        valid_results_set_opts : list
            Valid options for results sets. This is datatype specific.
        """
        for s in self.results_sets:
            if s not in valid_results_set_opts:
                err_msg = (
                    f"Specified results set {s} in results_sets is not a valid option. "
                )
                err_msg += f"Valid options are: {valid_results_set_opts}"
                raise ValueError(err_msg)
    
    def __verify_files_exist__(self, files):
        """
        Verify that all files in a list exist.
        
        Parameters
        ----------
        files : list
            List of file paths to verify.
        """
        for file in files:
            file_exists = os.path.exists(file)
            
            if file_exists:
                return True
            else:
                raise FileNotFoundError(f"File not found at path: {file_path}")
    

    def set_data(self, 
                 data, 
                 name):
        """
        Store or update data in the object as an attribute.

        This is an important method which allows for dynamic updating of internally
        stored data, which is stored as attributes of this object. This allows for 
        the `pywrdrb.Data()` object to perform both `load_observations()` and `load_output()` methods 
        while not overwriting the different data. 
        
        The final data is stored with the structure:
        data_loader.attribute_name = data = dict{data_label: dict{scenario_id: pd.DataFrame}}

        Parameters
        ----------
        data : dict
            Data to set or update, with format dict{data_label: dict{scenario_id: pd.DataFrame}}.
        name : str
            Attribute name for the data.
        """
        
        # if not already an attribute, setattr
        if not hasattr(self, name):
            setattr(self, name, data)
            
        elif hasattr(self, name):
            if getattr(self, name) is not None:
                getattr(self, name).update(data)
            else:
                setattr(self, name, data)