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
import pandas as pd
import h5py
from abc import ABC, abstractmethod

from pywrdrb.utils.constants import mg_to_mcm
from pywrdrb.utils.results_sets import pywrdrb_results_set_opts
from pywrdrb.utils.lists import (
    reservoir_list,
    reservoir_list_nyc,
    majorflow_list,
    reservoir_link_pairs,
    drbc_lower_basin_reservoirs
)

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
                raise FileNotFoundError(f"File not found at path: {file}")

    def get_base_results(
        self,
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
            Model name.
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