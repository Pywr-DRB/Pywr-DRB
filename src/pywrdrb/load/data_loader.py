import pandas as pd
import os
import pywrdrb
from pywrdrb.load.abstract_loader import AbstractDataLoader
from pywrdrb.load import Output, Observation
# from pywrdrb.load import HydrologicModelFlow

from pywrdrb.utils.results_sets import pywrdrb_results_set_opts, hydrologic_model_results_set_opts, obs_results_set_opts

from pywrdrb import get_pn_object
pn = get_pn_object()

default_kwargs = {
    "datatypes": [],
    "results_sets": [],
    "output_filenames": None,
    "units": "MG",
    "print_status": False,
}


all_valid_results_set_opts = {
    "output": pywrdrb_results_set_opts,
    "obs": obs_results_set_opts,
    "nhmv10": hydrologic_model_results_set_opts,
    "nwmv21": hydrologic_model_results_set_opts,
}

all_results_sets = pywrdrb_results_set_opts + obs_results_set_opts + hydrologic_model_results_set_opts


class Data(AbstractDataLoader):
    
    def __init__(self, pn=pn, **kwargs):
        """
        Initialize the Data loader with default and provided keyword arguments.

        Keyword Args:
            results_sets (list): List of results sets to load.
            output_filenames (list): List of pywrdrb output filenames, with path, to load. Only necessary for Data.load_output().
            units (str): Units for the results. (default 'MG').
            print_status (bool): Whether to print status updates (default False).
            
        Example usage:
        from pywrdrb import Data

        # For loading observations data
        data = Data(results_sets=['major_flow'], print_status=True)
        data.load_observations()

        # For output data, must provide output filenames
        f = "./output_data/drb_output_nhmv10.hdf5"
        data.load_output(output_filenames=[f])
        """
        
        # pathnavigator object
        self.pn = pn
        
        self.all_results_sets = all_results_sets
        self.default_kwargs = default_kwargs
        self.__parse_kwargs__(default_kwargs=self.default_kwargs,
                              **kwargs)

    def __get_valid_datatype_results_set_opts__(self, datatype):
        """
        Get a subset of results_sets that are valid for the specified datatype.
        
        Args:
            datatype (str): The datatype of interest. Options: 'output', 'obs', 'nhmv10', 'nwmv21'.

        Returns:
            list : A subset of results_sets that are valid for the specified datatype.
        """
        
        if datatype not in all_valid_results_set_opts:
            raise ValueError(f"Invalid datatype specified: {datatype}")
        return all_valid_results_set_opts[datatype]
    
    
    def __get_results_sets_subset__(self, datatype):
        """
        Get a subset of results_sets that are:
        (1) valid for the specified datatype and  
        (2) not already stored in the object.
        
        Args:
            datatype (str): The datatype of interest. Options: 'output', 'obs', 'nhmv10', 'nwmv21'.

        Returns:
            list : A subset of results_sets that are valid for the specified datatype.
        """
        
        valid_results_set_opts = self.__get_valid_datatype_results_set_opts__(datatype)
        results_sets_subset = [s for s in self.results_sets if s in valid_results_set_opts]
    
        existing_results_sets = [s for s in results_sets_subset if hasattr(self, s)]
        
        # check if results_set[datatype] already exists in the object
        # avoid re-loading data that already exists
        for s in existing_results_sets:
            results_set_data = getattr(self, s)
            results_set_keys = list(results_set_data.keys())
            if datatype in results_set_keys:
                results_sets_subset.remove(s)
        
        return results_sets_subset

    def __print_status__(self, message):
        """
        Print status message if print_status is True.

        Args:
            message (str): Message to be printed.

        Returns:
            None
        """
        if self.print_status:
            print(message)
            
    
    def __set_loader_data__(self, loader):
        """
        Set data stored in an loader object as attributes of this class.

        Args:
            loader (AbstractDataLoader): A data loader object with results_set attributes.

        Returns:
            None
        """
        for results_set in loader.results_sets:
            self.set_data(getattr(loader, results_set), results_set)
    
    
    def load_observations(self, **kwargs):
        """
        Load observational data.
        """
        self.__parse_kwargs__(default_kwargs=self.default_kwargs,
                                **kwargs)
        self.__print_status__(f"Loading observations data...")
        
        results_sets_subset = self.__get_results_sets_subset__('obs')
        
        # Directory with obs data from pn
        input_dir = self.pn.observations.get_str() + os.sep + "_raw" + os.sep 
        
        
        # Observation data loader
        loader = Observation(
            input_dir = input_dir,
            results_sets = results_sets_subset,
            units = self.units,
            print_status = self.print_status
        )

        loader.load()
        self.__set_loader_data__(loader)
        return 
    
    
    def load_output(self, **kwargs):
        """
        Load data from a pywrdrb output file. 
        """
        
        self.__parse_kwargs__(default_kwargs=self.default_kwargs,
                                **kwargs)
        self.__print_status__(f"Loading pywrdrb output data...")
        
        results_sets_subset = self.__get_results_sets_subset__('output')
        
        assert(self.output_filenames is not None), "output_filenames list must be provided for Data.load_output()"
        
        loader = Output(
            output_filenames=self.output_filenames, 
            results_sets=results_sets_subset, 
            units=self.units,
            print_status=self.print_status
            )
        
        loader.load()
        self.__set_loader_data__(loader)
        return 
    
        
    #TODO: Implement this for different models
    # def load_hydrologic_model(self, 
    #                           **kwargs):
        
    #     self.__parse_kwargs__(default_kwargs=self.default_kwargs,
    #                             **kwargs)
    #     self.__print_status__(f"Loading observations data...")
        
    #     results_sets_subset = self.__get_results_sets_subset__('obs')
        
        
    #     loader = HydrologicModelFlow(
    #         input_dir = self.input_dir,
    #         model = datatype,
    #         results_sets = results_sets_subset,
    #         units = self.units,
    #         print_status = self.print_status
    #     )
        
    #     loader.load()
    #     self.__set_loader_data__(loader)
    #     pass

        
    def export(self, file):
        """
        Export all data stored in this object to an HDF5 file, preservoing format.

        Args:
            file (str): Path to the new HDF5 file.

        Returns:
            None
        """
        with pd.HDFStore(file, mode='w') as store:
            for attr_name in self.all_results_sets:
                if hasattr(self, attr_name):
                    result_set = getattr(self, attr_name)
                    for datatype, scenarios in result_set.items():
                        for scenario_id, df in scenarios.items():
                            key = f"/{attr_name}/{datatype}/{scenario_id}"
                            store.put(key, df)
    
    
    def load_from_export(self, file):
        """
        Load data from an HDF5 file into the object.
        HDF5 should previously have been created using the Data.export() method.

        Args:
            file (str): Path to the HDF5 file.

        Returns:
            None
        """
        
        super().__verify_files_exist__([file])

        with pd.HDFStore(file, mode='r') as store:
            for key in store.keys():
                
                # Key format: /attr_name/datatype/scenario_id
                _, attr_name, datatype, scenario_id = key.split('/')
                scenario_id = int(scenario_id)

                if attr_name not in self.all_results_sets:
                    continue

                if not hasattr(self, attr_name):
                    setattr(self, attr_name, {})

                result_set = getattr(self, attr_name)
                if datatype not in result_set:
                    result_set[datatype] = {}

                result_set[datatype][scenario_id] = store[key]
    
