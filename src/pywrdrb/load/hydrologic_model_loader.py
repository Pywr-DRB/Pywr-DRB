"""
Used to load data from the available hydrologic model files.

Overview:
This module defines the HydrologicModelFlow class which provides functionality for loading.
It is designed to load from the pywrdrb/data/flows/ directory based on the specified
flowtype.  

Technical Notes:
- Extends AbstractDataLoader with flow-specific functionality 
- Organizes results by results_set, like other loaders
- Designed to be compatible with other data loaders using the `AbstractDataLoader.set_data()` method
- Uses the get_base_results function to load data

Links:
- See results_set options in the docs: https://pywr-drb.github.io/Pywr-DRB/results_set_options.html

Change Log:
TJA, 2025-05-05, Implemented pathnavigator usage & added consistent docstrings.
"""

from pywrdrb.load.abstract_loader import AbstractDataLoader, default_kwargs
from pywrdrb.load.get_results import get_base_results
from pywrdrb.utils.results_sets import hydrologic_model_results_set_opts

# TODO:
# This list should be generated dynamically 
# using the pathnavigator, based on all the 
# current data/flows/ folders or other directories
flowtype_opts = [
    'nhmv10',
    'nhmv10_withObsScaled', 
    'nwmv21',
    'nwmv21_withObsScaled',
    'wrf1960s_calib_nlcd2016',
    'wrf2050s_calib_nlcd2016',
    'wrfaorc_calib_nlcd2016',
    'wrfaorc_withObsScaled',
    ]


class HydrologicModelFlow(AbstractDataLoader):
    """
    Loads observed hydrological data.

    Methods
    -------
    load(flowtypes, **kwargs)
        Load data for the specified hydrologic model and results_sets.

    Attributes
    -----------
    default_kwargs : dict
        Default keyword arguments.
    pn : PathNavigator
        Path navigator object for directory management. Default is the global pn.
    valid_results_sets : list
        Valid result set options.
    datetime_index : pd.DatetimeIndex or None
        Datetime index for the loaded data. If None, full datetime is used.
    flowtype_opts : list
        List of valid flowtype options, which are available in the data/flows/ directory.
    """
    def __init__(self, 
                 flowtype_opts = flowtype_opts, 
                 **kwargs):
        """
        Initialize the loader with default and provided kwargs.
        
        Parameters
        ----------
        pn : PathNavigator, optional
            Path navigator object for directory management. Default is the global pn.
        results_sets : list, optional
            List of results sets to load.
        units : str, optional
            Units of the observation data. Options: 'MG' or 'MCM'.
        print_status : bool, optional
            Print status of the data loading process.
        """
        self.flowtype_opts = flowtype_opts
        self.default_kwargs = default_kwargs
        self.valid_results_sets = hydrologic_model_results_set_opts
        self.datetime_index = None
    
        # During parse, kwargs are set as attributes
        super().__parse_kwargs__(self.default_kwargs, 
                                 **kwargs)
        
        super().__validate_results_sets__(self.valid_results_sets)
    
    
    def load(self,
             flowtypes,
             **kwargs):
        """
        Load data for the specified hydrologic model and results_sets.

        Stores data as attributes of the object.
        
        Parameters
        ----------
        results_sets : list, optional
            List of results_set types to load. Default is all available sets.
        units : str, optional
            Units of the observation data. Options: 'MG' or 'MCM'. Default is 'MG'.
        print_status : bool, optional
            Print status of the data loading process. Default is False.
        
        Returns
        -------
        None
        """
        super().__parse_kwargs__(self.default_kwargs, 
                                 **kwargs)
                
        all_results_data = {}
        
        datetime = self.datetime_index
        
        # loop through the results sets
        for s in self.results_sets:
            all_results_data[s] = {}
            
            # loop through the flowtypes    
            for flowtype in flowtypes:
                assert(flowtype in self.flowtype_opts), f"Invalid flowtype specified: {flowtype}\nValid options are: {self.flowtype_opts}"    

                if self.print_status:
                    print(f'Loading {s} data from {flowtype}')
            
                # Get the input_dir from the pathnavigator
                flow_dir = self.pn.flows.get_str(flowtype)
            
                # load the data
                all_results_data[s][flowtype], datetime = get_base_results(
                    input_dir = flow_dir,
                    model = flowtype,
                    results_set = s,
                    datetime_index=datetime,
                    units=self.units,
                )
                
            # Set the data as an attribute
            super().set_data(data=all_results_data[s],
                             name=s)