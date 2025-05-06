"""
Used to load observed data in a standard format.

Overview:
Defines the Observation class which provides functionality for loading
observational data (streamflows, reservoir storages, etc) from the internally
stored datasets.

Technical Notes:
- Extends AbstractDataLoader with observation-specific functionality 
- Organizes results by results_set, like other loaders
- Designed to be compatible with other data loaders using the `AbstractDataLoader.set_data()` method
- Uses the get_base_results function to load data

Links:
- See results_set options in the docs: https://pywr-drb.github.io/Pywr-DRB/results_set_options.html

Change Log:
TJA, 2025-05-02, Added consistent docstrings.
"""

from pywrdrb.load.abstract_loader import AbstractDataLoader, default_kwargs
from pywrdrb.load.get_results import get_base_results
from pywrdrb.utils.results_sets import obs_results_set_opts


class Observation(AbstractDataLoader):
    """
    Loads observed hydrological data.
        
    Attributes
    -----------
    default_kwargs : dict
        Default keyword arguments.
    pn : PathNavigator
        Path navigator object for directory management. Default is the global pn.
    valid_results_sets : list
        Valid result set options.
    datetime_index : pd.DatetimeIndex or None
        Datetime index for the loaded data.
    """
    def __init__(self, 
                 **kwargs):
        """
        Initialize the Observation data loader with default and provided kwargs.
        
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
        self.default_kwargs = default_kwargs
        self.default_kwargs['input_dir'] = None
        self.valid_results_sets = obs_results_set_opts
        self.datetime_index = None
        
        super().__parse_kwargs__(self.default_kwargs, 
                                 **kwargs)
        
        super().__validate_results_sets__(self.valid_results_sets)
    
    def load(self, 
             **kwargs):
        """
        Load observation data based on results_sets provided.
        
        Loads data corresponding to the specified results sets and
        stores it in the object as attributes.  
        
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
        super().__validate_results_sets__(self.valid_results_sets)

        all_results_data = {}
        datetime = self.datetime_index
        
        for s in self.results_sets:
            all_results_data[s] = {}
            
            if self.print_status:
                print(f"Loading {s} data from observations")
            
            all_results_data[s]['obs'], datetime = get_base_results(
                input_dir = self.input_dir,
                model = 'obs',
                results_set = s,
                datetime_index=datetime,
                units=self.units,
            )
        
            super().set_data(data=all_results_data[s],
                             name=s)
        