from pywrdrb.load.abstract_loader import AbstractDataLoader
from pywrdrb.utils.directories import input_dir
from pywrdrb.load.get_results import get_base_results
from pywrdrb.utils.results_sets import base_results_set_opts

default_kwargs = {
    "input_dir": input_dir,
    "results_sets": [],
    "units": "MG",
    "print_status": False,
}

class Observation(AbstractDataLoader):
    
    def __init__(self, **kwargs):
        """
        Initalize the Observation data loader with default and provided kwargs.
        
        Keyword Arguments:
            input_dir (str): Directory where the observation data is stored.
            results_sets (list): List of results sets to load.
            units (str): Units of the observation data. Options: 'MG' or 'MCM'.
            print_status (bool): Print status of the data loading process.
        """
        self.default_kwargs = default_kwargs
        self.valid_results_sets = base_results_set_opts
        self.datetime_index = None
        
        super().__parse_kwargs__(self.default_kwargs, 
                                 **kwargs)
        
        super().__validate_results_sets__(self.valid_results_sets)
    
    def load(self, 
             **kwargs):
        """
        Load observation data based on results_sets provided and store in the object.
        
        Keyword Arguments:
            input_dir (str): Directory where the observation data is stored.
            results_sets (list): List of results_set types to load.
            units (str): Units of the observation data. Options: 'MG' or 'MCM'.
            print_status (bool): Print status of the data loading process.
        
        Returns:
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
                input_dir = input_dir,
                model = 'obs',
                results_set = s,
                datetime_index=datetime,
                units=self.units,
            )
        
            super().set_data(data=all_results_data[s],
                             name=s)
        