from pywrdrb.load.abstract_loader import AbstractDataLoader
from pywrdrb.load.get_results import get_base_results
from pywrdrb.utils.directories import input_dir
from pywrdrb.utils.results_sets import base_results_set_opts


default_kwargs = {
    "input_dir": input_dir,
    "model": None,
    "results_sets": [],
    "units": "MG",
    "print_status": False,
}


model_opts = ['nhmv10', 'nwmv21']

class HydrologicModelFlow(AbstractDataLoader):
    
    def __init__(self, **kwargs):
        """
        Initialize the HydrologicModelFlow loader with model and default options.

        Keyword Args:
            input_dir (str): Directory for input files.
            model (str): Model to load data for ('nhmv10' or 'nwmv21').
            results_sets (list): Results sets to load.
            units (str): Units for the results (default 'MG').
            print_status (bool): Whether to print status updates.
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
        Load data for the specified hydrologic model and results_sets.
        Stores data as attributes of the object.

        Keyword Args:
            model (str): Model to load data for ('nhmv10' or 'nwmv21').
            results_sets (list): Results sets to load.
            print_status (bool): Whether to print status updates.

        Returns:
            None
        """ 
        super().__parse_kwargs__(self.default_kwargs, 
                                 **kwargs)
        
        assert(self.model in model_opts), f"Invalid model specified: {self.model}\nValid options are: {model_opts}"
        
        all_results_data = {}
        datetime = self.datetime_index
        
        for s in self.results_sets:
            all_results_data[s] = {}
            
            if self.print_status:
                print(f'Loading {s} data from {self.model}')
            
            all_results_data[s][self.model], datetime = get_base_results(
                input_dir = self.input_dir,
                model = self.model,
                results_set = s,
                datetime_index=datetime,
                units=self.units,
            )
        
            super().set_data(data=all_results_data[s],
                             name=s)