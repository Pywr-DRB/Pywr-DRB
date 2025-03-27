
# Need to be reorganized!
#from .disaggregate_DRBC_demands import disaggregate_DRBC_demands
#from .extrapolate_NYC_NJ_diversions import extrapolate_NYC_NJ_diversions
#from .predict_inflows_diversions import predict_inflows_diversions, predict_ensemble_inflows_diversions
#from .prep_input_data_functions import (
#    read_modeled_estimates,
#    read_csv_data,
#    match_gages,
#    subtract_upstream_catchment_inflows,
#    add_upstream_catchment_inflows,
#    create_hybrid_modeled_observed_datasets
#)


from abc import ABC, abstractmethod
from .. import get_pn_object

class DataPreprocessor(ABC):
    def __init__(self):
        """
        Abstract class for data preprocessor.
        
        Attributes
        ----------
        input_dirs : dict
            Dictionary with filenames as keys and input directories as values.
        output_dirs : dict
            Dictionary with filenames as keys and output directories as values.
        raw_data : dict
            Dictionary to store raw data loaded from input directories.
        processed_data : dict
            Dictionary to store processed data.
        _dirs : pathnavigator.PathNavigator
            The global directories object.

        Methods
        -------
        load(**kwargs)
            Load raw data from directories or retrieve from API.
        process(**kwargs)
            Process the loaded raw data.
        save(**kwargs)
            Save the processed data to output directories.
        """
        # The following attributes should be predefined in each of the preprocessor 
        # classes, allowing users to overrite them as needed.
        
        # The input directories for the raw data. Filename as key and directory as value.
        self.input_dirs = {}
        # The output directories for the processed data. Filename as key and directory as value.
        self.output_dirs = {}
        # The raw data loaded from the input directories.
        self.raw_data = {}
        # The processed data.
        self.processed_data = {}
        
        # Get the global directories object
        self.pn = get_pn_object()
    
    @abstractmethod
    def load(self, **kwargs):
        """
        Load the raw data from the data directories and/or retrieve the latest data from
        API. For data retrieval from API, please allow keyword arguments to be passed to
        the function for flexibility. E.g., whether to retrieve the latest data or data
        for a specific date range.
        """
        pass

    @abstractmethod
    def process(self, **kwargs):
        pass

    @abstractmethod
    def save(self, **kwargs):
        pass
