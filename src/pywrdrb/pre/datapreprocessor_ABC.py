"""
Template for creating a new data preprocessor class in pywrdrb.

Overview: 
We use an abstract base class (ABC) to define the structure for data preprocessors.
This class should be inherited by all data preprocessor classes in the pywrdrb package, 
which will force the customized data preprocessor to have the `load`, `process`, and 
`save` methods.
 
Change Log:
Chung-Yi Lin, 2025-05-02, None
"""

from abc import ABC, abstractmethod
from pywrdrb.path_manager import get_pn_object

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
        Load the raw data from the data directories defined in `self.input_dirs`. 
        For data retrieval from API, please allow keyword arguments to be passed to
        the function for flexibility. E.g., whether to retrieve the latest data or data
        for a specific date range.
        """
        pass

    @abstractmethod
    def process(self, **kwargs):
        """
        Do the data processing. This method should be implemented to process the
        raw data loaded from the `load` method. The processed data should be stored in
        `self.processed_data`.
        """
        pass

    @abstractmethod
    def save(self, **kwargs):
        """
        This method should save the processed data to the output directories defined in 
        `self.output_dirs`.
        """
        pass
