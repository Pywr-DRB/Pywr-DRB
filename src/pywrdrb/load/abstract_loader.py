from abc import ABC, abstractmethod

from pywrdrb.utils.verify_file import verify_file_exists


class AbstractDataLoader(ABC):
    
    @abstractmethod
    def __init__(self, **kwargs):
        pass        
    
    
    def __parse_kwargs__(self, 
                         default_kwargs, 
                         **kwargs):
        """
        Parses and sets the provided keyword arguments as attributes,
        using the provided kwargs, existing attributes, or default values in that order.

        Args:
            default_kwargs (dict): Default keyword arguments with default values.

        Keyword Args:
            kwargs: User-provided keyword arguments to override defaults.

        Returns:
            None
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

        Args:
            valid_results_set_opts (list): Valid options for results sets.

        Returns:
            None

        Raises:
            ValueError: If an invalid results set is provided.
        """
        for s in self.results_sets:
            if s not in valid_results_set_opts:
                err_msg = (
                    f"Specified results set {s} in results_sets is not a valid option. "
                )
                err_msg += f"Valid options are: {valid_results_set_opts}"
                raise ValueError(err_msg)
        return
    
    
    def __verify_files_exist__(self, files):
        for file in files:
            verify_file_exists(file)
    

    def set_data(self, 
                 data, 
                 name):
        """
        Store or update data in the object as an attribute.

        Args:
            data (dict): Data to set or update, with format dict{dict{0: pd.DataFrame}}.
            name (str): Attribute name for the data.

        Returns:
            None
        """
        
        # if not already an attribute, setattr
        if not hasattr(self, name):
            setattr(self, name, data)
            
        elif hasattr(self, name):
            if getattr(self, name) is not None:
                getattr(self, name).update(data)
            else:
                setattr(self, name, data)
        return


    @abstractmethod
    def load(self):
        """
        Abstract method to load data and store data as object attributes. 
        Must be implemented in subclasses.

        Returns:
            None
        """
        pass

