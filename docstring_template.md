# NumPy Docstring Template 

Each FUNCTION/METHOD docstring should include:
- Description (single line and, if needed, longer description)
- Parameters
- Returns (unless nothing is returned)

Each CLASS docstring should include:
- Attributes
    - Key attributes that are used by the class
- Methods
    - Describe the class methods (functions) contained


The docstring _may_ include:
- Examples
    - Prioritize example only for key functions that the user will interact with.  Eg., ModelBuilder, Data, etc.  No need for examples that are 'under the hood' or not used by a user. 
- Notes
    - Focus on important notes to other developers or uses.  Could include #TODO flag here.
- Raises
    - If there are any Errors that get raised for different conditions.


## Function Docstrings

```python
def example_function(param1, param2=None, *args, **kwargs):
    """
    Short one-line summary of the function.
    
    Extended description of the function that can span multiple lines
    and provides more detailed explanation when needed.
    
    Parameters
    ----------
    param1 : type
        Description of first parameter.
    param2 : type, optional
        Description of second parameter.
        Default is None.
    *args
        Description of variable length argument list. Do NOT include type here.
    **kwargs
        Description of arbitrary keyword arguments. Do NOT include type here. 
        
    Returns
    -------
    type
        Description of the return value.
        
    Raises
    ------
    ExceptionType
        When and why this exception is raised.
        
    See Also
    --------
    related_function : Description of related function.
    another_function : Description of another related function.
        
    Notes
    -----
    Additional information about the function, implementation details, 
    algorithm specifics, mathematical formulas, etc.
    
    Examples
    --------
    >>> example_function(10)
    result
    
    >>> example_function('string', param2=True)
    another_result
    """
    # Function implementation here
    pass
```

## Class Docstring

```python
class ExampleClass:
    """
    Short one-line summary of the class.
    
    Extended description of the class functionality and purpose.
    Should explain the general use case and behavior.
    
    Attributes
    ----------
    attr1 : type
        Description of first attribute.
    attr2 : type
        Description of second attribute.
        
    Methods
    -------
    method1(param1, param2)
        Brief description of method1.
    method2()
        Brief description of method2.
    """
    
    def __init__(self, param1, param2=None):
        """
        Initialize an instance of ExampleClass.
        
        Parameters
        ----------
        param1 : type
            Description of first parameter.
        param2 : type, optional
            Description of second parameter.
            Default is None.
            
        Notes
        -----
        The __init__ method doesn't include a Returns section
        as it always returns None.
        """
        self.attr1 = param1
        self.attr2 = param2
    
    def method1(self, param1):
        """
        Short one-line summary of method1.
        
        Extended description of the method functionality.
        This can extend to multiple lines as needed, but
        avoid having very long lines. 
        
        Parameters
        ----------
        param1 : type
            Description of parameter.
            
        Returns
        -------
        type
            Description of the return value.
            
        Raises
        ------
        ExceptionType
            When and why this exception is raised.
        """
        # Method implementation here
        pass
```

## Module-Level Docstring
This should be at the top of each file. 


```python
"""
Short one-line summary of the module content. 

Extended description of the module functionality and purpose.
Focus on how this module interacts with other pywrdrb modules and code. 
Should explain what the module does and its overall organization.


Classes
-------
Class1
    Brief description of Class1.
Class2
    Brief description of Class2.

Functions
---------
Class1.function1(param1, param2)
    Brief description of function1.
function2()
    Brief description of function2.

"""
```

## Sections Reference

### Common Sections
- `Parameters`: Function/method inputs
- `Returns`: Function/method outputs
- `Raises`: Exceptions that might be raised
- `See Also`: Related functions/classes
- `Notes`: Additional information
- `Examples`: Usage examples
- `Attributes`: Class attributes (for classes)
- `Methods`: Class methods (for classes)

### Special Cases
- `Other Parameters`: Less commonly used parameters
- `Warns`: Warnings that might be issued
- `Yields`: For generators (instead of `Returns`)
- `Receives`: For callbacks
- `References`: Citations or references

## Format Rules

1. **Section Headers**: Use the section name followed by a line of dashes (e.g., `Parameters\n----------`)
2. **Parameter Format**: `name : type[, optional]`
3. **Indentation**: 4 spaces for descriptions
4. **Types**: Use Python types (`list`, `dict`, etc.) or specific types (`numpy.ndarray`, `pandas.DataFrame`)
5. **Default Values**: For optional parameters, mention "Default is value."

## Tips

1. Start with a one-line summary, then a blank line, then a more detailed description
2. Use full sentences with periods
3. Include type information in Parameters/Returns
4. Include default values for optional parameters
5. For `__init__` methods, document parameters but not returns
6. Use Examples section for executable code examples


## Example: pywrdrb.Data()

The following example code is taken from `pywrdrb/load/data_loader.py`.




```python
"""
The pywrdrb.Data() class is used to load and store different datasets. 

The class supports loading the following data:
- Observations (e.g., streamflows, reservoir storage)
- pywrdrb simulation output (any variable from the output file)
- #TODO Hydrologic model output (e.g., NHMv10, NWMv21) (not implemented correctly at the moment)

All data is stored as attributes of the Data class. The data is stored in a hierarchical format, following:
Data.results_set[datatype][scenario_id] -> pd.DataFrame.


Importantly, the Data class uses results_set specifications. These result_set keys are used to identify
specific variable subsets from the different data. To learn more about these, see:
https://pywr-drb.github.io/Pywr-DRB/results_set_options.html


Classes
-------
Data
    The data loader class. Interacts with other 'loader' classes all built on top of the AbstractDataLoader class.
    
Functions
---------
Data.load_observations()
    Load observational data from the specified results_sets.
Data.load_output(output_filenames)
    Load data from pywrdrb output files based on the specified results_sets.
Data.export(file)
    Export all loaded data to an HDF5 file.
Data.load_from_export(file)
    Load data from an HDF5 file into the object.
"""



import pandas as pd
import os
import pywrdrb
from pywrdrb.load.abstract_loader import AbstractDataLoader
from pywrdrb.load import Output, Observation

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
    """
    A data loader for hydrologic data from various sources.
    
    This class provides methods to load observation data, pywrdrb output data,
    and functionality to export/import data to/from HDF5 files.
    
    Attributes
    ----------
    pn : object
        A pathnavigator object for handling file paths.
    all_results_sets : list
        Combined list of all valid result sets across all data types.
    default_kwargs : dict
        Default keyword arguments used by the loader.
    results_sets : list
        List of result sets to load.
    output_filenames : list
        List of pywrdrb output filenames to load.
    units : str
        Units for the results.
    print_status : bool
        Whether to print status updates.
    """
    
    def __init__(self, pn=pn, **kwargs):
        """
        Initialize the Data loader with default and provided keyword arguments.

        Parameters
        ----------
        pn : object, optional
            A pathnavigator object for handling file paths.
            Default is the global pn object.
        results_sets : list, optional
            List of results sets to load.
        output_filenames : list, optional
            List of pywrdrb output filenames, with path, to load.
            Only necessary for Data.load_output().
        units : str, optional
            Units for the results. Default is 'MG' (Million Gallons).
        print_status : bool, optional
            Whether to print status updates. Default is False.
            
        Examples
        --------
        >>> from pywrdrb import Data
        >>> 
        >>> # For loading observations data
        >>> data = Data(results_sets=['major_flow'], print_status=True)
        >>> data.load_observations()
        >>> 
        >>> # For output data, must provide output filenames
        >>> f = "./output_data/drb_output_nhmv10.hdf5"
        >>> data.load_output(output_filenames=[f])
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
        
        Parameters
        ----------
        datatype : str
            The datatype of interest.
            Options: 'output', 'obs', 'nhmv10', 'nwmv21'.

        Returns
        -------
        list
            A subset of results_sets that are valid for the specified datatype.
            
        Raises
        ------
        ValueError
            If an invalid datatype is provided.
        """
        if datatype not in all_valid_results_set_opts:
            raise ValueError(f"Invalid datatype specified: {datatype}")
        return all_valid_results_set_opts[datatype]
    
    
    def __get_results_sets_subset__(self, datatype):
        """
        Get a subset of results_set names that are valid for the datatype and not already stored.
        
        Filters the results_sets to include only those that are:
        1. Valid for the specified datatype
        2. Not already stored in the object for that datatype
        
        Parameters
        ----------
        datatype : str
            The datatype of interest.
            Options: 'output', 'obs', 'nhmv10', 'nwmv21'.

        Returns
        -------
        list
            A filtered subset of results_sets for the specified datatype
            that have not yet been loaded.
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
        Print status message if print_status flag is True.

        Parameters
        ----------
        message : str
            Message to be printed.
        
        Returns
        -------
        None
        """
        if self.print_status:
            print(message)
            
    
    def __set_loader_data__(self, loader):
        """
        Set data stored in a loader object as attributes of this class.

        Parameters
        ----------
        loader : AbstractDataLoader
            A data loader object with results_set attributes.
            
        Returns
        -------
        None
            The data is stored as attributes of this object.
        """
        for results_set in loader.results_sets:
            self.set_data(getattr(loader, results_set), results_set)
    
    
    def load_observations(self, **kwargs):
        """
        Load observational data.
        
        This method loads observation data based on the specified results_sets.
        It uses the pathnavigator to locate the observation data directory and
        creates an Observation loader to handle the data loading process.
        
        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments to override instance attributes.
            See class initialization for available parameters.
        
        Returns
        -------
        None
            The loaded data is stored as attributes of this object.
        """
        self.__parse_kwargs__(default_kwargs=self.default_kwargs,
                                **kwargs)
        
        results_sets_subset = self.__get_results_sets_subset__('obs')
        
        self.__print_status__(f"Loading observations data sets {results_sets_subset}...")
        
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
        Load data from pywrdrb output files.
        
        This method loads data from the specified output files based on the
        results_sets. It creates an Output loader to handle the data loading process.
        
        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments to override instance attributes.
            See class initialization for available parameters.
            
        Returns
        -------
        None
            The loaded data is stored as attributes of this object.
            
        Raises
        ------
        AssertionError
            If output_filenames list is not provided.
            
        Notes
        -----
        The output_filenames parameter must be provided either during initialization
        or when calling this method. Each file should be an HDF5 file generated
        by pywrdrb.
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
    
        
    def export(self, file):
        """
        Export all data stored in this object to an HDF5 file.
        
        Exports all results sets and their associated data to an HDF5 file,
        preserving the hierarchical structure of the data.
        
        Parameters
        ----------
        file : str
            Path to the new HDF5 file to create.
            
        Returns
        -------
        None
            The data is written to the specified file.
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
        
        Loads data from an HDF5 file previously created using the Data.export()
        method. The data is loaded such that it maintains the original structure 
        of the Data object after using the load_*() methods.
        
        Parameters
        ----------
        file : str
            Path to the HDF5 file to load.
            
        Returns
        -------
        None
            The loaded data is stored as attributes of this object.
            
        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
            
        Notes
        -----
        The method expects the HDF5 file to have the structure:
        /{results_set}/{datatype}/{scenario_id}
        
        This structure is created by the Data.export() method.
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
    
```