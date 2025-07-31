import pytest
import pandas as pd
import pywrdrb 
from pywrdrb.utils.results_sets import obs_results_set_opts

def test_data_load_observations():
    
    data = pywrdrb.Data(results_sets=obs_results_set_opts,
                        print_status=True)
    data.load_observations()
    
    # Make sure that each results_set is stored as an attribute
    # of the data object
    for results_set in obs_results_set_opts:
        assert hasattr(data, results_set)
        
        # Make sure that the 'obs' key exists and is a list
        results_data = getattr(data, results_set)
        assert isinstance(results_data, dict)
        assert "obs" in results_data
        assert isinstance(results_data["obs"], dict)
        
        # Make sure the scenario of 'obs' is a DataFrame
        assert isinstance(results_data["obs"][0], pd.DataFrame)
