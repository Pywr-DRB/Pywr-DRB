import pytest
import pandas as pd
import pywrdrb 
from pywrdrb.utils.results_sets import hydrologic_model_results_set_opts
from pywrdrb.load.hydrologic_model_loader import flowtype_opts

def test_data_load_hydrologic_model_flow():
    
    data = pywrdrb.Data(results_sets=hydrologic_model_results_set_opts,
                        flowtypes = flowtype_opts,
                        print_status=True)
    data.load_hydrologic_model_flow(flowtypes = flowtype_opts)
    
    # Make sure that each results_set is stored as an attribute
    # of the data object
    for results_set in hydrologic_model_results_set_opts:
        assert hasattr(data, results_set)
        
        # Make sure that the each flowtype is a key
        for flowtype in flowtype_opts:
            assert flowtype in getattr(data, results_set).keys()            