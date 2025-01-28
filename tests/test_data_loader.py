import pytest

import pywrdrb

test_obs_results_sets = [
    'major_flow',
    'res_storage'
]

test_datatypes = [
    'nhmv10',
    'nwmv21',
    'obs'
]

def test_data_loader_stores_results_sets_as_attributes():
    
    directories = pywrdrb.get_directory()
    input_dir = directories.input_dir
    
    data = pywrdrb.Data(print_status=False,
                        input_dir=input_dir,
                        )
    data.load(datatypes=test_datatypes, 
            results_sets=test_obs_results_sets
            )
 
    for results_set in test_obs_results_sets:
        assert hasattr(data, results_set), f"Expected pywrdrb.Data object to have attribute {results_set} but it was not found."
    
    return