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
    input_dir = input_dir + "/"

    data = pywrdrb.Data(print_status=False,
                        input_dir=input_dir,
                        )
    data.load(datatypes=test_datatypes, 
            results_sets=test_obs_results_sets
            )
 
    for results_set in test_obs_results_sets:
        assert hasattr(data, results_set), f"Expected pywrdrb.Data object to have attribute {results_set} but it was not found."
    
    return


test_output_results_sets = [
    'major_flow',
    'res_storage'
]


def test_data_loader_with_pywrdrb_output(test_inflow_type, shared_tmp_path):
    


    data = pywrdrb.Data(print_status=True)
    
    try:
        data.load(
            datatypes=[test_inflow_type], 
            output_filenames=[shared_tmp_path / f"{test_inflow_type}_model_output.hdf5"],
            results_sets=test_output_results_sets,
            )
    except Exception as e:
        pytest.fail(f"pywrdrb.Data.load() raised an exception:\n{e}")
    
    for results_set in test_output_results_sets:
        assert hasattr(data, results_set), f"Expected pywrdrb.Data object to have attribute {results_set} but it was not found."
    
    return