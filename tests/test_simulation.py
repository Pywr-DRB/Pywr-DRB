import pywrdrb
import pytest

@pytest.mark.dependency(depends=["test_simulation::test_pywrdrb_simulation"], 
                        scope="session")
def test_pywrdrb_simulation(test_inflow_type, shared_tmp_path):

    # test Model.load()
    model_filename = shared_tmp_path / f"{test_inflow_type}_model.json"
    model = pywrdrb.Model.load(str(model_filename))
    assert isinstance(
        model, pywrdrb.Model
    ), f"Expected Model.load() to return a pywr.Model object but got {type(model)}."
    
    # test Model.run()
    try:
        model.run()
    except Exception as e:
        pytest.fail(f"Model.run() raised an exception: {e}")
        
    # Add a recorder inherited from pywr
    # NOTE: Doo not change the filename! It is used in later tests.
    output_filename = shared_tmp_path / f"{test_inflow_type}_model_output.hdf5"
    
    try:
        recorder = pywrdrb.OutputRecorder(
            model, output_filename, 
        )
    except Exception as e:
        pytest.fail(f"OutputRecorder raised an exception: {e}")
    
    # Run simulation
    try:
        stats = model.run()
    except Exception as e:
        pytest.fail(f"Model.run() raised an exception: {e}")
    
    return