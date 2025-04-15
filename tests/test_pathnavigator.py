import pytest
import pywrdrb

def test_pathnavigator():

    
    ### Load data used for prediction
    try:
        pn_config = pywrdrb.get_pn_config()
        expected_keys = [
            "data",
            "config",
            "timeseries_data",
            "regression_data",
            "predictions",
            "outputs"
        ]

        # Make sure keys are correct
        assert set(expected_keys) <= set(pn_config.keys()), "Some of {expected_keys} are missing in pn_config.keys(): {pn_config.keys()}"
    
    except Exception as e:
        pytest.fail(f"Missing keys: {e}")
    
    ### Get pn object
    try:
        import pathnavigator
        pn = pywrdrb.get_pn_object()
        isinstance(pn, pathnavigator.pathnavigator.PathNavigator)
    except Exception as e:
        pytest.fail(f"Fail to get pathnavigator object: {e}")
    
    
    return None