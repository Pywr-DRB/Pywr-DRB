import pytest
import pywrdrb
from pywrdrb import PredictedInflowPreprocessor

def test_PredictedInflowPreprocessor():

    inflow_predictor = PredictedInflowPreprocessor(
        flow_type="nhmv10",
        start_date="2000-01-01",
        end_date="2003-01-01",
        modes=("regression_disagg",),
    )

    ### Load data used for prediction
    try:
        inflow_predictor.load()

        # Make sure timeseries_data is loaded correctly
        assert inflow_predictor.timeseries_data is not None, "Timeseries data not loaded."
    
    except Exception as e:
        pytest.fail(f"Loading failed: {e}")
    
    ### Train regressions
    try:
        regressions = inflow_predictor.train_regressions()
    except Exception as e:
        pytest.fail(f"Regression training failed: {e}")
    
    
    ### Make predictions
    try:
        output = inflow_predictor.make_predictions(regressions)
    except Exception as e:
        pytest.fail(f"Prediction failed: {e}")
    
    
    return 