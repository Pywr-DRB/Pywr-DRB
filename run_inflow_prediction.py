import pandas as pd
import numpy as np
import pywrdrb
from pywrdrb.pre import (
    PredictedInflowPreprocessor,
    ExtrapolatedDiversionPreprocessor,
    PredictedDiversionPreprocessor
)

## Predicted inflows
model_date_ranges = pywrdrb.utils.dates.model_date_ranges
inflow_type_options = list(model_date_ranges.keys())

REDO_INFLOW_PREDICTION = True
REDO_DIVERSION_EXTRAPOLATION = True

remove_zeros_during_prediction = False

if REDO_DIVERSION_EXTRAPOLATION:

    # # Extrapolate NYC and NJ diversions 
    # # No flow_type -> uses historical data    
    # # NYC demand
    # nyc_extrapolator = ExtrapolatedDiversionPreprocessor(loc="nyc",)
    # nyc_extrapolator.load()
    # nyc_extrapolator.process()
    # nyc_extrapolator.save()


    # # Repeat for NJ
    # nj_extrapolator = ExtrapolatedDiversionPreprocessor(loc="nj",)
    # nj_extrapolator.load()
    # nj_extrapolator.process()
    # nj_extrapolator.save()

    # Now we need to predict the diversions to inform operations
    nyc_pred = PredictedDiversionPreprocessor(modes=('regression_disagg', 'perfect_foresight'),
                                              remove_zeros=remove_zeros_during_prediction)
    nyc_pred.load()
    nyc_pred.process()
    nyc_pred.save()


for inflow_type in inflow_type_options:
    print(f"Predicting inflows for: {inflow_type}")

    if REDO_INFLOW_PREDICTION:
        print(f"Generating predicted inflows for {inflow_type}...")
        # Create an instance of the PredictedInflowPreprocessor
        inflow_predictor = PredictedInflowPreprocessor(
            flow_type=inflow_type, 
            remove_zeros=remove_zeros_during_prediction,
            modes=('regression_disagg', 'perfect_foresight'),
        )

        # Running the following will create the file:
        # src/pywrdrb/data/flows/<inflow_type>/predicted_inflow_mgd.hdf5
        inflow_predictor.load()
        inflow_predictor.process()
        inflow_predictor.save()


print("Done with inflow predictions for all data types!")
