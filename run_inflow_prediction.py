import pandas as pd
import numpy as np

from pywrdrb.pre.flows import _subtract_upstream_catchment_inflows
from pywrdrb.pre import PredictedInflowPreprocessor

## Predicted inflows
inflow_type_options = [
    "nhmv10",
    "nhmv10_withObsScaled",
    "nwmv21",
    "nwmv21_withObsScaled",
    "wrf1960s_calib_nlcd2016",
    "wrf2050s_calib_nlcd2016",
    "wrfaorc_calib_nlcd2016",
    "wrfaorc_withObsScaled",
    "pub_nhmv10_BC_withObsScaled",
    "pub_nhmv10_withObsScaled",
]

inflow_type_options = ["wrfaorc_withObsScaled"]

REDO_INFLOW_CALCULATION = True
REDO_INFLOW_PREDICTION = True

for inflow_type in inflow_type_options:
    print(f"Predicting inflows for: {inflow_type}")

    ## Calculate catchment inflows
    if REDO_INFLOW_CALCULATION:
        f = f"src/pywrdrb/data/flows/{inflow_type}/gage_flow_mgd.csv"
        flow_df = pd.read_csv(
            f,
            index_col=0,
            parse_dates=True,
        )

        # Iteratively subtract upstream catchment flows
        inflow_df = _subtract_upstream_catchment_inflows(flow_df)
        inflow_df.index.name = "datetime"

        # Save the inflow_df to a CSV file
        f = f"src/pywrdrb/data/flows/{inflow_type}/catchment_inflow_mgd.csv"
        inflow_df.to_csv(f)

    if REDO_INFLOW_PREDICTION:
        print(f"Generating predicted inflows for {inflow_type}...")
        # Create an instance of the PredictedInflowPreprocessor
        inflow_predictor = PredictedInflowPreprocessor(
            flow_type=inflow_type, remove_zeros=True
        )

        # Running the following will create the file:
        # src/pywrdrb/data/flows/<inflow_type>/predicted_inflow_mgd.hdf5
        inflow_predictor.load()
        inflow_predictor.process()
        inflow_predictor.save()

print("Done with inflow predictions for all data types!")
