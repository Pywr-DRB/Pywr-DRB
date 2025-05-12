"""
Preprocessor for generating NJ diversion predictions for Delaware-Raritan Canal operations.

Overview: 
This module creates lag-based predictions for NJ diversions from the 
Delaware-Raritan Canal, which are needed to properly implement the NYC and lower basin
reservoir releases, when predicted flow is less than target flow. 
It uses regression models trained on historical diversion data to forecast demand 
1-4 days ahead, which helps determine appropriate reservoir releases that account 
for downstream travel time and anticipated withdrawals.

Technical Notes: 
- Extends PredictedTimeseriesPreprocessor specifically for diversion predictions
- The historical NJ diversions are based on DR Canal gauge flow
- Uses historical NJ diversion data from the extrapolated dataset
- Creates prediction files for "demand_nj" used by FFMP parameters
- Supports multiple prediction modes (regression_disagg, perfect_foresight, etc.)
- Predictions follow the pattern "demand_nj_lag{1-4}_{mode}" 
- Output is saved as a CSV file in the data/diversions directory for use by the model

Example usage:
from pywrdrb.pre import PredictedDiversionPreprocessor
diversion_predictor = PredictedDiversionPreprocessor(
    start_date="1945-01-01",
    end_date="2023-12-31",
    modes=("regression_disagg",),
)
diversion_predictor.process()
diversion_predictor.save()


Links:
- See SI for Hamilton et al. (2024) for more details on the method formulation.

Change Log:
TJA, 2025-05-07, review+docstrings
"""

import pandas as pd
from pywrdrb.pre.predict_timeseries import PredictedTimeseriesPreprocessor

__all__ = ["PredictedDiversionPreprocessor"]

class PredictedDiversionPreprocessor(PredictedTimeseriesPreprocessor):
    """
    Predicts catchment inflows at Montague and Trenton using specified modes
    (e.g., regression, perfect foresight, moving average).
    
    
    Methods
    -------
    load()
        Load NJ diversions and catchment water consumption data.
    process()
        Run the full prediction workflow.
    save()
        Save predicted diversion time series to pywrdrb/data/diversions.
    get_prediction_node_lag_combinations()
        Return dictionary of predicted diversion column names and their defining (node, lag, mode) tuples.
    
    Attributes
    ----------
    input_dirs : dict
        Input files used for prediction.
    output_dirs : dict
        Output locations for predicted timeseries.
    timeseries_data : DataFrame
        DataFrame containing the timeseries data. Is None until load() is called.
    predicted_timeseries : DataFrame
        DataFrame containing the predicted timeseries. Is None until process() is called.
    catchment_wc : DataFrame
        DataFrame containing the average water consumption data for node catchments.
    
    
    Example usage:
    ```python
    from pywrdrb.pre import PredictedDiversionPreprocessor
    diversion_predictor = PredictedDiversionPreprocessor(
        start_date="1983-10-01",
        end_date="2016-12-31",
        modes=("regression_disagg",),
    )
    diversion_predictor.process()
    diversion_predictor.save()
    ```
    """

    def __init__(self,
                 start_date=None,
                 end_date=None,
                 modes=('regression_disagg',),
                 use_log=True,
                 remove_zeros=False,
                 use_const=False):
        """Initialize the PredictedDiversionPreprocessor.
        
        Parameters
        ----------
        start_date : str, None
            Start date for the time series. If None, match the input data.
        end_date : str, None
            End date for the time series. If None, match the input data.
        modes : tuple
            Modes to use for prediction. Default is ('regression_disagg',). Options include:
            "regression_disagg", "perfect_foresight", "moving_average", "same_day".
        use_log : bool
            Whether to use log transformation for model vars. Default is True.
        remove_zeros : bool            
            Whether to remove zero values. Default is False.
        use_const : bool
            Whether to use a constant/intercept in regression. Default is False.
        
        Returns
        -------
        None                
        """
        # flow_type is not used for diversions, 
        # but is required for the parent class
        flow_type = None
        
        # Initialize the PredictedTimeseriesPreprocessor 
        super().__init__(flow_type, 
                         start_date, 
                         end_date, 
                         use_log, 
                         remove_zeros, 
                         use_const)
        
        # List of regression modes options
        self.regression_mode_options = [
            "regression_disagg",
            "perfect_foresight",
            "moving_average",
            "same_day",
        ]

        # Modes being used; check validity        
        self.modes = modes
        for mode in self.modes:
            assert mode in self.regression_mode_options, f"Invalid regression mode: {mode}. Must be one of {self.regression_mode_options}."
    
        # Input files used for prediction
        self.input_dirs = {
            "sw_avg_wateruse_pywrdrb_catchments_mgd.csv": self.pn.catchment_withdrawals.get("sw_avg_wateruse_pywrdrb_catchments_mgd.csv"),
            "diversion_nj_extrapolated_mgd.csv" : self.pn.diversions.get("diversion_nj_extrapolated_mgd.csv"),
        }
        
        # Output locations for predicted timeseries
        self.output_dirs = {
            "predicted_diversions_mgd.csv": self.pn.get("diversions") / "predicted_diversions_mgd.csv",
        }
        

    def load(self):
        """Load NJ diversions and catchment WC data (used for structural compatibility).
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
            timeseries_data is stored as a class attribute.
        """

        ### Load NJ diversions data
        fname = self.input_dirs["diversion_nj_extrapolated_mgd.csv"]
        df = pd.read_csv(fname, parse_dates=["datetime"])
        df.index = pd.DatetimeIndex(df["datetime"])
        df["demand_nj"] = df["D_R_Canal"]

        # # subset to the input start_date if provided
        # self.start_date = self.start_date if self.start_date is not None else df.index[0]
        # self.end_date = self.end_date if self.end_date is not None else df.index[-1]
        # self.timeseries_data = subset_timeseries(df, self.start_date, self.end_date)
        self.timeseries_data = df.copy()

        # Load average water consumption data
        # used to adjust the inflow prediction, accounting for water use
        fname = self.input_dirs["sw_avg_wateruse_pywrdrb_catchments_mgd.csv"]
        wc = pd.read_csv(fname)
        wc.index = wc["node"]
        self.catchment_wc = wc


    def process(self):
        """Run full prediction workflow.
        
        Steps:
        1. Load timeseries data (if not already loaded).
        2. Train regressions on the data.
        3. Make predictions using the trained regressions.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
            The predicted_timeseries is stored as a class attribute.
        """
        if self.timeseries_data is None:
            self.load()
        regressions = self.train_regressions()
        self.predicted_timeseries = self.make_predictions(regressions)


    def save(self):
        """Save predicted diversion time series to CSV.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
            The predicted timeseries is saved to the specified output directory.
        """
        # Make sure the predictions are done successfully
        assert self.predicted_timeseries is not None, "Predicted timeseries is None. Cannot save."
        
        # Save
        fname = self.output_dirs["predicted_diversions_mgd.csv"]
        self.predicted_timeseries.to_csv(fname, index=False)

    def get_prediction_node_lag_combinations(self):
        """Return dict of predicted diversion column names formatted as (node, lag, mode) tuples.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        combos : dict
            Dictionary where keys are column names (e.g., "demand_nj_lag1_regression_disagg")
            and values are lists of tuples defining the ((node, lag), mode) for each regression.
        """
        combos = {}
        node = "demand_nj"
        for lag in [1, 2, 3, 4]:
            for mode in self.modes:
                col = f"{node}_lag{lag}_{mode}"
                combos[col] = [((node, lag), mode)]
        return combos
