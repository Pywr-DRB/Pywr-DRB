import pandas as pd
from pywrdrb.utils.timeseries import subset_timeseries, get_rollmean_timeseries
from pywrdrb.pre.prep_input_data_functions import add_upstream_catchment_inflows
from pywrdrb.utils.directories import input_dir
from .predict_timeseries import PredictedTimeseriesPreprocessor

__all__ = ["PredictedDiversionPreprocessor"]

class PredictedDiversionPreprocessor(PredictedTimeseriesPreprocessor):
    """
    Predicts catchment inflows at Montague and Trenton using specified modes
    (e.g., regression, perfect foresight, moving average).
    
    
    Example usage:
    ```python
    diversion_predictor = PredictedDiversionPreprocessor(
        flow_type="nhmv10",
        start_date="2000-01-01",
        end_date="2003-01-01",
        modes=("regression_disagg",),
    )
    
    diversion_predictor.process()
    ```
    """

    def __init__(self,
                 flow_type,
                 start_date=None,
                 end_date=None,
                 modes=('regression_disagg',),
                 use_log=True,
                 remove_zeros=False,
                 use_const=False):
        """
        
        
        Args:
            flow_type (str): Label for the dataset.
            start_date (bool, None): Start date for the time series. If None, match the input data.
            end_date (bool, None): End date for the time series. If None, match the input data.
            modes (tuple): Modes to use for prediction. Default is ('regression_disagg',).
            use_log (bool): Whether to use log transformation. Default is True.
            remove_zeros (bool): Whether to remove zero values. Default is False.
            use_const (bool): Whether to use a constant in regression. Default is False.
        
        Returns:
            None        
        """
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
        """Load NJ diversions and catchment WC data (used for structural compatibility)."""

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



    def save(self):
        """Save predicted diversion time series to CSV."""
        # Make sure the predictions are done successfully
        assert self.predicted_timeseries is not None, "Predicted timeseries is None. Cannot save."
        
        # Save
        fname = self.output_dirs["predicted_diversions_mgd.csv"]
        self.predicted_timeseries.to_csv(fname, index=False)


    def process(self):
        """Run full prediction workflow."""
        self.load()
        regressions = self.train_regressions()
        self.predicted_timeseries = self.make_predictions(regressions)
        self.save()


    def get_prediction_node_lag_combinations(self):
        """
        Return dictionary of predicted diversion column names and 
        their defining (node, lag, mode) tuples.
        All combinations predict from demand_nj using self.modes.
        """
        combos = {}
        node = "demand_nj"
        for lag in [1, 2, 3, 4]:
            for mode in self.modes:
                col = f"{node}_lag{lag}_{mode}"
                combos[col] = [((node, lag), mode)]
        return combos
