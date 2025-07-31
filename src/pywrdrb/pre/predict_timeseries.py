"""
Abstract base class for time series prediction using autoregressive models.

Overview: 
This module provides a base class for creating prediction workflows that generate 
lag-based forecasts for different time series. It is used for prediction classes like 
PredictedInflowPreprocessor and PredictedDiversionPreprocessor, 
provides common regression training, prediction, and data management functionality.

Technical Notes: 
- Extends the DataPreprocessor to follow the load, process, save workflow
- Uses statsmodels.api for regression modeling with multiple prediction modes
- Supports various prediction strategies: regression, perfect foresight, moving average, same-day
- Handles log transformations, zero removal, and optional constant terms in regression
- Each class has a standardized data format with predictions stored in self.predicted_timeseries

Links:
- See SI for Hamilton et al. (2024) for more details on the method formulation.

Change Log:
TJA, 2025-05-07, Minor fixes + docstrings
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from abc import abstractmethod

from pywrdrb.utils.timeseries import subset_timeseries
from pywrdrb.pre.datapreprocessor_ABC import DataPreprocessor

class PredictedTimeseriesPreprocessor(DataPreprocessor):
    """
    Used to generated lag-based autoregressive predictions for time series data.
    
    Methods
    -------
    load()
        Abstract method to Load data for training and prediction.
    process()
        Abstract method to run the full prediction workflow.
    save()
        Abstract method to save the predicted timeseries data.
    get_prediction_node_lag_combinations()
        Abstract method to return a dictionary of predicted timeseries column names and their defining (node, lag, mode) tuples.
    _fit_regression(df, node, lag)
        Fit AR regression model for a given node and lag.
    _predict_value(idx, node, lag, mode, regressions)
        Generate a single prediction value for a given time index, node, lag, and mode.
    _regression_prediction(x, const, slope)
        Generate a regression prediction value using input x, constant, and slope.
    _unique_node_lag_pairs()
        Filter the node-lag pairs to only include unique, non-negative lags.    
    """
    def __init__(self, 
                 flow_type=None, 
                 start_date=None, 
                 end_date=None,
                 use_log=True, 
                 remove_zeros=False, 
                 use_const=False):
        """Initialize the PredictedTimeseriesPreprocessor.
        
        Parameters
        ----------
        flow_type : str
            Label for the dataset.
        start_date : str, None
            Start date for the time series. If None, match the input data.
        end_date : str, None
            End date for the time series. If None, match the input data.
        use_log : bool
            Whether to use log transformation of training data. Default is True.
        remove_zeros : bool
            Whether to remove zero values from training data. Default is False.
        use_const : bool
            Whether to use a constant/intercept in regression. Default is False.
        
        Returns
        -------
            None
        """
        super().__init__()
        self.flow_type = flow_type
        self.start_date = start_date
        self.end_date = end_date
        self.use_log = use_log
        self.remove_zeros = remove_zeros
        self.use_const = use_const

        self.timeseries_data = None
        self.catchment_wc = None
        self.predicted_timeseries = None

    @abstractmethod
    def load(self):
        """Load data for training and prediction."""
        pass
    
    @abstractmethod
    def save(self):
        """Save the predicted timeseries data."""
        pass
    
    @abstractmethod
    def process(self):
        """Run the full prediction workflow."""
        pass

    @abstractmethod
    def get_prediction_node_lag_combinations(self):
        """Generate a dictionary of predicted timeseries column names and their defining (node, lag, mode) tuples."""
        pass

    def train_regressions(self):
        """Train the AR models for different node, lag combinations.
        
        Returns
        -------
            dict: A dictionary of regression coefficients for each (node, lag) pair.
            The keys are tuples of (node, lag) and the values are dictionaries with "const" and "slope" keys.
        """
        training_start_date = self.start_date if self.start_date else self.timeseries_data.index[0]
        training_end_date = self.end_date if self.end_date else self.timeseries_data.index[-1]
        
        regressions = {}
        df = subset_timeseries(self.timeseries_data, 
                               training_start_date, training_end_date)
        
        for (node, lag) in self._unique_node_lag_pairs():
            const, slope = self._fit_regression(df, node, lag)
            regressions[(node, lag)] = {"const": const, "slope": slope}
        return regressions

    def _fit_regression(self, df, node, lag):
        """Fit AR regression model for a given node and lag.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame containing the time series data.
        node : str
            The name of the node to fit the regression for.
        lag : int
            The lag to use for the regression.
        
        Returns
        -------
        tuple
            A tuple containing the constant and slope of the regression.
        """
        
        Y = df[node].iloc[lag:].values.astype(float)
        X = df[node].iloc[:-lag].values if lag > 0 else df[node].values

        if lag == 0:
                    Y = Y[:len(X)]
                    
        if self.use_const:
            X = np.column_stack((np.ones(len(X)), X))

        if self.remove_zeros:
            if self.use_const:
                mask = (Y > 0.01) & (X[:, 1] > 0.01)
                Y, X = Y[mask], X[mask]
            else:
                mask = (Y > 0.01) & (X > 0.01)
                Y, X = Y[mask], X[mask]

        if self.use_log:
            eps = 0.001
            Y, X = np.log(Y + eps), np.log(X + eps)

        model = sm.OLS(Y, X, hasconst=self.use_const).fit()
        if self.use_const:
            return float(model.params[0]), float(model.params[1])
        return float(0.0), float(model.params[0])

    def make_predictions(self, regressions):
        """Generate lead-time predictions using the timeseries data and trained models.
        
        Parameters
        ----------
        regressions : dict
            A dictionary of regression coefficients for each (node, lag) pair.
            The keys are tuples of (node, lag) and the values are dictionaries with "const" and "slope" keys.
        
        Returns
        -------
        DataFrame
            A DataFrame containing the predicted timeseries data.
        """
        # Setip the prediction dataframe
        index = self.timeseries_data.index
        
        # use state_date and end_date if not None
        if self.start_date is not None:
            index = index[index >= self.start_date]
        if self.end_date is not None:
            index = index[index <= self.end_date]
        
        pred_df = pd.DataFrame({"datetime": index})
        node_lags = self.get_prediction_node_lag_combinations()

        for col, node_lag_mode_list in node_lags.items():
            pred_df[col] = np.zeros(len(index))
            for (node, lag), mode in node_lag_mode_list:
                pred_df[col] += np.array([
                    self._predict_value(idx, node, lag, mode, regressions)
                    for idx in range(len(index))
                ])
        return pred_df

    def _predict_value(self, idx, node, lag, mode, regressions):
        """Generate a single prediction value for a given timeidex, node, lag, and mode.
        
        Parameters
        ----------
        idx : int
            The index of the time series data to predict.
        node : str
            The name of the node to predict for.
        lag : int
            The lag to use for the prediction.
        mode : str
            The prediction mode to use (e.g., "same_day", "perfect_foresight", "regression", "moving_average").
        regressions : dict
            A dictionary of regression coefficients for each (node, lag) pair.
            The keys are tuples of (node, lag) and the values are dictionaries with "const" and "slope" keys.
        
        Returns
        -------
        float
            The predicted value for the given time index, node, and lag.
            
        Notes
        -----
        When 'node' has a non-None catchment water consumption, then 
        the predicted value is adjusted by the catchment water consumption ratio.
        """
        
        n = self.timeseries_data.shape[0]
        val_t = self.timeseries_data[node].iloc[idx]

        if mode == "same_day":
            value = val_t

        elif mode == "perfect_foresight":
            value = self.timeseries_data[node].iloc[min(idx + lag, n - 1)]

        elif mode.startswith("regression"):
            const = regressions[(node, lag)]["const"]
            slope = regressions[(node, lag)]["slope"]
            value = self._regression_prediction(val_t, const, slope)

        elif mode == "moving_average":
            start = max(0, idx - 6)
            value = self.timeseries_data[node].iloc[start:idx+1].mean()

        else:
            raise ValueError(f"Unknown mode: {mode}")

        if node in self.catchment_wc.index:
            wd = self.catchment_wc.loc[node, "Total_WD_MGD"]
            cu = self.catchment_wc.loc[node, "Total_CU_WD_Ratio"]
            prev_val = self.timeseries_data[node].iloc[max(idx - 1, 0)]
            value -= min(value, cu * min(prev_val, wd))

        return value

    def _regression_prediction(self, x, const, slope):
        """Generate a regression prediction value using input x, constant, and slope.
        
        Parameters
        ----------
        x : float
            The input value for the regression prediction.
        const : float
            The constant term from the regression model.
        slope : float
            The slope term from the regression model.
        
        Returns
        -------
        float
            The predicted value based on the regression model.
        """
        
        if self.use_log:
            x = max(x, 0.001)
            x = float(x)
            
            try:
                y = np.exp(const + slope * np.log(x))
                return y
            except:
                print(f'Failed with\n const:{const}\n slope:{slope}\n x:{x}')        
            
        return const + slope * x

    def _unique_node_lag_pairs(self):
        """Filter the node-lag pairs to only include unique, non-negative lags."""
        pairs = set()
        for combos in self.get_prediction_node_lag_combinations().values():
            for (node, lag), _ in combos:
                if lag >= 0:
                    pairs.add((node, lag))
        return pairs
