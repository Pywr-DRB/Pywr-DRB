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

from pywrdrb.utils.lists import reservoir_list, majorflow_list
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

        Returns empty dict if no regression modes are used.

        Returns
        -------
            dict: A dictionary of regression coefficients for each (node, lag) pair.
            The keys are tuples of (node, lag) and the values are dictionaries with "const" and "slope" keys.
        """
        # Check if any regression modes need training
        has_regression = any(mode.startswith("regression") for mode in self.modes)

        if not has_regression:
            # No regression modes - skip training
            return {}

        training_start_date = self.start_date if self.start_date else self.timeseries_data.index[0]
        training_end_date = self.end_date if self.end_date else self.timeseries_data.index[-1]

        regressions = {}
        df = subset_timeseries(self.timeseries_data,
                               training_start_date, training_end_date)

        for (node, lag) in self._unique_node_lag_pairs():
            # When lag < 0, we are 'predicting' past values so no regression needed
            # we will just use actual observations
            if lag < 0:
                continue
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

        ## Check if we have enough data points to fit the regression
        if len(Y) < 10 or len(X) < 10:
            summary_msg = f"Not enough data points to fit regression after zero removal for\nnode:{node}\nlag:{lag}\n"
            summary_msg += f"After removing zeros, we have {len(Y)} samples. Consider setting remove_zeros=False or adjusting the threshold."
            raise ValueError(summary_msg)

        if self.use_log:
            eps = 0.001
            Y, X = np.log(Y + eps), np.log(X + eps)

        # ### Print summary:
        # print(f'Fitting regression for node:{node}, lag:{lag}')
        # print(f'  Number of training samples: len(Y) = {len(Y)} | len(X) = {len(X)}')

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
        # Setup the prediction dataframe
        # Use gage_data index if timeseries_data is None (perfect_foresight only mode)
        if self.timeseries_data is not None:
            index = self.timeseries_data.index
        elif hasattr(self, 'gage_data') and self.gage_data is not None:
            index = self.gage_data.index
        else:
            raise ValueError("No data loaded for making predictions")

        # use start_date and end_date if not None
        if self.start_date is not None:
            index = index[index >= self.start_date]
        if self.end_date is not None:
            index = index[index <= self.end_date]

        pred_df = pd.DataFrame({"datetime": index})
        node_lags = self.get_prediction_node_lag_combinations()

        for col, node_lag_mode_list in node_lags.items():
            pred_df[col] = np.zeros(len(index))
            for (node, lag), mode in node_lag_mode_list:

                predicted_node_lag_flows= [
                    self._predict_value(idx, index[idx], node, lag, mode, regressions)
                    for idx in range(len(index))
                    ]

                pred_df[col] += np.array(predicted_node_lag_flows)
                
                ### Print summary:
                # print(f'Predicting {col} using node:{node}, lag:{lag}, mode:{mode}')
                # print(f'  First 5 predicted values: {predicted_node_lag_flows[:5]}')
                # print(f'  Last 5 predicted values: {predicted_node_lag_flows[-5:]}')
                # print(f'  Mean predicted value: {np.mean(predicted_node_lag_flows)}')
                # print(f'  Min predicted value: {np.min(predicted_node_lag_flows)}')
                # print(f'  Max predicted value: {np.max(predicted_node_lag_flows)}')
                # print('---'*20)
        return pred_df

    def _predict_value(self, idx, date_t, node, lag, mode, regressions):
        """Generate a single prediction value for a given time index, node, lag, and mode.

        Parameters
        ----------
        idx : int
            The position index in the prediction array.
        date_t : pd.Timestamp
            The date for this prediction.
        node : str
            The name of the node to predict for.
        lag : int
            The lag to use for the prediction.
        mode : str
            The prediction mode to use (e.g., "same_day", "gage_flow", "perfect_foresight", "regression", "moving_average").
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

        # Determine which data source to use based on mode and node
        use_gage_data = (mode in ("gage_flow", "perfect_foresight") and
                        hasattr(self, 'gage_data') and
                        self.gage_data is not None and
                        node in ['delMontague', 'delTrenton'])

        if use_gage_data:
            data_source = self.gage_data
            val_t = data_source.loc[date_t, node] if mode != "gage_flow" else None
        else:
            data_source = self.timeseries_data
            val_t = data_source.loc[date_t, node]

        # Initialize prediction variables for all modes
        # Both are needed by the catchment water consumption logic below
        Yhat_lag_prediction = None
        Yhat_lag_minus1_prediction = None

        if mode == "same_day":
            Yhat_lag_prediction = val_t
            Yhat_lag_minus1_prediction = val_t

        elif mode in ("gage_flow", "perfect_foresight"):
            # Use actual observations as predictions.
            # Note: PredictedInflowPreprocessor overrides _predict_value to handle
            # perfect_foresight differently (using STARFIT-simulated releases for
            # reservoir nodes). This base class implementation is used by
            # PredictedDiversionPreprocessor where perfect_foresight == raw demand data.
            # Date-based indexing (loc) is used instead of positional (iloc) to avoid
            # bugs when start_date/end_date subset the data.
            date_lag = date_t + pd.Timedelta(days=lag)
            date_lag_minus_1 = date_t + pd.Timedelta(days=lag - 1)

            if use_gage_data:
                if date_lag in data_source.index:
                    Yhat_lag_prediction = data_source.loc[date_lag, node]
                else:
                    Yhat_lag_prediction = data_source[node].iloc[-1]

                if date_lag_minus_1 in data_source.index:
                    Yhat_lag_minus1_prediction = data_source.loc[date_lag_minus_1, node]
                else:
                    Yhat_lag_minus1_prediction = data_source[node].iloc[-1]
            else:
                data_source = self.timeseries_data
                if date_lag in data_source.index:
                    Yhat_lag_prediction = data_source.loc[date_lag, node]
                else:
                    Yhat_lag_prediction = data_source[node].iloc[-1]

                if date_lag_minus_1 in data_source.index:
                    Yhat_lag_minus1_prediction = data_source.loc[date_lag_minus_1, node]
                else:
                    Yhat_lag_minus1_prediction = data_source[node].iloc[-1]

        elif mode.startswith("regression"):

            ### Handle negative lag (past) days
            # When lag < 0, we are 'predicting' past values so we use actual observations
            if lag <= 0:
                date_lag = date_t + pd.Timedelta(days=lag)
                date_lag_minus_1 = date_t + pd.Timedelta(days=lag - 1)

                if date_lag in self.timeseries_data.index:
                    Yhat_lag_prediction = self.timeseries_data.loc[date_lag, node]
                else:
                    Yhat_lag_prediction = self.timeseries_data[node].iloc[0]

                if date_lag_minus_1 in self.timeseries_data.index:
                    Yhat_lag_minus1_prediction = self.timeseries_data.loc[date_lag_minus_1, node]
                else:
                    Yhat_lag_minus1_prediction = self.timeseries_data[node].iloc[0]
            elif lag > 0:

                const = regressions[(node, lag)]["const"]
                slope = regressions[(node, lag)]["slope"]
                Yhat_lag_prediction = self._regression_prediction(val_t, const, slope)

                # need to get the lag-1 prediction for the autoregressive model
                if lag==1:
                    Yhat_lag_minus1_prediction = val_t
                else:
                    const = regressions[(node, lag - 1)]["const"]
                    slope = regressions[(node, lag - 1)]["slope"]
                    Yhat_lag_minus1_prediction = self._regression_prediction(val_t, const, slope)

        elif mode == "moving_average":
            date_start = date_t - pd.Timedelta(days=6)
            date_end = date_t
            Yhat_lag_prediction = self.timeseries_data.loc[date_start:date_end, node].mean()

            date_start_minus1 = date_t - pd.Timedelta(days=7)
            date_end_minus1 = date_t - pd.Timedelta(days=1)
            Yhat_lag_minus1_prediction = self.timeseries_data.loc[date_start_minus1:date_end_minus1, node].mean()

        else:
            raise ValueError(f"Unknown mode: {mode}")

        ### Account for catchment water consumption if applicable
        if node in reservoir_list + majorflow_list:
            pywr_node = f'reservoir_{node}' if node in reservoir_list else f'link_{node}'
            wd = self.catchment_wc.loc[pywr_node, "Total_WD_MGD"]
            cu = self.catchment_wc.loc[pywr_node, "Total_CU_WD_Ratio"]

            consumption_prediction = min(Yhat_lag_prediction,
                                         cu * min(Yhat_lag_minus1_prediction, wd))

            value = Yhat_lag_prediction - consumption_prediction

        # If no catchment water consumption, just return the prediction
        else:
            value = Yhat_lag_prediction

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
                pairs.add((node, lag))
        return pairs
