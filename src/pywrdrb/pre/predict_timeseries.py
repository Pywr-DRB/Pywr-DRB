import pandas as pd
import numpy as np
import statsmodels.api as sm

from pywrdrb.utils.timeseries import subset_timeseries
from . import DataPreprocessor

class PredictedTimeseriesPreprocessor(DataPreprocessor):
    def __init__(self, 
                 flow_type, 
                 start_date=None, 
                 end_date=None,
                 use_log=True, 
                 remove_zeros=False, 
                 use_const=False):
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

    def load(self):
        pass

    def save(self):
        pass

    def process(self):
        pass

    def get_prediction_node_lag_combinations(self):
        return None

    def train_regressions(self):
        
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
        Y = df[node].iloc[lag:].values.astype(float)
        X = df[node].iloc[:-lag].values if lag > 0 else df[node].values

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
        pairs = set()
        for combos in self.get_prediction_node_lag_combinations().values():
            for (node, lag), _ in combos:
                if lag >= 0:
                    pairs.add((node, lag))
        return pairs
