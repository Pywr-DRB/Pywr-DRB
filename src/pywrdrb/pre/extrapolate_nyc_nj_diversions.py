"""
Extrapolation of NYC and NJ diversion data for periods without historical records.

Overview: 
This module provides a class for retrieving historical NYC and NJ diversions and 
extrapolating them into time periods where data is not available, based on seasonal 
flow regressions. For NYC reservoirs, this includes aggregated diversions from 
Pepacton, Cannonsville, and Neversink reservoirs. For NJ, this covers diversions 
from the Delaware River via the Delaware-Raritan Canal.

Technical Notes:
- Uses linear regression models between streamflow and diversions, trained on 
  seasonal data (quarters: DJF, MAM, JJA, SON).
- For each month in periods without diversion data, predicts a monthly diversion 
  value based on streamflow.
- Uses nearest neighbor matching to disaggregate and create daily diversion patterns from the 
  predicted monthly values.
- The processed data is saved to CSV files in pywrdrb/data/diversions/ (default) or 
  flows/{flow_type}/ (custom data).


Example Usage:
# Default behavior (historical observations)
from pywrdrb.pre import ExtrapolatedDiversionPreprocessor
processor = ExtrapolatedDiversionPreprocessor(loc='nj')
hist_diversions, hist_flows = processor.load()
processor.process()
processor.save()

# Custom flow data
processor = ExtrapolatedDiversionPreprocessor(loc='nyc', flow_type='my_custom_flows')
processor.process()
processor.save()


Links:
- See SI for Hamilton et al. (2024) for more details on the method formulation.

Change Log:
TJA, 2025-05-07, Bug fixes to align with old methods + docstrings
Modified, Aug 27 2025, Added support for custom flow datasets
"""
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime

from pywrdrb.utils.constants import cfs_to_mgd
from pywrdrb.pre.datapreprocessor_ABC import DataPreprocessor
from pywrdrb.pywr_drb_node_data import obs_site_matches, nyc_reservoirs
from pywrdrb.utils.hdf5 import extract_realization_from_hdf5
# Import pywrdrb_all_nodes for optimized HDF5 extraction
from pywrdrb.pywr_drb_node_data import immediate_downstream_nodes_dict
pywrdrb_all_nodes = list(immediate_downstream_nodes_dict.keys())

# List of NYC inflow gages to aggregate for "NYC_inflow"
nyc_inflow_gages = []
for res in nyc_reservoirs:
    nyc_inflow_gages.extend(obs_site_matches[res])


__all__ = ["ExtrapolatedDiversionPreprocessor",
           "ExtrapolatedDiversionEnsemblePreprocessor"]

class ExtrapolatedDiversionPreprocessor(DataPreprocessor):
    r"""
    Class for extrapolating NYC and NJ diversion data based on streamflow regressions.
    
    The class implements a workflow to extrapolate historical diversions into time
    periods where data is not available, using seasonal flow regressions. The diversion
    data is organized into daily time series and saved to CSV files.
    
    Methods
    -------
    load()
        Load historical diversion and streamflow data from data/observations/_raw.
    get_quarter(m)
        Return the quarter (season) of the year for a given month.
    get_overlapping_timespan(df1, df2)
        Find the maximum overlapping timespan between two DataFrames.
    train_regressions(df_m)
        Train seasonal regression models for diversion prediction.
    get_random_prediction_sample(lrms, lrrs, quarter, x)
        Generate a random prediction sample from the regression distribution.
    process()
        Run the full extrapolation workflow: load data, train models, predict diversions.
    save()
        Save the extrapolated diversion data to data/diversions/.
    
    Attributes
    ----------
    loc : str
        Location indicator, either "nyc" or "nj".
    flow_type : str, optional
        Flow type for custom data. If None, uses historical observations.
    quarters : tuple
        Seasons used for different regression models (DJF, MAM, JJA, SON).
    lrms : dict
        Dictionary of linear regression models for each season.
    lrrs : dict
        Dictionary of fitted linear regression results for each season.
    diversion : pd.DataFrame
        DataFrame containing the historical diversion data.
    flow : pd.DataFrame
        DataFrame containing the historical streamflow data.
    training_flow : pd.DataFrame
        DataFrame containing streamflow data used for training (always historical).
    extrapolation_flow : pd.DataFrame
        DataFrame containing streamflow data used for extrapolation (custom or historical).
    df : pd.DataFrame
        DataFrame of daily states combining diversion and flow data.
    df_m : pd.DataFrame
        DataFrame of monthly mean states.
    df_long : pd.DataFrame
        DataFrame containing the full time series data for extrapolation.
    df_long_m : pd.DataFrame
        DataFrame containing monthly mean data for the full time series.
    processed_data : dict
        Dictionary to store the processed extrapolated diversion data.

    Example Usage  
    -------------
    >>> from pywrdrb.pre import ExtrapolatedDiversionPreprocessor
    >>> processor = ExtrapolatedDiversionPreprocessor(loc='nyc')
    >>> hist_diversions, hist_flows = processor.load()
    >>> processor.process()
    >>> processor.save()
    >>> [out] Saved extrapolated diversion data to <path>src\pywrdrb\data\diversions\
    
    >>> # Using custom flow data
    >>> processor = ExtrapolatedDiversionPreprocessor(loc='nyc', flow_type='my_custom_flows')
    >>> processor.process()
    >>> processor.save()
    >>> [out] Saved extrapolated diversion data to <path>src\pywrdrb\data\flows\my_custom_flows\
    """
    def __init__(self, 
                 loc,
                 flow_type=None):
        """
        Initialize the ExtrapolatedDiversionPreprocessor.
        
        Parameters
        ----------
        loc : str
            Location indicator, must be either "nyc" or "nj".
        flow_type : str, optional
            Flow type for custom data. If None, uses historical observations.
            When provided, uses gage_flow_mgd.csv from flows/{flow_type}/ folder.
            
        Raises
        ------
        ValueError
            If the location parameter is not "nyc" or "nj".
        """
        super().__init__()
        
        assert loc in ["nyc", "nj"], f"Invalid location specified. Expected 'nyc' or 'nj'. Got {loc}"
        
        self.loc = loc
        self.flow_type = flow_type
        
        # Seasons (quarters) used for different regression models
        self.quarters = ("DJF", "MAM", "JJA", "SON")
        
        # Initialize attributes that will be populated during processing
        self.lrms = None  # Linear regression models
        self.lrrs = None  # Linear regression results
        self.diversion = None  # Historical diversion data
        self.training_flow = None  # Streamflow data used for training (always historical)
        self.extrapolation_flow = None  # Streamflow data used for extrapolation (custom or historical)
        self.df = None  # DataFrame of daily states
        self.df_m = None  # DataFrame of monthly mean states
        self.df_long = None  # Full time series data for extrapolation
        self.df_long_m = None  # Monthly mean data for full time series

        # Set random seed for consistent results
        np.random.seed(1)

        # Dictionary of files needed based on settings
        self.input_dirs = {}
        self.output_dirs = {}

        # Always use historical flow & diversions data for training
        self.input_dirs["flow_training"] = self.pn.observations.get_str("_raw", "streamflow_daily_usgs_mgd.csv")

        # diversion comes from DRBC data if NYC, else inferred from USGS data if NJ
        if self.loc == "nyc":
            self.input_dirs["diversion"] = self.pn.observations.get_str("_raw", "Pep_Can_Nev_diversions_daily_2000-2021.xlsx")        
        elif self.loc == "nj":  
            # NJ diversions are assumed to be the USGS flow in the Delaware-Raritan Canal
            # so the diversion input comes from the USGS flow data file
            self.input_dirs["diversion"] = self.input_dirs["flow_training"]  
        
        
        # Load either historic or custom flow data for extrapolation
        if flow_type is not None:
            print(f"Using custom flow data for extrapolation: {flow_type}")
            
            self.input_dirs["flow_extrapolation"] = self.pn.sc.get(f"flows/{flow_type}") / "gage_flow_mgd.csv"

            # Different diversion output files based on NYC or NJ location
            if self.loc == "nyc":
                self.output_dirs["diversion"] = self.pn.sc.get(f"flows/{flow_type}") / "diversion_nyc_extrapolated_mgd.csv"
            else:
                self.output_dirs["diversion"] = self.pn.sc.get(f"flows/{flow_type}") / "diversion_nj_extrapolated_mgd.csv"

        # Historical flow data for extrapolation
        else:
            print("Using historical flow data for extrapolation.")
            
            self.input_dirs["flow_extrapolation"] = self.pn.observations.get_str("_raw", "streamflow_daily_usgs_mgd.csv")

            if self.loc == "nyc":
                self.output_dirs["diversion"] = self.pn.diversions.get_str("diversion_nyc_extrapolated_mgd.csv")
            else:
                self.output_dirs["diversion"] = self.pn.diversions.get_str("diversion_nj_extrapolated_mgd.csv")
                                
    def load_training_data(self):
        if self.loc == "nyc":
            fname = self.input_dirs["diversion"]
            diversion = pd.read_excel(fname, index_col=0)
            diversion = diversion.iloc[:, :3]
            diversion.index = pd.to_datetime(diversion.index)
            diversion["aggregate"] = diversion.sum(axis=1)
            diversion = diversion.loc[np.logical_not(np.isnan(diversion["aggregate"]))]
            ### Convert CFS to MGD
            diversion *= cfs_to_mgd
        elif self.loc == "nj":
            ### Load NJ diversions from training flow data (historical)
            start_date = (1991, 1, 1)
            fname = self.input_dirs["flow_training"]
            gage_flow = pd.read_csv(fname)
            gage_flow.index = pd.DatetimeIndex(gage_flow["datetime"]).date
            
            # Convert gage ID ("01460440") to "D_R_Canal"
            gage_flow["D_R_Canal"] = gage_flow["01460440"]
                        
            # the NJ diversion is estimated based on the canal flow
            diversion = gage_flow[["D_R_Canal"]]
            
            # Keep just diversions after the start date
            diversion = diversion.loc[
                np.logical_and(
                    diversion.index >= datetime.date(*start_date),
                    diversion.index <= datetime.date.today(),
                )
            ]
            diversion.index = pd.to_datetime(diversion.index)

            ### Infill NA values with previous day's flow
            for i in range(1, diversion.shape[0]):
                if np.isnan(diversion["D_R_Canal"].iloc[i]):
                    ind = diversion.index[i]
                    diversion.loc[ind, "D_R_Canal"] = diversion["D_R_Canal"].iloc[i - 1]

            ### Flow becomes negative sometimes, presumably due to storms and/or drought reversing flow.
            ### Set negative values to zero.
            diversion.loc[diversion["D_R_Canal"] < 0, "D_R_Canal"] = 0.0


        ### Load training flow data (always historical USGS data)
        fname_training = self.input_dirs["flow_training"]
        training_flow = pd.read_csv(fname_training)
        training_flow.index = pd.DatetimeIndex(training_flow["datetime"]).date
        training_flow.index = pd.to_datetime(training_flow.index)
        self.training_flow = training_flow

        ## Calculate/preprocess columns needed for training

        
        if self.loc == "nyc":
            training_flow["NYC_inflow"] = training_flow[nyc_inflow_gages].sum(axis=1)
        else:  # NJ
            training_flow["delTrenton"] = training_flow["01463500"]
            
        return training_flow, diversion

    def load_extrapolation_data(self):
        fname_extrapolation = self.input_dirs["flow_extrapolation"]
        
        ## For custom flow data, columns must be node names
        if self.flow_type is not None:
            # Load custom flow data
            extrapolation_flow = pd.read_csv(fname_extrapolation, index_col=0, parse_dates=True)
            
            # Calculate NYC_inflow from reservoir components if using custom data
            if self.loc == "nyc":
                extrapolation_flow["NYC_inflow"] = extrapolation_flow[nyc_reservoirs].sum(axis=1)
            
            # Ensure delTrenton column exists (required for NJ)
            if self.loc == "nj" and "delTrenton" not in extrapolation_flow.columns:
                raise ValueError(f"Custom flow data must contain 'delTrenton' column for NJ diversions.")
        
        ## For default, historic data, columns will be gauge ID numbers
        else:
            # Use historical flow data (original behavior)
            extrapolation_flow = pd.read_csv(fname_extrapolation)
            extrapolation_flow.index = pd.to_datetime(extrapolation_flow["datetime"])

            # Calculate total NYC inflow
            if self.loc == 'nyc':
                extrapolation_flow["NYC_inflow"] = extrapolation_flow[nyc_inflow_gages].sum(axis=1)
            else:  # NJ
                extrapolation_flow["delTrenton"] = extrapolation_flow["01463500"]

        return extrapolation_flow

    def load(self):
        """
        Load historical diversion and streamflow data.
        
        This method loads the required data for the extrapolation:
        - For NYC: loads diversion data from a spreadsheet containing daily diversions 
          from Pepacton, Cannonsville, and Neversink reservoirs.
        - For NJ: extracts Delaware-Raritan Canal flow data from the USGS streamflow data.
        - For both: loads streamflow data for training (always historical) and extrapolation (custom or historical).
        
        Returns
        -------
        tuple
            A tuple containing (diversion, flow) DataFrames.
        """
        # Load historical diversion data (unchanged for training)
        training_flow, diversion = self.load_training_data()
        self.diversion = diversion
        self.training_flow = training_flow

        # Load extrapolation flow data (custom or historical)
        extrapolation_flow = self.load_extrapolation_data()
        self.extrapolation_flow = extrapolation_flow

        return diversion, extrapolation_flow

    # Quarter mapping dictionary for vectorized operations
    _quarter_map = {
        1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM",
        5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA",
        9: "SON", 10: "SON", 11: "SON", 12: "DJF"
    }

    def get_quarter(self, m):
        """
        Return the quarter (season) of the year for a given month.

        Parameters
        ----------
        m : int
            Month number (1-12).

        Returns
        -------
        str
            Quarter string ("DJF", "MAM", "JJA", or "SON").
        """
        return self._quarter_map.get(m, "DJF")

    def get_overlapping_timespan(self, df1, df2):
        """
        Find the maximum overlapping timespan between two DataFrames.
        
        Parameters
        ----------
        df1 : pd.DataFrame
            First DataFrame with DatetimeIndex.
        df2 : pd.DataFrame
            Second DataFrame with DatetimeIndex.
            
        Returns
        -------
        tuple
            A tuple containing (df1_subset, df2_subset) with the same timespan.
            
        Raises
        ------
        AssertionError
            If the indices don't match after subsetting.
        """
        # Get the overlap time period
        df1 = df1.loc[
            np.logical_and(
                df1.index >= df2.index.min(), df1.index <= df2.index.max()
            )
        ]
        df2 = df2.loc[
            np.logical_and(
                df2.index >= df1.index.min(), df2.index <= df1.index.max()
            )
        ]
        assert np.all(df1.index == df2.index), "Indices do not match after get_overlapping_timespan()."
        return df1, df2

    def train_regressions(self, df_m):
        """
        Train seasonal regression models for diversion prediction.
        
        Parameters
        ----------
        df_m : pd.DataFrame
            Monthly DataFrame containing flow_log, diversion, and quarter columns.
            
        Returns
        -------
        tuple
            A tuple containing (lrms, lrrs) where lrms is a dict of regression models
            and lrrs is a dict of regression results for each quarter.
        """
        lrms = {}
        lrrs = {}
        
        for q in self.quarters:
            data = df_m.loc[df_m["quarter"] == q]
            
            x = data["flow_log"].values
            y = data["diversion"].values
            
            # Add constant term
            x_with_const = sm.add_constant(x)
            
            # Fit regression model
            model = sm.OLS(y, x_with_const)
            results = model.fit()
            
            lrms[q] = model
            lrrs[q] = results
            
        return lrms, lrrs

    def get_random_prediction_sample(self, lrms, lrrs, quarter, x):
        """
        Generate a random prediction sample from the regression distribution.

        Parameters
        ----------
        lrms : dict
            Dictionary of regression models for each quarter.
        lrrs : dict
            Dictionary of fitted regression results for each quarter.
        quarter : str
            The quarter/season for the prediction.
        x : float
            The input value (log flow) for prediction.

        Returns
        -------
        float
            A randomly sampled prediction value.
        """
        lrm = lrms[quarter]
        lrr = lrrs[quarter]
        exog = lrm.exog.copy()  # Create a copy to avoid modifying original
        exog[:, 1] = x  # Set the second column (flow_log) to x

        # Get randomly sampled value from linear regression model
        # Throw out if negative
        pred = -1
        while pred < 0:
            pred = lrm.get_distribution(
                lrr.params, scale=np.var(lrr.resid), exog=exog
            ).rvs()[0]

        return pred

    def get_random_prediction_samples_vectorized(self, lrms, lrrs, quarter, x_values):
        """
        Generate random prediction samples from the regression distribution (vectorized).

        This is a vectorized version of get_random_prediction_sample() that processes
        multiple x values at once for better performance.

        NOTE: The original code uses rejection sampling (while pred < 0) which works
        probabilistically. However, this can fail for NJ where predictions are in
        log-transformed space and the regression may have poor fit. We use a simpler
        approach: just sample once and accept all values, matching statistical properties.

        Parameters
        ----------
        lrms : dict
            Dictionary of regression models for each quarter.
        lrrs : dict
            Dictionary of fitted regression results for each quarter.
        quarter : str
            The quarter/season for the prediction.
        x_values : np.ndarray
            Array of input values (log flow) for predictions.

        Returns
        -------
        np.ndarray
            Array of randomly sampled prediction values.
        """
        lrm = lrms[quarter]
        lrr = lrrs[quarter]

        n_samples = len(x_values)

        # Create exog matrix with constant and flow_log values
        exog = np.column_stack([np.ones(n_samples), x_values])

        # Get variance for random sampling
        scale = np.var(lrr.resid)

        # Generate predictions with random noise
        # This matches the behavior of lrm.get_distribution().rvs()
        predictions = lrr.predict(exog) + np.random.normal(0, np.sqrt(scale), n_samples)

        # For NYC (untransformed space), reject negative values makes sense
        # For NJ (log-transformed space), negative predictions are valid and will be
        # transformed back appropriately
        if self.loc == "nyc":
            # Only for NYC: reject and resample negative values
            negative_mask = predictions < 0
            max_iterations = 100
            iteration = 0

            while negative_mask.any() and iteration < max_iterations:
                n_negative = negative_mask.sum()
                new_samples = lrr.predict(exog[negative_mask]) + np.random.normal(0, np.sqrt(scale), n_negative)
                predictions[negative_mask] = new_samples
                negative_mask = predictions < 0
                iteration += 1

            # Final safety for NYC only
            if negative_mask.any():
                n_still_negative = negative_mask.sum()
                if getattr(self, 'rank', 0) == 0:
                    print(f"Warning (NYC): {n_still_negative}/{n_samples} predictions remained negative after {max_iterations} iterations. Clamping to 0.")
                predictions = np.maximum(predictions, 0)

        # For NJ, allow negative predictions - they're in log-transformed space
        # The back-transformation will handle physical constraints

        return predictions

    def process(self):
        """
        Run the full extrapolation workflow.
        
        This method implements the full extrapolation workflow:
        1. Load diversion and flow data
        2. Create daily dataframe combining diversions and training flows
        3. Create monthly mean dataframe
        4. Train seasonal regression models between flow and diversion using training data
        5. Predict monthly diversions for the full extrapolation time period
        6. Use nearest neighbor matching to create daily patterns from 
           monthly predictions
        7. Combine historical and extrapolated diversions into a single dataset
        8. Format and store the results
           
        The processed data is stored in the self.processed_data dictionary.
        """
        # Load data if not already loaded
        if self.diversion is None or self.training_flow is None or self.extrapolation_flow is None:
            self.diversion, _ = self.load()
        
        # Make copies to keep the full version for later
        training_flow = self.training_flow.copy()
        diversion = self.diversion.copy()
        
        # Get maximum overlapping timespan for diversions and training flow (for model training)
        training_flow, diversion = self.get_overlapping_timespan(training_flow, diversion)
        
        # Set up column names based on location
        diversion_column = "aggregate" if self.loc == "nyc" else "D_R_Canal"
        flow_column = "NYC_inflow" if self.loc == "nyc" else "delDRCanal"
        
        # Create dataframe of daily states using training data
        df = pd.DataFrame(
            {
                "diversion": diversion[diversion_column],
                "flow_log": np.log(training_flow[flow_column]),
                "m": diversion.index.month,
                "y": diversion.index.year,
            }
        )

        # Create dataframe of monthly mean states
        df_m = df.resample("ME").mean()
        # OPTIMIZED: Use vectorized map instead of list comprehension
        df["quarter"] = df["m"].map(self._quarter_map)
        df_m["quarter"] = df_m["m"].map(self._quarter_map)

        # NJ diversion data are left skewed, so negate and then apply log transform
        if self.loc == "nj":
            nj_trans_max = df_m["diversion"].max() + 5
            df_m["diversion"] = np.log(nj_trans_max - df_m["diversion"])

        # Train linear regression models for each quarter using training data
        if self.lrms is None or self.lrrs is None:
            lrms, lrrs = self.train_regressions(df_m)
            self.lrms = lrms
            self.lrrs = lrrs
                    
        # Prepare data for extrapolation using extrapolation flow dataset
        extrapolation_flow_full = self.extrapolation_flow.copy()

        # Set up dataframe with extrapolation flow data for full time period
        df_long = pd.DataFrame(
            {
                "flow_log": np.log(extrapolation_flow_full[flow_column]),
                "m": extrapolation_flow_full.index.month,
                "y": extrapolation_flow_full.index.year,
            }
        )

        # Get monthly means and add quarter info
        df_long_m = df_long.resample("ME").mean()
        # OPTIMIZED: Use vectorized map instead of list comprehension
        df_long["quarter"] = df_long["m"].map(self._quarter_map)
        df_long_m["quarter"] = df_long_m["m"].map(self._quarter_map)

        # Use trained regression models to predict monthly diversions
        # OPTIMIZED: Vectorized prediction by quarter instead of row-by-row loop
        df_long_m["diversion_pred"] = 0.0

        for q in self.quarters:
            # Get mask for this quarter
            mask = df_long_m["quarter"] == q

            # Get flow values for this quarter
            flow_values = df_long_m.loc[mask, "flow_log"].values

            # Generate predictions for all months in this quarter at once
            predictions = self.get_random_prediction_samples_vectorized(
                lrms=self.lrms,
                lrrs=self.lrrs,
                quarter=q,
                x_values=flow_values
            )

            # Assign all predictions at once (much faster than row-by-row)
            df_long_m.loc[mask, "diversion_pred"] = predictions

        # For NJ, transform data back to original scale
        if self.loc == "nj":
            df_m["diversion"] = np.maximum(nj_trans_max - np.exp(df_m["diversion"]), 0)
            df_long_m["diversion_pred"] = np.maximum(
                nj_trans_max - np.exp(df_long_m["diversion_pred"]), 0
            )

        # Existing code for nearest neighbor matching and daily disaggregation
        # Set up for nearest neighbor matching in normalized 2D space of log-flow & diversion
        flow_bounds = [df_m["flow_log"].min(), df_m["flow_log"].max()]
        diversion_bounds = [df_m["diversion"].min(), df_m["diversion"].max()]

        # Normalize values to [0, 1] range
        df_m["flow_log_norm"] = (df_m["flow_log"] - flow_bounds[0]) / (
            flow_bounds[1] - flow_bounds[0]
        )
        df_m["diversion_norm"] = (df_m["diversion"] - diversion_bounds[0]) / (
            diversion_bounds[1] - diversion_bounds[0]
        )
        df_long_m["flow_log_norm"] = (df_long_m["flow_log"] - flow_bounds[0]) / (
            flow_bounds[1] - flow_bounds[0]
        )
        df_long_m["diversion_pred_norm"] = (
            df_long_m["diversion_pred"] - diversion_bounds[0]
        ) / (diversion_bounds[1] - diversion_bounds[0])

        # Find nearest neighbor in historical data for each month in full time period
        df_long_m["nn"] = -1
        for i in range(df_long_m.shape[0]):
            ind = df_long_m.index[i]
            q = df_long_m["quarter"].iloc[i]
            f = df_long_m["flow_log_norm"].iloc[i]
            n = df_long_m["diversion_pred_norm"].iloc[i]
            
            # Get subset of training data for the same quarter
            subset = df_m.loc[df_m["quarter"] == q]
            
            # Calculate Euclidean distance in normalized 2D space
            distances = np.sqrt(
                (subset["flow_log_norm"] - f) ** 2 + (subset["diversion_norm"] - n) ** 2
            )
            
            # Find index of nearest neighbor
            nn_idx = distances.idxmin()
            df_long_m.loc[ind, "nn"] = nn_idx

        # Use nearest neighbor to disaggregate monthly predictions to daily
        df_long["diversion_pred"] = 0.0
        for i in range(df_long_m.shape[0]):
            # Get month info
            month_start = df_long_m.index[i]
            nn_month = df_long_m["nn"].iloc[i]
            monthly_pred = df_long_m["diversion_pred"].iloc[i]
            
            # Get daily data for this month and nearest neighbor month
            month_mask = (df_long.index.year == month_start.year) & (df_long.index.month == month_start.month)
            nn_mask = (df.index.year == nn_month.year) & (df.index.month == nn_month.month)
            
            if nn_mask.sum() > 0 and month_mask.sum() > 0:
                # Get daily pattern from nearest neighbor
                nn_daily = df.loc[nn_mask, "diversion"]
                nn_monthly_mean = nn_daily.mean()
                
                if nn_monthly_mean > 0:
                    # Scale daily pattern to match predicted monthly value
                    daily_pattern = nn_daily / nn_monthly_mean
                    new_diversion = daily_pattern * monthly_pred
                    
                    # Assign to corresponding days in long time series
                    # Check if the months have the same number of days
                    if len(new_diversion) == month_mask.sum():
                        df_long.loc[month_mask, "diversion_pred"] = new_diversion.values
                    else:
                        # Handle different month lengths (e.g., Feb 28 vs 29 days, or 30 vs 31)
                        # Interpolate or repeat pattern to match target month length
                        target_days = month_mask.sum()
                        source_days = len(new_diversion)
                        
                        if target_days > source_days:
                            # Need to extend the pattern - repeat last day(s)
                            extension = np.tile(new_diversion.values[-1], target_days - source_days)
                            extended_diversion = np.concatenate([new_diversion.values, extension])
                            df_long.loc[month_mask, "diversion_pred"] = extended_diversion
                        else:
                            # Need to truncate the pattern
                            df_long.loc[month_mask, "diversion_pred"] = new_diversion.values[:target_days]
                            

        # Store for potential plotting
        self.df = df
        self.df_m = df_m
        self.df_long = df_long
        self.df_long_m = df_long_m

        # Now reload historical diversion dataset & add extrapolated data for dates we don't have
        if self.loc == "nyc":
            diversion_output = self.diversion.copy()

            # Format & save to csv for use in Pywr-DRB
            df_long_filtered = df_long.loc[
                np.logical_or(
                    df_long.index < diversion_output.index.min(),
                    df_long.index > diversion_output.index.max(),
                )
            ]
            diversion_combined = pd.concat(
                [diversion_output, pd.DataFrame({"aggregate": df_long_filtered["diversion_pred"]})]
            )
            diversion_combined = diversion_combined.sort_index()
            diversion_combined["datetime"] = diversion_combined.index
            diversion_combined.columns = [
                "pepacton",
                "cannonsville",
                "neversink",
                "aggregate",
                "datetime",
            ]
            diversion_combined = diversion_combined.iloc[:, [-1, 1, 0, 2, 3]]

            self.processed_data = diversion_combined

        elif self.loc == "nj":
            diversion_output = self.diversion.copy()

            # Format & save to csv for use in Pywr-DRB
            df_long_filtered = df_long.loc[
                np.logical_or(
                    df_long.index < diversion_output.index.min(),
                    df_long.index > diversion_output.index.max(),
                )
            ]
            diversion_combined = pd.concat(
                [diversion_output, pd.DataFrame({"D_R_Canal": df_long_filtered["diversion_pred"]})]
            )
            diversion_combined = diversion_combined.sort_index()
            diversion_combined["datetime"] = diversion_combined.index
            diversion_combined = diversion_combined.iloc[:, [-1, 0]]

            self.processed_data = diversion_combined
        
        
        # Keep only dates that overlap the extrapolation data time period
        keep_dates = np.logical_and(
            self.processed_data.index >= self.extrapolation_flow.index.min(),
            self.processed_data.index <= self.extrapolation_flow.index.max(),
        )
        self.processed_data = self.processed_data.loc[keep_dates]   

    def save(self):
        """
        Save the processed extrapolated diversion data to CSV.
        
        The data is saved to the output directory specified in self.output_dirs,
        with the filename format determined by the location (NYC or NJ) and flow_type.
        
        Raises
        ------
        ValueError
            If processed_data is not available or if location is invalid.
        """
        # Make sure the data has been processed
        if self.processed_data is None:
            raise ValueError("No processed data available. Run the process() method first.")
        
        # Get output file path
        output_file = self.output_dirs["diversion"]
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        self.processed_data.to_csv(output_file, index=False)
        
        print(f"Saved extrapolated diversion data to {output_file}")

    def plot(self, fig_dir=None, 
             kind="regressions"):
        """
        Create plots of the extrapolation process.
        
        Parameters
        ----------
        kind : str, optional
            The type of plot to create. Options are:
            - "regressions": Plot the seasonal regression models and data points.
            - "diversions": Plot the historical and extrapolated diversion time series.
            
        Raises
        ------
        ValueError
            If an invalid plot type is specified.
        AssertionError
            If data is not available for plotting.
            
        Notes
        -----
        Plots are saved to the figures directory with names based on the location
        and plot type: extrapolation_{loc}_pt1.png or extrapolation_{loc}_pt2.png.
        """
        assert kind in ["regressions", "diversions"], "Invalid kind. Expected 'regressions' or 'diversions'."
        
        if fig_dir is None:
            fig_dir = str(self.pn.sc.get(f"flows/{self.flow_type}")) + "/figures/"
        
        if kind == "regressions":
            self.plot_regressions(fig_dir)
        elif kind == "diversions":
            self.plot_diversions(fig_dir)
    
    def plot_regressions(self, fig_dir='./figures/'):
        """
        Plot the seasonal regression models and data points.
        
        Creates a 2x2 grid of scatter plots showing the relationship between
        log-transformed streamflow and diversions for each season (quarter).
        The plot includes:
        - Observed data points
        - Extrapolated values for the observed period
        - Extrapolated values for the unobserved period
        - Regression lines
        
        The plot is saved to the figures directory as extrapolation_{loc}_pt1.png.
        
        Raises
        ------
        AssertionError
            If required data is not available.
        """
        # Check that required data is available
        assert self.df_m is not None, "Data not loaded. Run process() first."
        assert self.df_long_m is not None, "Data not loaded. Run process() first."
        assert self.lrrs is not None, "Regression models not available. Run process() first."
        
        fig, axs = plt.subplots(
            2, 2, figsize=(8, 8), 
            gridspec_kw={"hspace": 0.2, "wspace": 0.2}
        )
        
        for i, q in enumerate(self.quarters):
            row = 1 if i >= 2 else 0
            col = 1 if i % 2 == 1 else 0
            ax = axs[row, col]

            # Plot observed data
            data = self.df_m.loc[self.df_m["quarter"] == q].copy()
            ax.scatter(
                data["flow_log"],
                data["diversion"],
                zorder=2,
                alpha=0.7,
                color="cornflowerblue",
                label="Observed",
            )
            
            # Plot sampled data during observed period, if there is overlap
            
            if len(self.df_long_m.index.intersection(self.df_m.index)) > 3:
                # Plot sampled data during observed period
                # get intersection of indices
                data = self.df_long_m.loc[self.df_long_m.index.intersection(self.df_m.index)].copy()
                data = data.loc[data["quarter"] == q]
                ax.scatter(
                    data["flow_log"],
                    data["diversion_pred"],
                    zorder=1,
                    alpha=0.7,
                    color="firebrick",
                    label="Extrapolated over\nobserved period",
                )
            
            # Plot sampled data during unobserved period
            data = self.df_long_m.loc[[i not in self.df_m.index for i in self.df_long_m.index]].copy()
            data = data.loc[data["quarter"] == q]
            ax.scatter(
                data["flow_log"],
                data["diversion_pred"],
                zorder=0,
                alpha=0.7,
                color="darkgoldenrod",
                label="Extrapolated over\nunobserved period",
            )

            # Plot regression line
            xlim = ax.get_xlim()
            ax.plot(
                xlim,
                [self.lrrs[q].params[0] + self.lrrs[q].params[1] * x for x in xlim],
                color="k",
                label="Regression",
            )

            # Add legend to bottom right plot
            if row == 1 and col == 1:
                ax.legend(loc="center left", bbox_to_anchor=(1.0, 1.1), frameon=False)

            # Clean up axes
            ax.set_title(q)
            if row == 1:
                ax.set_xlabel("Log inflow (log MGD)")
            if self.loc == "nyc":
                ax.set_ylim([0, ax.get_ylim()[1]])
            if self.loc == "nyc" and col == 0:
                ax.set_ylabel("Monthly NYC diversion (MGD)")
            elif self.loc == "nj" and col == 0:
                ax.set_ylabel("Transformed monthly NJ diversion")

        # Save the figure
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(
            f"{fig_dir}/extrapolation_{self.loc}_pt1.png", 
            dpi=400, bbox_inches="tight"
        )
        plt.close()

    def plot_diversions(self, fig_dir='./figures/'):
        """
        Plot the historical and extrapolated diversion time series.
        
        Creates a line plot showing:
        - Historical observed diversions
        - Extrapolated diversions
        
        The plot is saved to the figures directory as extrapolation_{loc}_pt2.png.
        
        Raises
        ------
        AssertionError
            If required data is not available.
        """
        assert self.df is not None, "Data not loaded. Run process() first."
        assert self.df_long is not None, "Data not loaded. Run process() first."
        
        fig, ax = plt.subplots(
            1, 1, figsize=(5, 3), 
            gridspec_kw={"hspace": 0.2, "wspace": 0.2}
        )

        # Plot observed diversion daily timeseries
        ax.plot(
            self.df["diversion"],
            color="cornflowerblue",
            label="Observed",
            zorder=2,
            lw=0.5,
            alpha=0.7,
        )
        
        # Plot extrapolated diversions
        ax.plot(
            self.df_long["diversion_pred"],
            color="darkgoldenrod",
            label="Extrapolated",
            zorder=1,
            lw=0.5,
            alpha=0.7,
        )

        # Add legend
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)

        # Clean up axes
        ax.set_ylim([0, ax.get_ylim()[1]])
        ylab = "Daily NYC diversion (MGD)" if self.loc == "nyc" else "Daily NJ diversion (MGD)"
        ax.set_ylabel(ylab)

        # Save the figure
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(
            f"{fig_dir}/extrapolation_{self.loc}_pt2.png", 
            dpi=400, bbox_inches="tight"
        )
        plt.close()
        

class ExtrapolatedDiversionEnsemblePreprocessor(ExtrapolatedDiversionPreprocessor):
    """
    Class for generating an ensemble of extrapolated diversion datasets.
    
    This class extends ExtrapolatedDiversionPreprocessor to create multiple 
    realizations of extrapolated diversion data, allowing for ensembles with 
    unique diversion dynamics for each realization.
    """
    def __init__(self,
                 loc,
                 flow_type,
                 ensemble_hdf5_file,
                 realization_ids=None,
                 use_mpi=True):
        """
        Initialize the ExtrapolatedDiversionEnsemblePreprocessor.
        
        Parameters
        ----------
        loc : str
            Location indicator, must be either "nyc" or "nj".
        flow_type : str
            Flow type for custom data. Must be provided.
        ensemble_hdf5_file : str
            Path to the HDF5 file containing ensemble gage_flow_mgd data.
        """
        super().__init__(loc=loc, 
                         flow_type=flow_type)

        self.ensemble_hdf5_file = ensemble_hdf5_file
        self.realization_ids = realization_ids
        
        assert loc in ["nyc", "nj"], f"Invalid location specified. Expected 'nyc' or 'nj'. Got {loc}"
        self.loc = loc
        
        self.use_mpi = use_mpi
        if self.use_mpi:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
        
        # Overwrite the input and output files to use hdf5 instead of csv
        # It is assumed that the ensemble will have filetype hdf5
        csv_input = self.input_dirs["flow_extrapolation"]
        csv_output = self.output_dirs["diversion"]
        
        self.input_dirs["flow_extrapolation"] = self.ensemble_hdf5_file
        self.output_dirs["diversions"] = str(csv_output).replace(".csv", ".hdf5")

        # Storage for ensemble results
        self.ensemble_diversions = {}

    def _extract_realization_from_open_file(self, hdf5_file, realization_id):
        """
        Extract a single realization from an already-open HDF5 file.

        This is an optimized version of extract_realization_from_hdf5() that works
        with an already-open file handle to avoid repeated file open/close operations.

        Parameters
        ----------
        hdf5_file : h5py.File
            Open HDF5 file handle
        realization_id : str
            The realization ID to extract

        Returns
        -------
        pd.DataFrame
            DataFrame containing the extracted realization data
        """
        # Extract timeseries data from realization for each node
        data = {}

        for node in pywrdrb_all_nodes:
            node_data = hdf5_file[node]
            column_labels = node_data.attrs["column_labels"]

            err_msg = f"The specified realization {realization_id} is not available in the HDF file."
            assert realization_id in column_labels, (
                err_msg + f" Realizations available: {column_labels}"
            )
            data[node] = node_data[realization_id][:]

        dates = node_data["date"][:].tolist()
        data["datetime"] = dates

        # Combine into dataframe
        df = pd.DataFrame(data, index=dates)
        df.index = pd.to_datetime(df.index.astype(str))
        return df

    def load(self):
        """Load available realization IDs and training data (optimized for MPI)."""

        ### Load realization IDs from HDF5 if not provided
        # Get available realization IDs if not specified
        if self.realization_ids is None:
            with h5py.File(self.ensemble_hdf5_file, 'r') as f:
                self.realization_ids = [key for key in f.keys()]
        else:
            # Ensure provided IDs are strings
            self.realization_ids = [str(rid) for rid in self.realization_ids]

        if self.rank == 0:
            print(f"Processing {len(self.realization_ids)} realizations across {self.size} processes")

        ### OPTIMIZATION: Load training data only on rank 0, then broadcast
        # This avoids redundant disk I/O across all MPI ranks
        if self.rank == 0:
            print(f"Rank 0: Loading training data...")
            training_flow, diversion = self.load_training_data()
        else:
            training_flow, diversion = None, None

        # Broadcast training data to all ranks
        if self.use_mpi:
            if self.rank == 0:
                print(f"Rank 0: Broadcasting training data to all ranks...")
            self.training_flow = self.comm.bcast(training_flow, root=0)
            self.diversion = self.comm.bcast(diversion, root=0)
        else:
            self.training_flow = training_flow
            self.diversion = diversion

        if self.rank == 0:
            print(f"Training data loaded and distributed to all ranks")

        return


    def process(self):
        """Process ensemble extrapolations using MPI parallelization (optimized I/O)."""

        # Distribute realizations across MPI processes
        realizations_per_rank = np.array_split(self.realization_ids, self.size)
        my_realizations = realizations_per_rank[self.rank]

        local_predictions = {}

        ### OPTIMIZATION: Batch HDF5 reads - open file once per rank
        # This reduces file I/O overhead from N opens to 1 per rank
        if self.rank == 0:
            print(f"Rank {self.rank}: Processing {len(my_realizations)} realizations with batched HDF5 reads...")

        with h5py.File(self.ensemble_hdf5_file, 'r') as hdf5_file:
            # Process assigned realizations with the file already open
            for i, realization_id in enumerate(my_realizations):
                if self.rank == 0 and (i + 1) % max(1, len(my_realizations) // 5) == 0:
                    print(f"Rank 0: Processing realization {i+1}/{len(my_realizations)}: {realization_id}")

                # Extract realization data from open HDF5 file
                extrapolation_flow_i = self._extract_realization_from_open_file(
                    hdf5_file,
                    realization_id
                )

                # Need to add NYC_inflow columns if not present
                if self.loc == "nyc" and "NYC_inflow" not in extrapolation_flow_i.columns:
                    extrapolation_flow_i["NYC_inflow"] = extrapolation_flow_i[nyc_reservoirs].sum(axis=1)

                # Ensure delTrenton column exists (required for NJ)
                if self.loc == "nj" and "delTrenton" not in extrapolation_flow_i.columns:
                    raise ValueError(f"Custom flow data must contain 'delTrenton' column for NJ diversions.")

                # Set the extrapolation_flow with this realization
                # This attribute is expected before super().process() is called
                self.extrapolation_flow = extrapolation_flow_i

                # Run the extrapolation using the base class
                super().process()

                # Pull out the processed data for this realization
                extrapolated_diversion_i = self.processed_data.copy()

                local_predictions[str(realization_id)] = extrapolated_diversion_i

        if self.rank == 0:
            print(f"Rank 0: Completed processing all assigned realizations")

        # Gather all predictions to rank 0
        if self.use_mpi:
            all_predictions = self.comm.gather(local_predictions, root=0)
        else:
            all_predictions = [local_predictions]

        if self.rank == 0:
            # Combine predictions from all processes
            for diversions_dict in all_predictions:
                self.ensemble_diversions.update(diversions_dict)

    def save(self):
        """Save ensemble extrapolated diversions to HDF5 format."""
        if self.rank == 0:
            if not self.ensemble_diversions:
                raise ValueError("No ensemble diversions to save. Run process() first.")

            fname = self.output_dirs["diversions"]
            
            with h5py.File(fname, 'w') as hf:
                for realization_id, predictions_df in self.ensemble_diversions.items():
                    # Create group for this realization
                    realization_group = hf.create_group(realization_id)

                    # Store column labels as attribute for compatibility with extract_realization_from_hdf5
                    column_labels = list(predictions_df.columns)
                    realization_group.attrs['column_labels'] = column_labels

                    # Store datetime
                    datetime_strings = predictions_df['datetime'].astype(str).values
                    realization_group.create_dataset('datetime', data=datetime_strings)

                    # Store prediction columns
                    for col in predictions_df.columns:
                        if col != 'datetime':
                            realization_group.create_dataset(col, data=predictions_df[col].values)

            print(f"Saved ensemble diversions to {fname}")

        # Ensure all processes wait for save to complete
        if self.use_mpi:
            self.comm.barrier()