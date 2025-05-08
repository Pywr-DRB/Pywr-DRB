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
- The processed data is saved to CSV files in pywrdrb/data/diversions/.

Links:
- See SI for Hamilton et al. (2024) for more details on the method formulation.

Change Log:
TJA, 2025-05-07, Bug fixes to align with old methods + docstrings
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime

from pywrdrb.utils.constants import cfs_to_mgd, cms_to_mgd
from pywrdrb.pre.datapreprocessor_ABC import DataPreprocessor

__all__ = ["ExtrapolatedDiversionPreprocessor"]

class ExtrapolatedDiversionPreprocessor(DataPreprocessor):
    """
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
    start_date : str
        Start date for the extrapolation period.
    end_date : str
        End date for the extrapolation period.
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
    >>> processor = ExtrapolatedDiversionPreprocessor(loc='nyc', start_date='1945-01-01', end_date='2024-12-31')
    >>> hist_diversions, hist_flows = processor.load()
    >>> processor.process()
    >>> processor.save()
    >>> [out] Saved extrapolated diversion data to <path>src\pywrdrb\data\diversions\
    """
    def __init__(self, 
                 loc,
                 start_date=None,
                 end_date=None):
        """
        Initialize the ExtrapolatedDiversionPreprocessor.
        
        Parameters
        ----------
        loc : str
            Location indicator, must be either "nyc" or "nj".
        start_date : str, optional
            Start date for the extrapolation, format: "YYYY-MM-DD".
        end_date : str, optional
            End date for the extrapolation, format: "YYYY-MM-DD".
            
        Raises
        ------
        ValueError
            If the location parameter is not "nyc" or "nj".
        """
        super().__init__()
        
        assert loc in ["nyc", "nj"], f"Invalid location specified. Expected 'nyc' or 'nj'. Got {loc}"
        
        self.loc = loc
        self.start_date = start_date
        self.end_date = end_date
        
        # Seasons (quarters) used for different regression models
        self.quarters = ("DJF", "MAM", "JJA", "SON")
        
        # Initialize attributes that will be populated during processing
        self.lrms = None  # Linear regression models
        self.lrrs = None  # Linear regression results
        self.diversion = None  # Historical diversion data
        self.flow = None  # Historical streamflow data
        self.df = None  # DataFrame of daily states
        self.df_m = None  # DataFrame of monthly mean states
        self.df_long = None  # Full time series data for extrapolation
        self.df_long_m = None  # Monthly mean data for full time series

        # Set random seed for consistent results
        np.random.seed(1)
        
        # Setup input and output directories based on location
        if self.loc == "nyc":
            self.input_dirs = {
                "diversion": self.pn.observations.get_str("_raw", "Pep_Can_Nev_diversions_daily_2000-2021.xlsx"),
                "flow": self.pn.observations.get_str("_raw", "streamflow_daily_usgs_1950_2022_cms_for_NYC_NJ_diversions.csv")
            }
            self.output_dirs = {
                "diversion": self.pn.diversions.get_str("diversion_nyc_extrapolated_mgd.csv")
            }
        elif self.loc == "nj":
            self.input_dirs = {
                "flow": self.pn.observations.get_str("_raw", "streamflow_daily_usgs_1950_2022_cms_for_NYC_NJ_diversions.csv")
            }
            self.output_dirs = {
                "diversion": self.pn.diversions.get_str("diversion_nj_extrapolated_mgd.csv")
            }

    def load(self):
        """
        Load historical diversion and streamflow data.
        
        This method loads the required data for the extrapolation:
        - For NYC: loads diversion data from a spreadsheet containing daily diversions 
          from Pepacton, Cannonsville, and Neversink reservoirs.
        - For NJ: extracts Delaware-Raritan Canal flow data from the USGS streamflow data.
        - For both: loads streamflow data for relevant locations.
        
        Returns
        -------
        tuple
            A tuple containing (diversion, flow) DataFrames.
        """
        ### Get historical diversion data
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
            ### Now get NJ demands/deliveries
            ### The gage for D_R_Canal starts 1989-10-23, but lots of NA's early on. 
            ### Pretty good after 1991-01-01, but a few remaining to clean up.
            start_date = (1991, 1, 1)
            fname = self.input_dirs["flow"]
            diversion = pd.read_csv(fname)
            diversion.index = pd.DatetimeIndex(diversion["datetime"])
            diversion = diversion[["D_R_Canal"]]
            diversion = diversion.loc[diversion.index >= datetime.datetime(*start_date)]

            ### Infill NA values with previous day's flow
            for i in range(1, diversion.shape[0]):
                if np.isnan(diversion["D_R_Canal"].iloc[i]):
                    ind = diversion.index[i]
                    diversion.loc[ind, "D_R_Canal"] = diversion["D_R_Canal"].iloc[i - 1]

            ### Convert cms to mgd
            diversion *= cms_to_mgd
            ### Flow becomes negative sometimes, presumably due to storms and/or drought reversing flow. 
            ### Don't count as deliveries
            diversion[diversion < 0] = 0

        ### Get historical flows
        fname = self.input_dirs["flow"]
        flow = pd.read_csv(fname)
        flow.index = pd.to_datetime(flow["datetime"])
        
        # Store the data for later processing
        self.diversion = diversion
        self.flow = flow
        
        return diversion, flow
    
    def get_quarter(self, m):
        """
        Return the quarter (season) of the year for a given month.
        
        Parameters
        ----------
        m : int
            The month (1-12).
        
        Returns
        -------
        str
            The quarter/season abbreviation: "DJF", "MAM", "JJA", or "SON".
        """
        if m in (12, 1, 2):
            return "DJF"
        elif m in (3, 4, 5):
            return "MAM"
        elif m in (6, 7, 8):
            return "JJA"
        elif m in (9, 10, 11):
            return "SON"
    
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
        
        For each season (quarter), trains a linear regression model relating 
        log-transformed streamflow to diversion amounts.
        
        Parameters
        ----------
        df_m : pd.DataFrame
            DataFrame containing monthly mean values with 'diversion', 'flow_log',
            'm', 'y', and 'quarter' columns.
            
        Returns
        -------
        tuple
            A tuple containing (lrms, lrrs) where:
            - lrms is a dictionary of regression models for each quarter
            - lrrs is a dictionary of fitted regression results for each quarter
        """
        # Check for required columns
        required_cols = ['diversion', 'flow_log', 'm', 'y', 'quarter']
        for col in required_cols:
            assert col in df_m.columns, f"Required column '{col}' not found in df_m"

        lrms = {}
        lrrs = {}
        
        # Train a separate regression model for each season (quarter)
        for q in self.quarters:
            # Create OLS regression model
            lrms[q] = sm.OLS(
                df_m["diversion"].loc[df_m["quarter"] == q],
                sm.add_constant(df_m["flow_log"].loc[df_m["quarter"] == q]),
            )
            # Fit the model
            lrrs[q] = lrms[q].fit()
            
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
    
    def process(self):
        """
        Process the loaded data to extrapolate diversions.
        
        This method implements the full extrapolation workflow:
        1. Load diversion and flow data
        2. Create daily dataframe combining diversions and flows
        3. Create monthly mean dataframe
        4. Train seasonal regression models between flow and diversion
        5. Predict monthly diversions for the full time period
        6. Use nearest neighbor matching to create daily patterns from 
           monthly predictions
        7. Combine historical and extrapolated diversions into a single dataset
        8. Format and store the results
           
        The processed data is stored in the self.processed_data dictionary.
        """
        # Load data if not already loaded
        if self.diversion is None or self.flow is None:
            self.diversion, self.flow = self.load()
        
        # Make copies to keep the full version for later
        flow = self.flow.copy()
        diversion = self.diversion.copy()
        
        # Get maximum overlapping timespan for diversions and flow
        flow, diversion = self.get_overlapping_timespan(flow, diversion)
        
        # Set up column names based on location
        diversion_column = "aggregate" if self.loc == "nyc" else "D_R_Canal"
        flow_column = "NYC_inflow" if self.loc == "nyc" else "delTrenton"
        
        # Create dataframe of daily states
        df = pd.DataFrame(
            {
                "diversion": diversion[diversion_column],
                "flow_log": np.log(flow[flow_column]),
                "m": diversion.index.month,
                "y": diversion.index.year,
            }
        )

        # Create dataframe of monthly mean states
        df_m = df.resample("m").mean()
        df["quarter"] = [self.get_quarter(m) for m in df["m"]]
        df_m["quarter"] = [self.get_quarter(m) for m in df_m["m"]]

        # NJ diversion data are left skewed, so negate and then apply log transform
        if self.loc == "nj":
            nj_trans_max = df_m["diversion"].max() + 5
            df_m["diversion"] = np.log(nj_trans_max - df_m["diversion"])

        # Train linear regression models for each quarter
        lrms, lrrs = self.train_regressions(df_m)
        self.lrms = lrms
        self.lrrs = lrrs
        
        # Prepare data for extrapolation using full flow dataset
        flow_full = self.flow.copy()

        # Set up dataframe with flow data for full time period
        df_long = pd.DataFrame(
            {
                "flow_log": np.log(flow_full[flow_column]),
                "m": flow_full.index.month,
                "y": flow_full.index.year,
            }
        )

        # Get monthly means and add quarter info
        df_long_m = df_long.resample("m").mean()
        df_long["quarter"] = [self.get_quarter(m) for m in df_long["m"]]
        df_long_m["quarter"] = [self.get_quarter(m) for m in df_long_m["m"]]

        # Use trained regression models to predict monthly diversions
        df_long_m["diversion_pred"] = 0.0
        for i in range(df_long_m.shape[0]):
            ind = df_long_m.index[i]
            q = df_long_m["quarter"].iloc[i]
            f = df_long_m["flow_log"].iloc[i]

            # Get random sample value from linear regression model
            pred = self.get_random_prediction_sample(
                lrms=lrms, 
                lrrs=lrrs, 
                quarter=q, 
                x=f
            )
            
            df_long_m.loc[ind, "diversion_pred"] = pred

        # For NJ, transform data back to original scale
        if self.loc == "nj":
            df_m["diversion"] = np.maximum(nj_trans_max - np.exp(df_m["diversion"]), 0)
            df_long_m["diversion_pred"] = np.maximum(
                nj_trans_max - np.exp(df_long_m["diversion_pred"]), 0
            )

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
            
            # Get subset of monthly data for the same quarter
            df_m_sub = df_m.loc[df_m["quarter"] == q]
            
            # Calculate squared distance in normalized space
            dist_squ = (f - df_m_sub["flow_log_norm"]) ** 2 + (
                n - df_m_sub["diversion_norm"]
            ) ** 2
            
            # Find index of minimum distance
            nn = np.argmin(dist_squ)
            df_long_m.loc[ind, "nn"] = df_m_sub.index[nn]

        # Use each month's nearest neighbor to get flow shape for predicted diversion at daily time step
        df_long["diversion_pred"] = -1
        for i, row in df_long_m.iterrows():
            m = row["m"]
            y = row["y"]
            
            # Get the daily diversions in nearest neighbor from shorter record
            df_long_idx = df_long.loc[
                np.logical_and(df_long["m"] == m, df_long["y"] == y)
            ].index
            
            # Get matching daily pattern from historical data
            df_m_match = df_m.loc[row["nn"]]
            df_match = df.loc[
                np.logical_and(df["m"] == df_m_match["m"], df["y"] == df_m_match["y"])
            ]
            
            # Scale daily diversions based on ratio of monthly prediction to match
            new_diversion = (
                df_match["diversion"].values
                * row["diversion_pred"]
                / df_m_match["diversion"]
            )
            
            # Warn if negative values are created
            if np.any(new_diversion < 0):
                print(f"Warning: Negative diversion values detected for {row['m']}-{row['y']}")
                
            # Adjust length of record when months have different number of days
            len_new = len(df_long_idx)
            len_match = len(new_diversion)
            if len_match > len_new:
                new_diversion = new_diversion[:len_new]
            elif len_match < len_new:
                new_diversion = np.append(
                    new_diversion, [new_diversion[-1]] * (len_new - len_match)
                )
                
            # Assign daily predicted values
            df_long.loc[df_long_idx, "diversion_pred"] = new_diversion

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

    def save(self):
        """
        Save the processed extrapolated diversion data to CSV.
        
        The data is saved to the output directory specified in self.output_dirs,
        with the filename format determined by the location (NYC or NJ).
        
        Raises
        ------
        ValueError
            If processed_data is not available or if location is invalid.
        """
        # Make sure the data has been processed
        if self.processed_data is None:
            raise ValueError("No processed data available. Run the process() method first.")
        
        # Save to the appropriate output file
        output_path = self.output_dirs["diversion"]
        self.processed_data.to_csv(output_path, index=False)
        print(f"Saved extrapolated diversion data to {output_path}")
        
    def plot(self, kind="regressions"):
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
        
        if kind == "regressions":
            self.plot_regressions()
        elif kind == "diversions":
            self.plot_diversions()
    
    def plot_regressions(self):
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
            
            # Plot sampled data during observed period
            data = self.df_long_m.loc[self.df_m.index].copy()
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
        fig_dir = self.pn.data.get_str("../figs")
        plt.savefig(
            f"{fig_dir}/extrapolation_{self.loc}_pt1.png", 
            dpi=400, bbox_inches="tight"
        )
        plt.close()

    def plot_diversions(self):
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
        fig_dir = self.pn.data.get_str("../figs")
        plt.savefig(
            f"{fig_dir}/extrapolation_{self.loc}_pt2.png", 
            dpi=400, bbox_inches="tight"
        )
        plt.close()