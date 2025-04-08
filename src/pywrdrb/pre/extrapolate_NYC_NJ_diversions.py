import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pygeohydro import NWIS
import datetime

import pywrdrb
from pywrdrb.utils.constants import cfs_to_mgd, cms_to_mgd

# TODO: use pn to get the directories
from pywrdrb.utils.directories import input_dir, fig_dir

# Get pathnavigator object
global pn
pn = pywrdrb.get_pn_object()


class ExtrapolatedDiversionPreprocessor():
    def __init__(self, 
                 loc,
                 start_date,
                 end_date,):
        
        assert loc in ["nyc", "nj"], "Invalid location specified. Expected 'nyc' or 'nj'. Got {}".format(loc)
        
        self.loc = loc
        self.dates = (start_date, end_date)
        
        # quarters (seasons) used for different models
        self.quarters = ("DJF", "MAM", "JJA", "SON")
        
        self.lrms = None
        self.lrrs = None
        self.diversion = None
        self.flow = None
        self.df = None
        self.df_long = None

        ### set seed for consistent results
        np.random.seed(1)

    
    def load(self):
        self.diversion = None
        self.flow = None
        
        ### get historical diversion data
        if loc == "nyc":
            fname = pn.data.observations.get_str("_raw", "Pep_Can_Nev_diversions_daily_2000-2021.xlsx")
            diversion = pd.read_excel(fname, index_col=0,)
            diversion = diversion.iloc[:, :3]
            diversion.index = pd.to_datetime(diversion.index)
            diversion["aggregate"] = diversion.sum(axis=1)
            diversion = diversion.loc[np.logical_not(np.isnan(diversion["aggregate"]))]
            ### convert CFS to MGD
            diversion *= cfs_to_mgd
        elif loc == "nj":
            ### now get NJ demands/deliveries
            ### The gage for D_R_Canal starts 1989-10-23, but lots of NA's early on. Pretty good after 1991-01-01, but a few remaining to clean up.
            start_date = (1991, 1, 1)
            fname = pn.data.observations.get_str("_raw", "streamflow_daily_usgs_1950_2022_cms_for_NYC_NJ_diversions.csv")
            diversion = pd.read_csv(fname)
            diversion.index = pd.DatetimeIndex(diversion["datetime"])
            diversion = diversion[["D_R_Canal"]]
            diversion = diversion.loc[diversion.index >= datetime.datetime(*start_date)]

            ### infill NA values with previous day's flow
            for i in range(1, diversion.shape[0]):
                if np.isnan(diversion["D_R_Canal"].iloc[i]):
                    ind = diversion.index[i]
                    diversion.loc[ind, "D_R_Canal"] = diversion["D_R_Canal"].iloc[i - 1]

            ### convert cms to mgd
            diversion *= cms_to_mgd
            ### flow becomes negative sometimes, presumably due to storms and/or drought reversing flow. dont count as deliveries
            diversion[diversion < 0] = 0

        ### get historical flows
        fname = pn.data.observations.get_str("_raw", "streamflow_daily_usgs_1950_2022_cms_for_NYC_NJ_diversions.csv")
        flow = pd.read_csv(fname)
        flow.index = pd.to_datetime(flow["datetime"])
        return diversion, flow
    
    
    def process(self):
        pass
    
    def train_regressions(self, df_m):
        
        assert(["diversion", "flow_log", "m", "y", "quarter"] in df_m.columns), "Expected columns 'diversion', 'flow_log', 'm', 'y', 'quarter'."
        
        lrms = {
            q: sm.OLS(
                df_m["diversion"].loc[df_m["quarter"] == q],
                sm.add_constant(df_m["flow_log"].loc[df_m["quarter"] == q]),
            )
            for q in quarters
        }
        lrrs = {q: lrms[q].fit() for q in quarters}
        return lrms, lrrs
    
    def get_random_prediction_sample(self, 
                                     lrms, lrrs, 
                                     quarter, x):
        
        lrm = lrms[quarter]
        lrr = lrrs[quarter]
        exog = lrm.exog
        exog[:, 1] = x
        
        # throw out if negative
        pred = -1
        while pred < 0:
            pred = lrm.get_distribution(
                lrr.params, scale=np.var(lrr.resid), exog=exog
            ).rvs()[0]
        
        return pred
    
    def get_overlapping_timespan(self, df1, df2):
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
    
    def extrapolate_NYC_NJ_diversions(self):
        """
        Function for retrieving NYC and NJ historical diversions and extrapolating them into time periods
        where we don't have data based on seasonal flow regressions.

        Args:
            loc (str): The location to extrapolate. Options: "nyc" or "nj".
            make_figs (bool): Whether to make figures of the extrapolation process.

        Returns:
            pd.DataFrame: The dataframe containing the extrapolated diversions.
        """

        ### Load data
        # Get historical diversion and streamflows for regression
        diversion_full_series, flow_full_series = self.load()
        
        # make copies to keep the full version for later
        flow = flow_full_series.copy()
        diversion = diversion_full_series.copy()
        
        # get maximum overlapping timespan for diversions and flow
        flow, diversion = self.get_overlapping_timespan(flow, diversion)
        

        # dataframe of daily states
        diversion_column = "aggregate" if loc == "nyc" else "D_R_Canal"
        flow_column = "NYC_inflow" if loc == "nyc" else "delTrenton"
        
        df = pd.DataFrame(
            {
                "diversion": diversion[diversion_column],
                "flow_log": np.log(flow[flow_column]),
                "m": diversion.index.month,
                "y": diversion.index.year,
            }
        )

        # dataframe of monthly mean states
        df_m = df.resample("m").mean()
        df["quarter"] = [self.get_quarter(m) for m in df["m"]]
        df_m["quarter"] = [self.get_quarter(m) for m in df_m["m"]]

        ### NJ diversion data are left skewed, so negate and then apply log transform
        if loc == "nj":
            nj_trans_max = df_m["diversion"].max() + 5
            df_m["diversion"] = np.log(nj_trans_max - df_m["diversion"])


        ### Train linear regression model for each quarter
        lrms, lrrs = self.train_regressions(df_m)

        
        ### Prepare data for extrapolation
        # now use longer dataset of flows for extrapolation
        flow = flow_full_series.copy()

        flow_column = "NYC_inflow" if loc == "nyc" else "delTrenton"

        df_long = pd.DataFrame(
            {
                "flow_log": np.log(flow[flow_column]),
                "m": flow.index.month,
                "y": flow.index.year,
            }
        )

        # Resample to monthly
        df_long_m = df_long.resample("m").mean()
        df_long["quarter"] = [get_quarter(m) for m in df_long["m"]]
        df_long_m["quarter"] = [get_quarter(m) for m in df_long_m["m"]]

        # use trained regression model to sample a delivery value
        # for each month based on log flow.
        df_long_m["diversion_pred"] = 0.0
        for i in range(df_long_m.shape[0]):
            ind = df_long_m.index[i]
            q = df_long_m["quarter"].iloc[i]
            f = df_long_m["flow_log"].iloc[i]

            # get random sample value from linear regression model
            pred = self.get_random_prediction_sample(lrms, lrrs, 
                                                     quarter = q, 
                                                     x = f)
            
            df_long_m.loc[ind, "diversion_pred"] = pred


        ### for NJ, transform data back to original scale
        if loc == "nj":
            df_m["diversion"] = np.maximum(nj_trans_max - np.exp(df_m["diversion"]), 0)
            df_long_m["diversion_pred"] = np.maximum(
                nj_trans_max - np.exp(df_long_m["diversion_pred"]), 0
            )

        ### now get nearest neighbor in normalized 2d space of log-flow&diversion, within q.
        flow_bounds = [df_m["flow_log"].min(), df_m["flow_log"].max()]
        diversion_bounds = [df_m["diversion"].min(), df_m["diversion"].max()]

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

        df_long_m["nn"] = -1
        for i in range(df_long_m.shape[0]):
            ind = df_long_m.index[i]
            q = df_long_m["quarter"].iloc[i]
            f = df_long_m["flow_log_norm"].iloc[i]
            n = df_long_m["diversion_pred_norm"].iloc[i]
            df_m_sub = df_m.loc[df_m["quarter"] == q]
            dist_squ = (f - df_m_sub["flow_log_norm"]) ** 2 + (
                n - df_m_sub["diversion_norm"]
            ) ** 2
            nn = np.argmin(dist_squ)
            df_long_m.loc[ind, "nn"] = df_m_sub.index[nn]
        

        ### now use each month's nearest neighbor to get flow shape for predicted diversion at daily time step
        df_long["diversion_pred"] = -1
        for i, row in df_long_m.iterrows():
            m = row["m"]
            y = row["y"]
            
            ### get the daily diversions in nearest neighbor from shorter record
            df_long_idx = df_long.loc[
                np.logical_and(df_long["m"] == m, df_long["y"] == y)
            ].index
            df_m_match = df_m.loc[row["nn"]]
            df_match = df.loc[
                np.logical_and(df["m"] == df_m_match["m"], df["y"] == df_m_match["y"])
            ]
            ### scale daily diversions based on ratio of monthly prediction to match
            new_diversion = (
                df_match["diversion"].values
                * row["diversion_pred"]
                / df_m_match["diversion"]
            )
            if np.any(new_diversion < 0):
                print(row, new_diversion)
            ### adjust length of record when months have different number of days
            len_new = len(df_long_idx)
            len_match = len(new_diversion)
            if len_match > len_new:
                new_diversion = new_diversion[:len_new]
            elif len_match < len_new:
                new_diversion = np.append(
                    new_diversion, [new_diversion[-1]] * (len_new - len_match)
                )
            df_long.loc[df_long_idx, "diversion_pred"] = new_diversion


        ### Now reload historical diversion dataset, 
        # & add extrapolated data for the dates we don't have
        if loc == "nyc":
            diversion = diversion_full_series.copy()

            ### format & save to csv for use in Pywr-DRB
            df_long = df_long.loc[
                np.logical_or(
                    df_long.index < diversion.index.min(),
                    df_long.index > diversion.index.max(),
                )
            ]
            diversion = pd.concat(
                [diversion, pd.DataFrame({"aggregate": df_long["diversion_pred"]})]
            )
            diversion = diversion.sort_index()
            diversion["datetime"] = diversion.index
            diversion.columns = [
                "pepacton",
                "cannonsville",
                "neversink",
                "aggregate",
                "datetime",
            ]
            diversion = diversion.iloc[:, [-1, 1, 0, 2, 3]]


        elif loc == "nj":
            diversion = diversion_full_series.copy()

            ### format & save to csv for use in Pywr-DRB
            df_long = df_long.loc[
                np.logical_or(
                    df_long.index < diversion.index.min(),
                    df_long.index > diversion.index.max(),
                )
            ]
            diversion = pd.concat(
                [diversion, pd.DataFrame({"D_R_Canal": df_long["diversion_pred"]})]
            )
            diversion = diversion.sort_index()
            diversion["datetime"] = diversion.index
            diversion = diversion.iloc[:, [-1, 0]]
            return 


    def save(self):
        if self.loc == "nyc":
            fname = 'deliveryNYC_ODRM_extrapolated.csv' 
        elif self.loc == 'nj':
            fname = 'deliveryNJ_DRCanal_extrapolated.csv'
        else:
            raise ValueError("Invalid location specified. Expected 'nyc' or 'nj'.")
        
        diversion.to_csv(f"{input_dir}{fname}", index=False)
        return 
    
    def get_quarter(self, m):
        """
        Return the quarter of the year for a given month.
        
        Args:
            m (int): The month.
        
        Returns:
            str: The quarter. (DJF, MAM, JJA, SON)
        """
        if m in (12, 1, 2):
            return "DJF"
        elif m in (3, 4, 5):
            return "MAM"
        elif m in (6, 7, 8):
            return "JJA"
        elif m in (9, 10, 11):
            return "SON"
        
    def plot(self, 
             kind="regressions"):
        """
        Makes a plot of the extrapolation process.
        
        Args:
            kind (str): The kind of plot to make. Options: "regressions", "diversions".
        
        Returns:
            None
        """
        
        assert(kind in ["regressions", "diversions"]), "Invalid kind. Expected 'regressions' or 'diversions'."
        
        if kind == "regressions":
            self.plot_regressions()
        elif kind == "diversions":
            self.plot_diversions()
    
    
    def plot_diversions(self):
        
        assert self.df is not None, "Data not loaded. Expected attribute `df`."
        assert self.df_long is not None, "Data not loaded. Expected attribute `df_long`."
        
        fig, ax = plt.subplots(
            1, 1, figsize=(5, 3), 
            gridspec_kw={"hspace": 0.2, "wspace": 0.2}
        )

        ### plot observed diversion daily timeseries
        ax.plot(
            self.df["diversion"],
            color="cornflowerblue",
            label="Observed",
            zorder=2,
            lw=0.5,
            alpha=0.7,
        )
        ### plot extrapolated
        ax.plot(
            self.df_long["diversion_pred"],
            color="darkgoldenrod",
            label="Extrapolated",
            zorder=1,
            lw=0.5,
            alpha=0.7,
        )

        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)

        ### clean up
        ax.set_ylim([0, ax.get_ylim()[1]])
        ylab = "Daily NYC diversion (MGD)" if self.loc == "nyc" else "Daily NJ diversion (MGD)"
        ax.set_ylabel(ylab)

        plt.savefig(
            f"{fig_dir}/extrapolation_{loc}_pt2.png", 
            dpi=400, bbox_inches="tight"
        )

    def plot_regressions(self, df_m, df_long_m, lrrs):
        
        fig, axs = plt.subplots(
            2, 2, figsize=(8, 8), 
            gridspec_kw={"hspace": 0.2, "wspace": 0.2}
        )
        for i, q in enumerate(self.quarters):
            row = 1 if i >= 2 else 0
            col = 1 if i % 2 == 1 else 0
            ax = axs[row, col]

            ### first plot observed data
            data = df_m.loc[df_m["quarter"] == q].copy()
            ax.scatter(
                data["flow_log"],
                data["diversion"],
                zorder=2,
                alpha=0.7,
                color="cornflowerblue",
                label="Observed",
            )
            ### now plot sampled data during observed period
            data = df_long_m.loc[df_m.index].copy()
            data = data.loc[data["quarter"] == q]
            ax.scatter(
                data["flow_log"],
                data["diversion_pred"],
                zorder=1,
                alpha=0.7,
                color="firebrick",
                label="Extrapolated over\nobserved period",
            )
            ### now plot sampled data during unobserved period
            data = df_long_m.loc[[i not in df_m.index for i in df_long_m.index]].copy()
            ax.scatter(
                data["flow_log"],
                data["diversion_pred"],
                zorder=0,
                alpha=0.7,
                color="darkgoldenrod",
                label="Extrapolated over\nunobserved period",
            )

            ### plot regression line
            xlim = ax.get_xlim()
            ax.plot(
                xlim,
                [lrrs[q].params[0] + lrrs[q].params[1] * x for x in xlim],
                color="k",
                label="Regression",
            )

            ### legend
            if row == 1 and col == 1:
                ax.legend(loc="center left", bbox_to_anchor=(1.0, 1.1), frameon=False)

            ### clean up
            ax.set_title(q)
            if row == 1:
                ax.set_xlabel("Log inflow (log MGD)")
            if loc == "nyc":
                ax.set_ylim([0, ax.get_ylim()[1]])
            if loc == "nyc" and col == 0:
                ax.set_ylabel("Monthly NYC diversion (MGD)")
            elif loc == "nj" and col == 0:
                ax.set_ylabel("Transformed monthly NJ diversion")

        plt.savefig(
            f"{fig_dir}/extrapolation_{loc}_pt1.png", dpi=400, bbox_inches="tight"
        )
