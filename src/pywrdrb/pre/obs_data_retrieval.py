"""
Retrieves and prcoesses USGS observational data for pywrdrb relevant locations.

Overview:
This module defines the `ObservationalDataRetriever` class, which retrieves USGS 
data from the NWIS, processes, and saves it to the src/pywrdrb/data/observations directory. 

Key Steps:
1. Retrieve flow and elevation data from NWIS using USGS gauge IDs.
2. Convert elevation to storage using reservoir-specific curves.
3. Aggregate inflows, relabel from gauge IDs to node names
4. Save raw and processed data to data/observations directory.

Technical Notes:
- Handles both "catchment_inflows" (unmanaged reservoir inflows) and "gage_flows" (managed, total flow at USGS gauges)
- Uses the `dataretrieval` package to access NWIS data.

Example Usage:
from pywrdrb.pre import ObservationalDataRetriever
retriever = ObservationalDataRetriever()
retriever.load()
retriever.process()
retriever.save()

Change Log
----------
Marilyn Smith, 2025-05-07, Initial implementation of observational data retrieval and processing.
tja, 2025-05-20, Edited heavily, formatted as single class, better handling of catchment inflow vs gage flow
"""
import os
import numpy as np
import pandas as pd
import datetime
from dataretrieval import nwis

from pywrdrb.utils.constants import ACRE_FEET_TO_MG, GAL_TO_MG, cfs_to_mgd
from pywrdrb.pre.datapreprocessor_ABC import DataPreprocessor

from pywrdrb.pywr_drb_node_data import obs_site_matches, obs_pub_site_matches
from pywrdrb.pywr_drb_node_data import all_flow_gauges, nyc_reservoirs
from pywrdrb.pywr_drb_node_data import storage_curves, storage_gauge_map

__all__ = ["ObservationalDataRetriever"]

# DRBC-curated daily NYC reservoir storage in MG. Despite the filename, the
# series actually starts 1999-12-01 and ends 2021-11-30.
DRBC_NYC_STORAGE_FILENAME = "NYC_storage_daily_2000-2021.csv"

# Per-reservoir start dates for trustworthy storage observations. Anything
# before is set to NaN to drop known-bad early gauge records. Add new entries
# here as data quality issues are identified during diagnostic review.
STORAGE_VALID_FROM = {
    # USGS 01428900 has anomalous elevation values 1986-1990 then a long gap
    # until ~2017. The early window is unreliable; drop it.
    "prompton": "1990-01-01",
}

class ObservationalDataRetriever(DataPreprocessor):
    """
    A retriever class for observational reservoir data using the DataPreprocessor interface.

    This class collects inflow, release, and elevation data from USGS NWIS,
    processes and saves raw time series, and converts elevation to storage using
    predefined storage curves.
    """

    def __init__(self, 
                 start_date="1945-01-01", 
                 end_date=None):
        """
        Initialize an ObservationalDataRetriever instance.

        Parameters
        ----------
        start_date : str
            Start date for data retrieval, typically set to the model start date (e.g., '1980-01-01').
        end_date : str, optional
            End date for data retrieval, defaults to today's date if not provided.
        """
        super().__init__()
        
        self.start_date = start_date
        self.end_date = end_date or datetime.date.today().strftime("%Y-%m-%d")

        # directories
        self.raw_dir = self.pn.observations.get_str() + os.sep + "_raw"
        self.processed_dir = self.pn.observations.get_str()

        # USGS statistic code; default is mean val (00003)
        self.default_stat_code = "00003"
        
        ## Lists of USGS gauge IDs for different data types
        # All USG flow gauges, unmanaged inflows and managed downstream flows
        self.all_flow_gauges = all_flow_gauges
        
        # reservoir elevation gauges
        self.nyc_storage_gauge_map = {n:v for n, v in storage_gauge_map.items() if n in nyc_reservoirs}
        self.non_nyc_storage_gauge_map = {n:v for n, v in storage_gauge_map.items() if n not in nyc_reservoirs}
        self.nyc_storage_gauges = self._flatten_gauges_from_dict_vals(self.nyc_storage_gauge_map)
        self.non_nyc_storage_gauges = self._flatten_gauges_from_dict_vals(self.non_nyc_storage_gauge_map)

        # DRBC NYC storage (used to fill the pre-2019 USGS gap)
        self.drbc_nyc_storage_path = os.path.join(self.raw_dir, DRBC_NYC_STORAGE_FILENAME)
        self.drbc_nyc_storages = None
        self.usgs_nyc_storages = None


    def get(self, 
            gauges, 
            param_cd=None, 
            stat_cd=None, 
            label_map=None, 
            type="flow"):
        """
        Download USGS daily time series for specified gauges.

        Parameters
        ----------
        gauges : list of str
            List of USGS gauge site IDs to retrieve.
        param_cd : str, optional
            USGS parameter code (e.g., '00060' for flow). Defaults set based on `type`.
        stat_cd : str, optional
            USGS statistic code (e.g., '00003' for mean). Defaults to `default_stat_code`.
        label_map : dict, optional
            Mapping of gauge site IDs to custom column names.
        type : str, optional
            Type of data to retrieve. One of ['flow', 'elevation_std', 'elevation_nyc'].
            Default is 'flow'.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame of time series indexed by date and labeled by gauge ID or custom label.

        Raises
        ------
        ValueError
            If `type` is not one of the recognized values.
        """

        if type == "flow":
            param_cd = "00060"
            stat_cd = stat_cd or self.default_stat_code  # usually '00003'
        elif type == "elevation":
            param_cd = "00062"
            stat_cd = stat_cd 
        elif type == "elevation_nyc":
            param_cd = "62615"
            stat_cd = stat_cd 
        else:
            raise ValueError(f"Unknown type '{type}'. Must be one of ['flow', 'elevation_std', 'elevation_nyc']")


        if type == "flow":
            expected_cols = [f"{param_cd}_Mean"]
        elif type.startswith("elevation"):
            expected_cols = [f"{param_cd}_Mean", f"{param_cd}_Minimum", f"{param_cd}_Maximum", param_cd]
        else:
            expected_cols = [param_cd]  # fallback


        all_dfs = []
        for g in gauges:
            try:
                data = nwis.get_dv(
                    sites=g, parameterCd=param_cd, statCd=stat_cd,
                    start=self.start_date, end=self.end_date)[0]
                data.reset_index(inplace=True)
                data["datetime"] = pd.to_datetime(data["datetime"])


                #   Try to find a matching column
                found_col = next((col for col in expected_cols if col in data.columns), None)

                if not found_col:
                    print(f"  No expected columns found for site {g}")
                    print(f"    Expected: {expected_cols}")
                    print(f"    Available: {list(data.columns)}")
                    print(f"    Sample data:\n{data.head(2)}\n")
                    continue  # skip this gauge

                # Keep only the param_cd_Mean column and rename to gauge ID
                data = data.loc[:, ["datetime", f"{param_cd}_Mean"]]
                data.rename(columns={f"{param_cd}_Mean": g}, inplace=True)

                data.set_index("datetime", inplace=True)
                all_dfs.append(data)

            except Exception as e:
                print(f"Failed to retrieve {g}: {e}")

        if not all_dfs:
            return pd.DataFrame()

        df_combined = pd.concat(all_dfs, axis=1)
        print(f"Retrieved data for: {df_combined.columns.tolist()}")
        
        # make index datetime
        df_combined.index = pd.to_datetime(df_combined.index).date
        df_combined.index.name = "datetime"
    
        # replace <0.0 with NaN
        df_combined[df_combined <= 0.0] = np.nan

        return df_combined

    def elevation_to_storage(self, 
                             elevation_df, 
                             storage_curve_dict):
        """
        Convert reservoir elevation time series to volume using storage curves.

        Parameters
        ----------
        elevation_df : pd.DataFrame
            DataFrame with elevation time series (indexed by datetime, columns as gauge IDs).
        storage_curve_dict : dict
            Dictionary mapping gauge IDs to CSV file paths for elevation-storage curves.

        Returns
        -------
        pd.DataFrame
            DataFrame of converted storage time series in million gallons (MG), indexed by datetime.

        Raises
        ------
        ValueError
            If no valid elevation series can be converted to storage.
        """
        storage_dfs = []

        for col in elevation_df.columns:
            res_name = col # this is the gauge ID
            curve_file = storage_curve_dict.get(res_name)

            if not curve_file or not os.path.exists(curve_file):
                print(f"Storage curve missing for {res_name}. Skipping.")
                storage_dfs.append(pd.Series(name=res_name))
                continue

            try:
                curve = pd.read_csv(curve_file)
                if "Elevation (ft)" not in curve.columns:
                    print(f"[Invalid Curve] Missing 'Elevation (ft)' in curve for {res_name}")
                    storage_dfs.append(pd.Series(name=res_name))
                    continue
                curve.set_index("Elevation (ft)", inplace=True)
            except Exception as e:
                print(f"[Curve Error] Failed to read curve for {res_name}: {e}")
                storage_dfs.append(pd.Series(name=res_name))
                continue

            # Interpolate using the column labeled by the gauge ID
            if "Acre-Ft" in curve.columns:
                expected_col = "Acre-Ft"
                conversion_factor = ACRE_FEET_TO_MG
            elif "Volume, gal" in curve.columns:
                expected_col = "Volume, gal"
                conversion_factor = GAL_TO_MG
            else:
                print(f"[Missing Curve Column] No known volume column in curve for {res_name}")
                storage_dfs.append(pd.Series(name=res_name))
                continue

            if expected_col not in curve.columns:
                print(f"[Missing Curve Column] '{expected_col}' missing in curve for {res_name}")
                storage_dfs.append(pd.Series(name=res_name))
                continue

            if res_name not in elevation_df.columns:
                print(f"[Missing Data Column] Elevation column for {res_name} not found in elevation_df")
                storage_dfs.append(pd.Series(name=res_name))
                continue

            series = elevation_df[res_name].apply(
                lambda x: np.interp(x, curve.index.values, curve[expected_col].values) * conversion_factor
                if pd.notnull(x) else np.nan
            )
            series.name = res_name
            storage_dfs.append(series)
            
        if not storage_dfs:
            raise ValueError("No valid storage series found — nothing to concatenate.")

        return pd.concat(storage_dfs, axis=1)


    def _flatten_gauges_from_dict_vals(self, gauge_dict):
        """
        Flatten a nested gauge mapping dictionary into a sorted list of unique gauges.

        Parameters
        ----------
        gauge_dict : dict
            Dictionary mapping reservoir names to lists of gauge IDs.

        Returns
        -------
        list of str
            Flattened and sorted list of all unique gauge IDs.
        """

        all_gauges = []
        for gauges in gauge_dict.values():
            if isinstance(gauges, list):
                if len(gauges) > 0:            
                    all_gauges.extend(gauges)
        return all_gauges


    def _load_drbc_nyc_storage(self):
        """
        Load the DRBC daily NYC reservoir storage CSV.

        The file lives at ``_raw/NYC_storage_daily_2000-2021.csv`` and provides
        clean daily storage in Million Gallons for cannonsville, pepacton, and
        neversink. Despite the filename, the series spans 1999-12-01 to
        2021-11-30. The first cell carries a UTF-8 BOM, so we read with
        ``encoding="utf-8-sig"``.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by ``datetime.date`` with columns
            ``[pepacton, cannonsville, neversink]`` in MG. Cached on
            ``self.drbc_nyc_storages``.
        """
        df = pd.read_csv(
            self.drbc_nyc_storage_path,
            index_col=0,
            parse_dates=[0],
            date_format="%m/%d/%Y",
            encoding="utf-8-sig",
        )

        keep_cols = [r for r in nyc_reservoirs if r in df.columns]
        df = df[keep_cols]

        df.index = pd.to_datetime(df.index).date
        df.index.name = "datetime"

        df[df <= 0.0] = np.nan

        self.drbc_nyc_storages = df
        return df


    def _merge_drbc_nyc_storage(self):
        """
        Merge DRBC NYC storage into ``self.storages``.

        Precedence: DRBC overrides USGS-derived values for the DRBC date range
        (1999-12-01 -> 2021-11-30). USGS is preserved for 2021-12-01 onward and
        used as a defensive fallback inside the DRBC range only where DRBC is
        NaN. The pre-merge USGS NYC slice is cached on ``self.usgs_nyc_storages``
        for diagnostics.

        Assumes ``self.storages`` already has reservoir-name columns (i.e. this
        runs after the gauge-id -> reservoir-name rename in ``process()``).
        """
        # USGS NWIS occasionally returns duplicate datestamps; the existing
        # process() comment notes this. Drop duplicates (keep first) before
        # any reindex/loc-by-label work.
        if not self.storages.index.is_unique:
            n_dup = int(self.storages.index.duplicated().sum())
            self.storages = self.storages[~self.storages.index.duplicated(keep="first")]
            print(f"  Dropped {n_dup} duplicate datestamps from storages before DRBC merge.")
        self.storages = self.storages.sort_index()
        self.storages.index.name = "datetime"

        # Cache pre-merge USGS NYC storage for diagnostics / audit
        nyc_cols = [r for r in nyc_reservoirs if r in self.storages.columns]
        self.usgs_nyc_storages = self.storages[nyc_cols].copy()

        drbc = self._load_drbc_nyc_storage()
        if not drbc.index.is_unique:
            drbc = drbc[~drbc.index.duplicated(keep="first")]
        drbc = drbc.sort_index()

        # Defensive index union (USGS already covers DRBC range, but cheap)
        merged_index = pd.Index(self.storages.index).union(pd.Index(drbc.index))
        if len(merged_index) != len(self.storages.index):
            self.storages = self.storages.reindex(merged_index)
            self.storages.index.name = "datetime"

        n_filled = 0
        n_usgs_fallback = 0
        for r in nyc_cols:
            if r not in drbc.columns:
                continue
            drbc_series = drbc[r].reindex(self.storages.index)
            usgs_series = self.usgs_nyc_storages[r].reindex(self.storages.index)

            # Within DRBC date range: DRBC overrides; if DRBC NaN, fall back to USGS
            in_drbc_range = drbc_series.notna()
            self.storages.loc[in_drbc_range, r] = drbc_series[in_drbc_range].values
            n_filled += int(in_drbc_range.sum())

            drbc_idx = drbc.index
            drbc_mask = self.storages.index.isin(drbc_idx)
            usgs_only_mask = drbc_mask & drbc_series.isna() & usgs_series.notna()
            if usgs_only_mask.any():
                self.storages.loc[usgs_only_mask, r] = usgs_series[usgs_only_mask].values
                n_usgs_fallback += int(usgs_only_mask.sum())

        post_drbc_mask = self.storages.index > drbc.index.max()
        n_post = int(self.storages.loc[post_drbc_mask, nyc_cols].notna().any(axis=1).sum())

        print(
            f"DRBC merge: filled {n_filled} NYC values from "
            f"{drbc.index.min()} to {drbc.index.max()} "
            f"(USGS fallback inside range: {n_usgs_fallback}); "
            f"preserved {n_post} dates with USGS NYC values after {drbc.index.max()}."
        )


    def _apply_storage_valid_from(self):
        """
        Drop known-bad early storage observations on a per-reservoir basis.

        For each ``reservoir -> "YYYY-MM-DD"`` entry in ``STORAGE_VALID_FROM``,
        set ``self.storages[reservoir]`` to NaN for any date strictly before the
        cutoff. Intended for cases where a USGS elevation gauge has a brief
        early-record anomaly followed by a multi-decade gap (e.g. prompton
        1986-1990).
        """
        if not STORAGE_VALID_FROM:
            return

        index_as_dt = pd.to_datetime(self.storages.index)
        for reservoir, cutoff in STORAGE_VALID_FROM.items():
            if reservoir not in self.storages.columns:
                continue
            cutoff_ts = pd.Timestamp(cutoff)
            mask = index_as_dt < cutoff_ts
            n_dropped = int(self.storages.loc[mask, reservoir].notna().sum())
            if n_dropped > 0:
                self.storages.loc[mask, reservoir] = np.nan
                print(f"  Dropped {n_dropped} {reservoir} storage values before {cutoff}.")


    def load(self):
        """
        Download raw observational data from USGS NWIS.

        Retrieves inflows, releases, and elevation data for both NYC and standard reservoirs.

        Returns
        -------
        None
            All data is stored as instance attributes.
        """
        print("Loading data from USGS NWIS...")
        
        ### USGS flow gauges
        self.flows = self.get(self.all_flow_gauges, type="flow")

        # convert from CFS to MGD
        self.flows = self.flows * cfs_to_mgd


        ### USGS elevation gauges
        # For some reason, NYC reservoirs have different parameter codes
        # than standard reservoirs, so we need to get them separately
        self.nyc_elevations = self.get(self.nyc_storage_gauges, type="elevation_nyc")
        self.non_nyc_elevations = self.get(self.non_nyc_storage_gauges, type="elevation")
    
        # combine into single dataframe
        self.nyc_elevations.index = pd.to_datetime(self.nyc_elevations.index).date
        self.non_nyc_elevations.index = pd.to_datetime(self.non_nyc_elevations.index).date
        
        self.elevations = pd.concat(
            [self.nyc_elevations, self.non_nyc_elevations], axis=1
        )
    


    def process(self):
        """
        Transform and combine elevation data into usable storage time series.

        Combines NYC and standard reservoir elevation data, then uses storage curves
        to convert elevation into volume.

        Returns
        -------
        None
            Processed data is saved in `self.storage_converted`.
        """
        ### Raw gauge ID
        ## Gauge flows
        # Convert from gauge IDs to node names
        self.gage_flows = pd.DataFrame(index=self.flows.index)
        self.gage_flows.index.name = "datetime"
        for node, gauges in obs_site_matches.items():
            
            # if no inflow gauges, add column of NaNs
            if len(gauges) == 0:
                self.gage_flows[node] = np.nan
            
            # otherwise, sum (for reservoir inflows) and rename (for all gauges)
            else:
                # check that all gauges are in the inflow dataframe
                assert all(
                    g in self.flows.columns for g in gauges
                    ), f"Missing inflow gauges {[g for g in gauges if g not in self.flow.columns]} for node {node}"
                self.gage_flows[node] = self.flows[gauges].sum(axis=1)                
        
        ### Processed and transformed data
        ## Inflows (only unmanaged flow data)
        # Aggregate (sum) inflow gauges and rename from gauge IDs to node names
        self.catchment_inflows = pd.DataFrame(index=self.flows.index)
        self.catchment_inflows.index.name = "datetime"
        
        for node, gauges in obs_pub_site_matches.items():
            
            # if no inflow gauges, add column of NaNs
            if gauges is None:
                self.catchment_inflows[node] = np.nan
            
            # otherwise, sum inflow from gauges
            else:
                # check that all gauges are in the inflow dataframe
                assert all(
                    g in self.flows.columns for g in gauges
                    ), f"Missing inflow gauges {[g for g in gauges if g not in self.flows.columns]} for node {node}"
                self.catchment_inflows[node] = self.flows[gauges].sum(axis=1)
        
        ## Storage
        # For some reason, the storages dataframe has datetime which is not in order
        # this originally is due to elevations data being misaligned
        # sort elevations by datetime
        self.elevations = self.elevations.sort_index()
        self.elevations.index.name = "datetime"
        
        # Convert elevation to volume using storage curves
        self.storages = self.elevation_to_storage(
            self.elevations, storage_curves
        )
        
        # Rename columns to match reservoir names
        self.storages.rename(columns={v[0]:k for k,v in storage_gauge_map.items()},
                             inplace=True)
        self.storages.index.name = "datetime"

        # Merge DRBC NYC storage (extends NYC coverage back to 1999-12-01)
        self._merge_drbc_nyc_storage()

        # Drop known-bad early storage observations (e.g. prompton pre-1990)
        self._apply_storage_valid_from()

        ### For each dataframe, replace <=0.0 with NaN
        self.gage_flows[self.gage_flows <= 0.0] = np.nan
        self.catchment_inflows[self.catchment_inflows <= 0.0] = np.nan
        self.storages[self.storages <= 0.0] = np.nan
        
        

    def save(self):
        """
        Save raw and processed observational data.

        Final saved CSV files include:
        - data/observations/_raw/streamflow_daily_usgs_mgd.csv
            Gauge flow obs for all gauges of interest.  Includes full natural flows and managed flows.
        - data/observations/_raw/reservoir_elevation.csv
            Raw elevation gauge data for all reservoirs, columns are gauge IDs.
        - data/observations/_raw/usgs_nyc_storage_mg.csv
            Pre-merge USGS-derived NYC storage (cannonsville, pepacton, neversink),
            preserved for audit/comparison against the DRBC-merged series.
        - data/observations/reservoir_storage_mg.csv
            Volumetric storage for all reservoirs (with obs), columns are reservoir names.
            For NYC reservoirs, DRBC values override USGS-derived values over
            1999-12-01 -> 2021-11-30; USGS-derived values are used post-2021.
        - data/observations/catchment_inflow_mgd.csv
            Inflow data for all nodes (with obs), columns are node names.
        - data/observations/gage_flow_mgd.csv
            Streamflow at nodes (with obs), columns are node names. Includes full natural flows and managed flows.

        Returns
        -------
        None
        """
        # Save raw USGS flow data, columns are gauge IDS
        flows_df = self.flows.copy()
        flows_fname = os.path.join(self.raw_dir, "streamflow_daily_usgs_mgd.csv")
        flows_df.to_csv(flows_fname)
        
        # Save raw elevation data, columns are gauge IDs
        elev_df = self.elevations.copy()
        elev_fname = os.path.join(self.raw_dir, "reservoir_elevation_ft.csv")
        elev_df.to_csv(elev_fname)

        # Save pre-merge USGS-derived NYC storage for audit
        if self.usgs_nyc_storages is not None:
            usgs_nyc_fname = os.path.join(self.raw_dir, "usgs_nyc_storage_mg.csv")
            self.usgs_nyc_storages.to_csv(usgs_nyc_fname)

        # Save reservoir storage volume (DRBC-merged for NYC over its date range)
        storage_df = self.storages.copy()
        storage_fname = os.path.join(self.processed_dir, "reservoir_storage_mg.csv")
        storage_df.to_csv(storage_fname)
        
        # save inflow data, columns are node names
        inflow_df = self.catchment_inflows.copy()
        inflow_fname = os.path.join(self.processed_dir, "catchment_inflow_mgd.csv")
        inflow_df.to_csv(inflow_fname)
        
        # save gage flow data, columns are node names
        gage_flow_df = self.gage_flows.copy()
        gage_flow_fname = os.path.join(self.processed_dir, "gage_flow_mgd.csv")
        gage_flow_df.to_csv(gage_flow_fname)


    def plot_storage_diagnostics(self, save_dir=None, show=False):
        """
        Generate diagnostic plots for observed reservoir storage.

        Produces three figures:

        1. observed_storage_all_reservoirs.png -- one panel per reservoir
           covering the full date range, with NaN runs shaded so coverage gaps
           are visually obvious.
        2. nyc_storage_overlap_validation.png -- 2019-01-01 to 2022-06-01 view
           of cannonsville/pepacton/neversink showing USGS-derived, DRBC raw,
           and merged series overlaid; per-panel MAE between USGS and DRBC over
           the overlap is reported in the title.
        3. nyc_storage_full_history.png -- full-history NYC view with shaded
           backgrounds indicating which source is authoritative when, to
           confirm continuity across the 2021-11-30 splice.

        Requires ``process()`` to have already run (so ``self.storages``,
        ``self.usgs_nyc_storages``, and ``self.drbc_nyc_storages`` are set).

        Parameters
        ----------
        save_dir : str, optional
            Directory to write PNG files into. Defaults to
            ``experiments/observed_storage_diagnostics`` relative to CWD.
        show : bool, optional
            If True, call ``plt.show()`` after saving. Default False.

        Returns
        -------
        dict
            Mapping ``figure_name -> saved_path``.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if save_dir is None:
            save_dir = os.path.join("experiments", "observed_storage_diagnostics")
        os.makedirs(save_dir, exist_ok=True)

        if self.storages is None:
            raise RuntimeError("plot_storage_diagnostics requires process() to have run.")

        def _to_datetime_unique(df):
            """Convert index to DatetimeIndex and drop any duplicates introduced
            by mixed date/Timestamp object types in the source index."""
            if df is None:
                return None
            out = df.copy()
            out.index = pd.to_datetime(out.index)
            if not out.index.is_unique:
                out = out[~out.index.duplicated(keep="first")]
            return out.sort_index()

        storages_dt = _to_datetime_unique(self.storages)

        saved = {}

        # --- Figure 1: all reservoirs, full date range, NaN runs shaded -----
        cols = list(storages_dt.columns)
        fig, axes = plt.subplots(len(cols), 1, sharex=True, figsize=(11, 1.6 * len(cols)))
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            ax.plot(storages_dt.index, storages_dt[col].values, linewidth=0.7, color="tab:blue")
            # Shade contiguous NaN runs
            isna = storages_dt[col].isna().values
            if isna.any():
                # find run boundaries
                edges = np.diff(np.concatenate(([0], isna.astype(int), [0])))
                starts = np.where(edges == 1)[0]
                ends = np.where(edges == -1)[0]
                idx = storages_dt.index
                for s, e in zip(starts, ends):
                    if e > s:
                        x0 = idx[s]
                        x1 = idx[min(e, len(idx) - 1)]
                        ax.axvspan(x0, x1, color="lightgray", alpha=0.4, linewidth=0)
            ax.set_ylabel(col, fontsize=8)
            ax.tick_params(axis="both", labelsize=7)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("Date")
        axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.suptitle(
            "Observed reservoir storage (MG) -- gray shading = NaN.\n"
            "NYC reservoirs: DRBC 1999-12-01 -> 2021-11-30, USGS-derived after.",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(save_dir, "observed_storage_all_reservoirs.png")
        fig.savefig(path, dpi=150)
        if not show:
            plt.close(fig)
        saved["observed_storage_all_reservoirs"] = path

        # --- Figure 2: NYC overlap validation 2019-2022 ---------------------
        nyc_cols = [r for r in nyc_reservoirs if r in storages_dt.columns]
        usgs = self.usgs_nyc_storages
        drbc = self.drbc_nyc_storages
        usgs_dt = _to_datetime_unique(usgs)
        drbc_dt = _to_datetime_unique(drbc)

        fig, axes = plt.subplots(len(nyc_cols), 1, sharex=True, figsize=(11, 2.2 * len(nyc_cols)))
        if len(nyc_cols) == 1:
            axes = [axes]
        x_lo = pd.Timestamp("2019-01-01")
        x_hi = pd.Timestamp("2022-06-01")
        drbc_end = pd.Timestamp("2021-11-30")
        for ax, col in zip(axes, nyc_cols):
            mae_str = ""
            if usgs_dt is not None and drbc_dt is not None and col in usgs_dt and col in drbc_dt:
                joined = pd.concat(
                    {"u": usgs_dt[col], "d": drbc_dt[col]}, axis=1
                ).dropna()
                if not joined.empty:
                    mae = float(np.mean(np.abs(joined["u"].values - joined["d"].values)))
                    mae_str = f"  USGS-vs-DRBC MAE over overlap: {mae:,.0f} MG (n={len(joined)})"

            if usgs_dt is not None and col in usgs_dt:
                ax.plot(usgs_dt.index, usgs_dt[col].values,
                        linestyle="--", color="gray", linewidth=1.0, label="USGS-derived")
            if drbc_dt is not None and col in drbc_dt:
                ax.plot(drbc_dt.index, drbc_dt[col].values,
                        linestyle="-", color="tab:blue", linewidth=1.2, label="DRBC")
            ax.plot(storages_dt.index, storages_dt[col].values,
                    linestyle=":", color="black", linewidth=1.0, alpha=0.8, label="merged (loaded)")
            ax.axvline(drbc_end, color="tab:red", linewidth=0.8, alpha=0.7)
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylabel(f"{col}\n(MG)", fontsize=8)
            ax.set_title(col + mae_str, fontsize=9, loc="left")
            ax.tick_params(axis="both", labelsize=7)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="lower left")
        axes[-1].set_xlabel("Date")
        fig.suptitle("NYC storage: USGS vs DRBC vs merged (red line = DRBC end)", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(save_dir, "nyc_storage_overlap_validation.png")
        fig.savefig(path, dpi=150)
        if not show:
            plt.close(fig)
        saved["nyc_storage_overlap_validation"] = path

        # --- Figure 3: NYC full history with source shading -----------------
        fig, axes = plt.subplots(len(nyc_cols), 1, sharex=True, figsize=(11, 2.0 * len(nyc_cols)))
        if len(nyc_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, nyc_cols):
            ax.plot(storages_dt.index, storages_dt[col].values,
                    color="black", linewidth=0.7)
            # Shade source regions
            if drbc_dt is not None and not drbc_dt.empty:
                ax.axvspan(drbc_dt.index.min(), drbc_dt.index.max(),
                           color="tab:blue", alpha=0.08, linewidth=0, label="DRBC range")
            if usgs_dt is not None:
                post_idx = storages_dt.index[storages_dt.index > drbc_end]
                if len(post_idx) > 0:
                    ax.axvspan(post_idx.min(), post_idx.max(),
                               color="tab:orange", alpha=0.08, linewidth=0, label="USGS range")
            ax.axvline(drbc_end, color="tab:red", linewidth=0.8, alpha=0.7)
            ax.set_ylabel(f"{col}\n(MG)", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="lower right")
        axes[-1].set_xlabel("Date")
        fig.suptitle(
            "NYC storage full history -- blue = DRBC, orange = USGS, red line = splice (2021-11-30)",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(save_dir, "nyc_storage_full_history.png")
        fig.savefig(path, dpi=150)
        if not show:
            plt.close(fig)
        saved["nyc_storage_full_history"] = path

        print(f"Wrote {len(saved)} diagnostic figures to {save_dir}")
        return saved


if __name__ == "__main__":
    """
    Run observational data retrieval when executed directly.

    Usage:
        python -m pywrdrb.pre.obs_data_retrieval

    This will retrieve all USGS observational data including:
    - Flow data at all gauges (including flood monitoring gauges)
    - Reservoir elevation data
    - Storage data (converted from elevation)
    """
    start_date = "1945-01-01"
    end_date = None

    print("=" * 60)
    print("USGS Observational Data Retrieval")
    print("=" * 60)
    print(f"Start date: {start_date}")
    print(f"End date: {end_date or 'today'}")
    print()

    retriever = ObservationalDataRetriever(
        start_date=start_date,
        end_date=end_date
    )

    print("Step 1: Loading data from USGS NWIS...")
    retriever.load()

    print("\nStep 2: Processing data...")
    retriever.process()

    print("\nStep 3: Saving data...")
    retriever.save()

    print("\nStep 4: Generating diagnostic plots...")
    try:
        retriever.plot_storage_diagnostics()
    except Exception as e:
        print(f"  Diagnostic plotting failed: {e}")

    print("\n" + "=" * 60)
    print("Data retrieval complete!")
    print("=" * 60)
        