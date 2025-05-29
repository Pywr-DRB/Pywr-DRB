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
                assert all(g in self.flows.columns for g in gauges), f"Missing inflow gauge {g} for node {node}"
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
                assert all(g in self.flows.columns for g in gauges), f"Missing inflow gauge {g} for node {node}"
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
        - data/observations/reservoir_storage_mg.csv
            Volumetric storage for all reservoirs (with obs), columns are reservoir names.
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

        # Save reservoir storage volume
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
        