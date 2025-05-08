"""
Observational data retriever for Pywr-DRB model.

Overview
--------
This module defines the `ObservationalDataRetriever` class, which loads and processes 
observed inflow, release, and elevation data from USGS NWIS for use in the Pywr-DRB model.
It follows the `DataPreprocessor` interface and produces model-ready CSV files.

Key Steps
---------
1. Retrieve flow and elevation data from NWIS using USGS gauge IDs.
2. Convert elevation to storage using reservoir-specific curves.
3. Aggregate and save raw and processed data for inflow and storage inputs.

Technical Notes
---------------
- Interacts with `DataRetriever`, `PathNavigator`, and gauge mapping dictionaries.
- Depends on standardized storage curves and NYC vs non-NYC handling.
- Output paths are determined automatically using `get_pn_object()`.
- Inherits from `DataPreprocessor` abstract class to standardize preprocessing workflow.

Change Log
----------
Marilyn Smith, 2025-05-07, Initial implementation of observational data retrieval and processing.
"""
"""
Data retrieval and processing tools for observational data from USGS NWIS.

Overview
--------
This module defines the `DataRetriever` class, which downloads and processes
observational inflow, release, and elevation data from the USGS NWIS system.
It supports conversion of elevation data to storage volume using predefined
storage curves and outputs standardized CSV files compatible with the Pywr-DRB
modeling framework.

Key Steps
---------
1. Download raw daily data for specified gauge types (flow or elevation).
2. Convert elevation to volume using gauge-specific storage curves.
3. Aggregate raw gauge-level time series into model-ready inflow and storage files.

Technical Notes
---------------
- Depends on USGS NWIS `dataretrieval` package.
- Elevation-to-storage conversion uses custom CSV curves defined in `storage_curve_dict`.
- Handles NYC vs non-NYC storage gauge differences (e.g., parameter code 62615).
- Automatically saves data to Pywr-DRB structure using `PathNavigator`.
- Includes helper methods for quality checking (e.g., missing dates).

Change Log
----------
Marilyn Smith, 2025-05-07, Initial version with elevation-to-storage conversion logic.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from dataretrieval import nwis

from pywrdrb.path_manager import get_pn_object
pn = get_pn_object()

from pywrdrb.utils.constants import ACRE_FEET_TO_MG, GAL_TO_MG
from pywrdrb.pre.datapreprocessor_ABC import DataPreprocessor

from pywrdrb.pywr_drb_node_data import (
    inflow_gauge_map,
    release_gauge_map,
    storage_gauge_map,
    storage_curves,
    nyc_reservoirs
)

# Directories for raw and processed data
RAW_DATA_DIR = pn.observations.get_str() + os.sep + "_raw"
PROCESSED_DATA_DIR = pn.observations.get_str()

# from Chung-Yi
# no figures folder in pywrdrb
# repetive in ObservationalDataRetriever
#FIG_DIR = pn.figures.get_str()



class DataRetriever:
    """
    Retrieves and processes USGS NWIS observational data for reservoir modeling.

    Attributes
    ----------
    start_date : str
        Start date for data retrieval in 'YYYY-MM-DD' format.
    end_date : str
        End date for data retrieval (defaults to today's date).
    out_dir : str
        Output directory path for saved CSVs.
    default_stat_code : str
        Default statistic code used when querying USGS data.

    Methods
    -------
    get(gauges, param_cd=None, stat_cd=None, label_map=None, type="flow")
        Download daily values from NWIS for a list of gauges.
    elevation_to_storage(elevation_df, storage_curve_dict, nyc_reservoirs)
        Convert elevation values to storage using predefined curves.
    postprocess_and_save(df, reservoir_to_gauges, output_name)
        Aggregate gauge time series to reservoir-level and save as CSV.
    save_raw_gauge_data(inflow_df, release_df)
        Save combined inflow + release raw data.
    save_to_csv(df, name)
        Save generic DataFrame to CSV with standardized file naming.
    find_missing_dates(df)
        Identify missing dates in a DataFrame's datetime index.
    """
    def __init__(self, 
                 start_date="1945-01-01", 
                 end_date=None,
                 out_dir="src/pywrdrb/data/observations/", 
                 default_stat_code="00003"): #mean
        """
        Initialize the DataRetriever.

        Parameters
        ----------
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format. Default is '1945-01-01'.
        end_date : str, optional
            End date in 'YYYY-MM-DD' format. Defaults to today's date.
        out_dir : str, optional
            Output directory to save CSVs. Default is 'src/pywrdrb/data/observations/'.
        default_stat_code : str, optional
            USGS statistic code (e.g., '00003' for mean daily). Default is '00003'.

        Notes
        -----
        The `out_dir` will be created if it does not exist. The PathNavigator default
        is used if `out_dir` is not explicitly provided.
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime("%Y-%m-%d")

        if out_dir is None:
            self.out_dir = PROCESSED_DATA_DIR
        else:
            self.out_dir = out_dir

        os.makedirs(self.out_dir, exist_ok=True)
        self.default_stat_code = default_stat_code


    def get(self, gauges, param_cd=None, stat_cd=None, label_map=None, type="flow"):
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
        elif type == "elevation_std":
            param_cd = "00062"
            stat_cd = stat_cd 
        elif type == "elevation_nyc":
            param_cd = "62615"
            stat_cd = stat_cd 
        else:
            raise ValueError(f"Unknown type '{type}'. Must be one of ['flow', 'elevation_std', 'elevation_nyc']")

        all_dfs = []
        for g in gauges:
            try:
                data = nwis.get_dv(
                    sites=g, parameterCd=param_cd, statCd=stat_cd,
                    start=self.start_date, end=self.end_date)[0]
                data.reset_index(inplace=True)
                data["datetime"] = pd.to_datetime(data["datetime"])

                if type == "flow":
                    expected_cols = [f"{param_cd}_Mean"]
                elif type.startswith("elevation"):
                    expected_cols = [f"{param_cd}_Mean", f"{param_cd}_Minimum", f"{param_cd}_Maximum", param_cd]
                else:
                    expected_cols = [param_cd]  # fallback

                #   Try to find a matching column
                found_col = next((col for col in expected_cols if col in data.columns), None)

                if not found_col:
                    print(f"  No expected columns found for site {g}")
                    print(f"    Expected: {expected_cols}")
                    print(f"    Available: {list(data.columns)}")
                    print(f"    Sample data:\n{data.head(2)}\n")
                    continue  # skip this gauge

                col_name = label_map.get(g, g) if label_map else g
                data.set_index("datetime", inplace=True)
                renamed = data[[found_col]].rename(columns={found_col: col_name})
                all_dfs.append(renamed)

            except Exception as e:
                print(f"Failed to retrieve {g}: {e}")

        if not all_dfs:
            return pd.DataFrame()

        df_combined = pd.concat(all_dfs, axis=1)
        print(f"Retrieved data for: {df_combined.columns.tolist()}")
        return df_combined

    def elevation_to_storage(self, elevation_df, storage_curve_dict, nyc_reservoirs):
        """
        Convert reservoir elevation time series to volume using storage curves.

        Parameters
        ----------
        elevation_df : pd.DataFrame
            DataFrame with elevation time series (indexed by datetime, columns as gauge IDs).
        storage_curve_dict : dict
            Dictionary mapping gauge IDs to CSV file paths for elevation-storage curves.
        nyc_reservoirs : list of str
            List of NYC reservoir names (used to distinguish units and formats).

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

    def postprocess_and_save(self, df, reservoir_to_gauges, output_name):
        """
        Aggregate multiple gauge columns to reservoir-level series and save as CSV.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of gauge-level data (e.g., inflows or storage).
        reservoir_to_gauges : dict
            Dictionary mapping reservoir names to associated gauge IDs.
        output_name : str
            Filename for the output CSV.

        Returns
        -------
        None
            Output saved to disk in `out_dir`.
        """
        result_df = pd.DataFrame(index=df.index)

        for res, gauges in reservoir_to_gauges.items():
            valid_gauges = [g for g in gauges if g in df.columns]
            if valid_gauges:
                result_df[res] = df[valid_gauges].sum(axis=1)
            else:
                result_df[res] = np.nan
                print(f"No valid gauges found for {res}")

        result_df.index.name = "datetime"
        output_path = os.path.join(self.out_dir, output_name)
        result_df.to_csv(output_path)
        print(f"Saved aggregated data to: {output_path}")

    def save_raw_gauge_data(self, inflow_df, release_df):
        """
        Save raw inflow and release time series combined in a single CSV.

        Parameters
        ----------
        inflow_df : pd.DataFrame
            Inflow gauge-level time series.
        release_df : pd.DataFrame
            Release gauge-level time series.

        Returns
        -------
        None
            Output saved to 'gage_flow_raw.csv' in `raw_dir`.
        """
        combined = pd.concat([inflow_df, release_df], axis=1)
        combined.index = pd.to_datetime(combined.index).date
        combined.index.name = "datetime"
        raw_path = os.path.join(self.raw_dir, "gage_flow_raw.csv")
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        combined.to_csv(raw_path)
        print(f"Saved raw gage flow data: {raw_path}")

    def save_to_csv(self, df, name):
        """
        Save a generic DataFrame to CSV with standardized file naming.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to save, indexed by datetime.
        name : str
            Keyword used to determine output filename (e.g., 'inflow_raw').

        Returns
        -------
        None
            Output saved to `out_dir` with appropriate filename mapping.
        """
        name_map = {
            "inflow_raw": "gage_flow_mgd.csv",
            "storage_raw": "reservoir_storage_mg.csv",
            # Add others if needed
        }
        filename = name_map.get(name, f"{name}.csv")
        filepath = os.path.join(self.out_dir, filename)
        df.index = pd.to_datetime(df.index).date
        df.index.name = "datetime"
        df.to_csv(filepath)
        print(f"Saved: {filepath}")

    def find_missing_dates(self, df):
        """
        Identify missing dates in a DataFrame's datetime index.

        Parameters
        ----------
        df : pd.DataFrame
            Time series DataFrame indexed by datetime.

        Returns
        -------
        pd.DatetimeIndex
            Dates missing from the time series between `start_date` and `end_date`.
        """
        full_range = pd.date_range(self.start_date, self.end_date, freq="D")
        return full_range.difference(df.index)




class ObservationalDataRetriever(DataPreprocessor):
    """
    A retriever class for observational reservoir data using the DataPreprocessor interface.

    This class collects inflow, release, and elevation data from USGS NWIS,
    processes and saves raw time series, and converts elevation to storage using
    predefined storage curves.

    Attributes
    ----------
    start_date : str
        Start date for data retrieval in 'YYYY-MM-DD' format.
    retriever : DataRetriever
        Wrapper around NWIS data access and processing tools.
    pn : PathNavigator
        Object for managing standardized Pywr-DRB file paths.
    inflow_gauges : list of str
        Flattened list of USGS gauge IDs for inflow.
    release_gauges : list of str
        Flattened list of USGS gauge IDs for release.
    storage_gauges_std : list of str
        List of gauges associated with non-NYC reservoirs for elevation retrieval.
    storage_gauges_nyc : list of str
        List of gauges associated with NYC reservoirs for elevation retrieval.

    Methods
    -------
    load()
        Downloads raw inflow, release, and elevation data from NWIS.
    process()
        Combines and transforms elevation data into volume using storage curves.
    save()
        Saves raw and processed data to model-compatible CSV files.
    """

    def __init__(self, start_date: str):
        """
        Initialize an ObservationalDataRetriever instance.

        Parameters
        ----------
        start_date : str
            Start date for data retrieval, typically set to the model start date (e.g., '1980-01-01').

        Notes
        -----
        This class wraps around the DataRetriever class and formats output to match
        the Pywr-DRB expected input files (e.g., `reservoir_storage_mg.csv`).
        """
        super().__init__()
        self.start_date = start_date
        self.pn = get_pn_object()
        self.retriever = DataRetriever(start_date=start_date)
        self.raw_dir = self.pn.observations.get_str() + os.sep + "_raw"
        self.processed_dir = self.pn.observations.get_str()
        # self.fig_dir = self.pn.figures.get_str() # CL: error here. No such folder
        self._define_gauges()

    def _define_gauges(self):
        """
        Define lists of USGS gauge IDs for inflow, release, and storage.

        Splits storage gauges into NYC and non-NYC categories for separate elevation retrieval.

        Returns
        -------
        None
        """
        self.inflow_gauges = self._flatten_gauges(inflow_gauge_map)
        self.release_gauges = self._flatten_gauges(release_gauge_map)
        self.storage_gauges_nyc = [g for k, v in storage_gauge_map.items() if k in nyc_reservoirs for g in v]
        self.storage_gauges_std = [g for k, v in storage_gauge_map.items() if k not in nyc_reservoirs for g in v]

    def _flatten_gauges(self, gauge_dict):
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

        return sorted({g for gauges in gauge_dict.values() for g in gauges})

    def load(self):
        """
        Download raw observational data from USGS NWIS.

        Retrieves inflows, releases, and elevation data for both NYC and standard reservoirs.

        Returns
        -------
        None
            All data is stored as instance attributes (`self.inflows`, `self.releases`, etc.)
        """
        print("Loading data from USGS NWIS...")
        self.inflows = self.retriever.get(self.inflow_gauges, type="flow")
        self.releases = self.retriever.get(self.release_gauges, type="flow")
        self.elev_std = self.retriever.get(self.storage_gauges_std, type="elevation_std")
        self.elev_nyc = self.retriever.get(self.storage_gauges_nyc, type="elevation_nyc")

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
        self.elev_all = pd.concat([self.elev_std, self.elev_nyc], axis=1)
        self.storage_converted = self.retriever.elevation_to_storage(
            self.elev_all, storage_curves, nyc_reservoirs
        )

    def save(self):
        """
        Save raw and processed observational data to disk.

        Saves:
        - Raw inflow and release data
        - Raw elevation data
        - Converted storage time series
        - Aggregated model-ready inflow and storage CSVs

        Returns
        -------
        None
        """
        print("Saving raw and processed data to CSV...")
        self.retriever.save_raw_gauge_data(self.inflows, self.releases)
        self.retriever.save_to_csv(self.inflows, "inflow_raw")
        self.retriever.save_to_csv(self.elev_all, "elevation_raw")
        self.retriever.save_to_csv(self.storage_converted, "storage_raw")
        self.retriever.postprocess_and_save(self.inflows, inflow_gauge_map, "catchment_inflow_mgd.csv")
        self.retriever.postprocess_and_save(self.storage_converted, storage_gauge_map, "reservoir_storage_mg.csv")



