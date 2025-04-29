"""
This script retrieves daily observational data from USGS NWIS for a set of inflow,
release, and storage gauges associated with the Pywr-DRB model.

Steps:
1. Downloads raw inflow, release, and elevation data (using gauge numbers).
2. Saves raw inflow + release data combined as `gage_flow_raw.csv`.
3. Converts elevation to storage using reservoir-specific curves.
4. Aggregates gauges and renames columns to reservoir names.
5. Saves final model-ready CSVs:
   - `gage_flow_mgd.csv` (raw gauges with labels)
   - `catchment_inflow_mgd.csv` (aggregated inflow)
   - `reservoir_storage_mg.csv` (converted and aggregated storage)
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
from dataretrieval import nwis
import matplotlib.pyplot as plt

ACRE_FEET_TO_MG = 0.325851  # Acre-feet to million gallons
GAL_TO_MG = 1 / 1_000_000   # Gallons to million gallons


class DataRetriever:
    def __init__(self, 
                 start_date="1945-01-01", 
                 end_date=None,
                 out_dir="src/pywrdrb/data/observations/", 
                 default_stat_code="00003"): #mean
        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime("%Y-%m-%d")
        self.out_dir = out_dir
        self.default_stat_code = default_stat_code
        os.makedirs(self.out_dir, exist_ok=True)

    def get(self, gauges, param_cd=None, stat_cd=None, label_map=None, type="flow"):
        """Download USGS daily values for a list of gauges."""

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
                    print(f"⚠️  No expected columns found for site {g}")
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
        """Convert elevation to storage using appropriate storage curves."""
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
        """Aggregate multiple gauges to reservoir level and save."""
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
        """Save combined inflow + release gauge-level time series."""
        combined = pd.concat([inflow_df, release_df], axis=1)
        combined.index = pd.to_datetime(combined.index).date
        combined.index.name = "datetime"
        raw_path = os.path.join(self.out_dir, "_raw", "gage_flow_raw.csv")
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        combined.to_csv(raw_path)
        print(f"Saved raw gage flow data: {raw_path}")

    def save_to_csv(self, df, name):
        """Save generic dataframe to standard location."""
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
        full_range = pd.date_range(self.start_date, self.end_date, freq="D")
        return full_range.difference(df.index)


