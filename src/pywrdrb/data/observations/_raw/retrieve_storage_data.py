# ##############################################################################
# # Marilyn Smith
# # Script to retrieve reservoir data from NWIS and convert elevation to volume
# # Data is saved to CSV files
# 
# # Overview:
# This script retrieves elevation data for multiple reservoirs from the National Water Information System (NWIS) 
# and converts the data into volume measurements (in Million Gallons, MG) based on predefined storage curves. 
# It processes the data by converting elevation measurements to volume for each reservoir, 
# depending on the reservoir's specific conversion requirements. 
# The script outputs the resulting volume time series to a CSV file for further analysis.
# 
# # Key Updates:
# - NYC Reservoirs: For the reservoirs Pepacton, Cannonsville, and Neversink, a different parameter code is used ('62615'), which returns only the mean value (not all three statistics). This change ensures accurate data retrieval for these specific reservoirs.
# - Acre-Feet to MG Conversion: For most reservoirs, the script converts the elevation data from Acre-Feet to Million Gallons (MG) using the ACRE_FEET_TO_MG conversion factor. However, for the NYC reservoirs (Pepacton, Cannonsville, Neversink), the script uses the 'Volume, gal' column from their storage curves, converting directly to MG using the GAL_TO_MG factor, as their data is already in gallons.
# 
# # Reservoirs and Conversion Details:
# - Acre-Feet to MG Conversion:
#     - Prompton
#     - Beltzville Combined
#     - Fewalter
#     - Blue Marsh
# - Direct Volume Conversion (Gallons to MG):
#     - Pepacton
#     - Cannonsville
#     - Neversink
# 
# The script processes the data for each reservoir, saves elevation data to CSV, and then applies the appropriate conversion based on the reservoir's specific needs. The final data is combined and saved into a single CSV file for all reservoirs.
# ##############################################################################

import pandas as pd
import numpy as np
from dataretrieval import nwis
import os
import sys
from datetime import datetime

# Define directories (relative paths)
raw_storage_dir = "."
processed_storage_dir = "../storage/"

# NYC reservoirs (use ODRM before 12/01/2021, NWIS after)
nyc_reservoirs = ["neversink", "cannonsville", "pepacton"]

# Lower Basin reservoirs (use NWIS only)
lower_basin_reservoirs = ["prompton", "beltzvilleCombined", "fewalter", "blueMarsh"]

# ODRM storage data file (historical NYC reservoir storage)
odrm_storage_file = os.path.join(processed_storage_dir, "reservoir_storage_nyc_mg.csv")

# Full NWIS-retrieved storage file
nwis_storage_file = os.path.join(raw_storage_dir, "observed_storage_full.csv")

# Define cutoff date for switching NYC data sources
odrm_end_date = "2021-11-30"

class DataPreprocessor:
    def __init__(self, storages, start_date="1986-03-14", end_date=None):
        self.storages = storages
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.today().strftime("%Y-%m-%d")
        self.acre_feet_to_mg = 0.325851
        self.gal_to_mg = 1 / 1000000
        self.parameter_code = "00062"
        self.nyc_parameter_code = "62615"
        self.statistic_codes = ["00003", "00002", "00001"]

    def retrieve_reservoir_data(self, gage_id, parameter_code):
        try:
            data = nwis.get_dv(
                sites=gage_id,
                parameterCd=parameter_code,
                statCd=self.statistic_codes,
                start=self.start_date,
                end=self.end_date,
            )
            df = data[0]
            df.reset_index(inplace=True)
            df["datetime"] = pd.to_datetime(df["datetime"]).dt.date
            return df
        except Exception as e:
            print(f"Error retrieving data for Gauge ID {gage_id}: {e}")
            return None

    def elevation_to_volume(self, elevation_data, storage_curve, reservoir_name):
        if "Elevation (ft)" not in storage_curve.columns:
            print(f"Invalid storage curve for {reservoir_name}. Expected columns not found.")
            elevation_data["volume_mg"] = np.nan
        else:
            storage_curve.set_index("Elevation (ft)", inplace=True)
            if reservoir_name in nyc_reservoirs:
                elevation_data["volume_mg"] = elevation_data["62615_Mean"].apply(
                    lambda elevation: np.interp(
                        elevation, storage_curve.index, storage_curve["Volume, gal"]
                    ) * self.gal_to_mg
                )
            else:
                elevation_data["volume_mg"] = elevation_data["00062_Mean"].apply(
                    lambda elevation: np.interp(
                        elevation, storage_curve.index, storage_curve["Acre-Ft"]
                    ) * self.acre_feet_to_mg
                )

        volume_time_series = elevation_data[["datetime", "volume_mg"]].copy()
        volume_time_series.set_index("datetime", inplace=True)
        volume_time_series.columns = [reservoir_name]
        return volume_time_series

    def process_all_reservoirs(self):
        all_volume_time_series = []

        for gage_id, reservoir_name, curve_filename in self.storages:
            parameter_code = self.nyc_parameter_code if reservoir_name in nyc_reservoirs else self.parameter_code
            daily_elevation = self.retrieve_reservoir_data(gage_id, parameter_code)

            if daily_elevation is not None and not daily_elevation.empty:
                raw_output_filename = os.path.join(raw_storage_dir, f"{reservoir_name}_elevation.csv")
                daily_elevation.to_csv(raw_output_filename, index=False)
                print(f"Data saved to '{raw_output_filename}' for Gauge ID {gage_id}.")

                if curve_filename and os.path.isfile(curve_filename):
                    elevation_curve = pd.read_csv(curve_filename)
                    volume_time_series = self.elevation_to_volume(daily_elevation, elevation_curve, reservoir_name)
                else:
                    print(f"Storage curve file for {reservoir_name} not found or not applicable.")
                    volume_time_series = pd.DataFrame(columns=[reservoir_name], index=pd.to_datetime(daily_elevation["datetime"]))
            else:
                print(f"No data found for Gauge ID {gage_id}.")
                volume_time_series = pd.DataFrame(columns=[reservoir_name])

            all_volume_time_series.append(volume_time_series)

        # Combine all retrieved volume time series into a single DataFrame
        if all_volume_time_series:
            final_volume_df = pd.concat(all_volume_time_series, axis=1, join="outer")
            final_volume_df.sort_index(inplace=True)

            # Save full NWIS dataset
            final_volume_df.to_csv(nwis_storage_file)
            print(f"Full NWIS storage data saved to '{nwis_storage_file}'.")

            # **Merge ODRM and NWIS storage data for NYC reservoirs**
            if os.path.exists(odrm_storage_file):
                odrm_data = pd.read_csv(odrm_storage_file, parse_dates=["datetime"], index_col="datetime")

                # NYC reservoirs: Use ODRM before 12/01/2021, NWIS after
                odrm_data_nyc = odrm_data.loc[:odrm_end_date, nyc_reservoirs]
                nwis_data_nyc = final_volume_df.loc[odrm_end_date:, nyc_reservoirs]

                final_volume_df.update(odrm_data_nyc)
                final_volume_df.loc[odrm_end_date:, nyc_reservoirs] = nwis_data_nyc

                print("Merged NYC observed storage data from ODRM and NWIS.")

            # Rename final columns to match internal references
            final_volume_df = final_volume_df[["datetime"] + nyc_reservoirs + lower_basin_reservoirs]

            # Save final processed storage data
            processed_csv_filename = os.path.join(processed_storage_dir, "reservoir_storage_combined_mg.csv")
            final_volume_df.to_csv(processed_csv_filename)
            print(f"Final combined NYC + Lower Basin storage data saved to '{processed_csv_filename}'.")
        else:
            print("No volume data to save.")

if __name__ == "__main__":
    # Define storage reservoirs (gage ID, reservoir name, storage curve file)
    storages = [
        ("01435000", "prompton", "prompton_storage_curve.csv"),
        ("01449790", "beltzvilleCombined", "beltzvilleCombined_storage_curve.csv"),
        ("01447780", "fewalter", "fewalter_storage_curve.csv"),
        ("01452500", "blueMarsh", "bluemarsh_storage_curve.csv"),
        ("01436000", "neversink", "neversink_storage_curve.csv"),
        ("01417000", "cannonsville", "cannonsville_storage_curve.csv"),
        ("01415000", "pepacton", "pepacton_storage_curve.csv"),
    ]

    # Instantiate the class
    processor = DataPreprocessor(storages)

    # Run the processing function
    print("Starting data retrieval and processing...")
    processor.process_all_reservoirs()
    print("Processing complete.")
# ##############################################################################
# ##############################################################################
