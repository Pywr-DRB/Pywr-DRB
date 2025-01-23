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

path_to_pywrdrb = "../../"
sys.path.append(path_to_pywrdrb)
from pywrdrb.utils.directories import input_dir

# Define storage sites with gauge IDs
storages = [
    (
        "01428900",
        "prompton",
        f"{input_dir}/historic_reservoir_ops/prompton_elevation_storage_curve.csv",
    ),
    (
        "01449790",
        "beltzvilleCombined",
        f"{input_dir}/historic_reservoir_ops/beltzvilleCombined_elevation_storage_curve.csv",
    ),
    (
        "01447780",
        "fewalter",
        f"{input_dir}/historic_reservoir_ops/fewalter_elevation_storage_curve.csv",
    ),
    (
        "01470870",
        "blueMarsh",
        f"{input_dir}/historic_reservoir_ops/blueMarsh_elevation_storage_curve.csv",
    ),
    (
        "01435900",
        "neversink",
        f"{input_dir}/historic_reservoir_ops/neversink_elevation_storage_curve.csv",
    ),
    (
        "01423910",
        "cannonsville",
        f"{input_dir}/historic_reservoir_ops/cannonsville_elevation_storage_curve.csv",
    ),
    (
        "01414750",
        "pepacton",
        f"{input_dir}/historic_reservoir_ops/pepacton_elevation_storage_curve.csv",
    ),

]

print("Retrieving data from NWIS...")

parameterCode = "00062"  # Elevation of reservoir water surface above datum
nyc_parameterCode = '62615'

statisticCodes = ["00003", "00002", "00001"]  # Mean, Minimum, Maximum

ACRE_FEET_TO_MG = 0.325851  # Conversion factor from Acre-Feet to Million Gallons

GAL_TO_MG = 1 / 1000000  # Conversion factor from gallons to million gallons



def retrieve_reservoir_data(
    gage_id,
    parameter_code,
    statistic_codes,
    start_date="1986-03-14",
    end_date="2024-12-31",
):
    try:
        data = nwis.get_dv(
            sites=gage_id,
            parameterCd=parameter_code,
            statCd=statistic_codes,
            start=start_date,
            end=end_date,
        )
        df = data[0]
        df.reset_index(inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.date
        return df
    except Exception as e:
        print(f"Error retrieving data for Gauge ID {gage_id}: {e}")
        return None


def elevation_to_volume(elevation_data, storage_curve, reservoir_name):
    if (
        "Elevation (ft)" not in storage_curve.columns
        #or "Acre-Ft" not in storage_curve.columns
    ):
        print(
            f"Invalid storage curve for {reservoir_name}. Expected columns not found."
        )
        elevation_data["volume_MG"] = np.nan
        volume_time_series = elevation_data[["datetime", "volume_MG"]].copy()
        volume_time_series.set_index("datetime", inplace=True)
        volume_time_series.columns = [reservoir_name]
        return volume_time_series

    storage_curve.set_index("Elevation (ft)", inplace=True)

    if reservoir_name in ["cannonsville", "neversink", "pepacton"]:
        #use the 'Volume, gal' column instead of Acre-Ft and convert to MG
        elevation_data["volume_MG"] = elevation_data["62615_Mean"].apply(
        lambda elevation: np.interp(
            elevation, storage_curve.index, storage_curve["Volume, gal"]
        )
        * GAL_TO_MG
    )
    else:

        elevation_data["volume_MG"] = elevation_data["00062_Mean"].apply(
        lambda elevation: np.interp(
            elevation, storage_curve.index, storage_curve["Acre-Ft"]
        )
        * ACRE_FEET_TO_MG
    )

    volume_time_series = elevation_data[["datetime", "volume_MG"]].copy()
    volume_time_series.set_index("datetime", inplace=True)
    volume_time_series.columns = [reservoir_name]

    return volume_time_series


all_volume_time_series = []

for gage_id, reservoir_name, curve_filename in storages:
    #if its cannansville, neversink, or pepacton, use the NYC parameter code
    if reservoir_name in ["cannonsville", "neversink", "pepacton"]:
        parameterCode = nyc_parameterCode
    daily_elevation = retrieve_reservoir_data(gage_id, parameterCode, statisticCodes)

    if daily_elevation is not None and not daily_elevation.empty:
        elevation_csv_filename = f"{reservoir_name}_{gage_id}_elevation.csv"
        daily_elevation.to_csv(elevation_csv_filename, index=False)
        print(f"Data saved to '{elevation_csv_filename}' for Gauge ID {gage_id}.")

        if curve_filename and os.path.isfile(curve_filename):
            elevation_curve = pd.read_csv(curve_filename)

            volume_time_series = elevation_to_volume(
                daily_elevation, elevation_curve, reservoir_name
            )
        else:
            print(
                f"Storage curve file for {reservoir_name} not found or not applicable."
            )
            empty_df = pd.DataFrame(columns=[reservoir_name])
            empty_df.index.name = "datetime"
            volume_time_series = empty_df
    else:
        print(f"No data found for Gauge ID {gage_id}.")
        empty_df = pd.DataFrame(columns=[reservoir_name])
        empty_df.index.name = "datetime"
        volume_time_series = empty_df

    all_volume_time_series.append(volume_time_series)

if all_volume_time_series:
    final_volume_df = pd.concat(all_volume_time_series, axis=1, join="outer")

    final_volume_df.sort_index(
        inplace=True
    )  # Sort the index to ensure consecutive datetime

    final_csv_filename = "observed_storage_data.csv"
    final_csv_filename = f"{input_dir}/historic_reservoir_ops/{final_csv_filename}"
    final_volume_df.to_csv(final_csv_filename)
    print(f"Combined volume data saved to '{final_csv_filename}'.")
else:
    print("No volume data to save.")
