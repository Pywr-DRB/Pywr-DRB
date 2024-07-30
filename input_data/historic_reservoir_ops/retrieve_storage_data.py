# ##############################################################################
# #Marilyn Smith 
# #Script to retrieve reservoir data from NWIS and convert elevation to volume
# #Data is saved to CSV files


import pandas as pd
import numpy as np
from dataretrieval import nwis
import os
import sys
path_to_pywrdrb = '../../'
sys.path.append(path_to_pywrdrb)
from pywrdrb.utils.directories import input_dir

# Define storage sites with gauge IDs
storages = [
    ("01428900", "prompton", f'{input_dir}/historic_reservoir_ops/prompton_elevation_storage_curve.csv'),
    ("01449790", "beltzvilleCombined", f'{input_dir}/historic_reservoir_ops/beltzvilleCombined_elevation_storage_curve.csv'),
    ("01447780", "fewalter", f'{input_dir}/historic_reservoir_ops/fewalter_elevation_storage_curve.csv'),
    ("01470870", "blueMarsh", f'{input_dir}/historic_reservoir_ops/blueMarsh_elevation_storage_curve.csv')
]

print("Retrieving data from NWIS...")

parameterCode = "00062"  # Elevation of reservoir water surface above datum
statisticCodes = ["00003", "00002", "00001"]  # Mean, Minimum, Maximum

ACRE_FEET_TO_MG = 0.325851  # Conversion factor from Acre-Feet to Million Gallons

def retrieve_reservoir_data(gage_id, parameter_code, statistic_codes, start_date='1986-03-14', end_date='2021-09-30'):
    try:
        data = nwis.get_dv(sites=gage_id, parameterCd=parameter_code, statCd=statistic_codes, start=start_date, end=end_date)
        df = data[0]
        df.reset_index(inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime']).dt.date
        return df
    except Exception as e:
        print(f"Error retrieving data for Gauge ID {gage_id}: {e}")
        return None

def elevation_to_volume(elevation_data, storage_curve, reservoir_name):
    if 'Elevation (ft)' not in storage_curve.columns or 'Acre-Ft' not in storage_curve.columns:
        print(f"Invalid storage curve for {reservoir_name}. Expected columns not found.")
        elevation_data['volume_MG'] = np.nan
        volume_time_series = elevation_data[['datetime', 'volume_MG']].copy()
        volume_time_series.set_index('datetime', inplace=True)
        volume_time_series.columns = [reservoir_name]
        return volume_time_series

    storage_curve.set_index('Elevation (ft)', inplace=True)

    elevation_data['volume_MG'] = elevation_data['00062_Mean'].apply(
        lambda elevation: np.interp(elevation, storage_curve.index, storage_curve['Acre-Ft']) * ACRE_FEET_TO_MG
    )

    volume_time_series = elevation_data[['datetime', 'volume_MG']].copy()
    volume_time_series.set_index('datetime', inplace=True)
    volume_time_series.columns = [reservoir_name]

    return volume_time_series

all_volume_time_series = []

for gage_id, reservoir_name, curve_filename in storages:
    daily_elevation = retrieve_reservoir_data(gage_id, parameterCode, statisticCodes)

    if daily_elevation is not None and not daily_elevation.empty:
        elevation_csv_filename = f'{reservoir_name}_{gage_id}_elevation.csv'
        daily_elevation.to_csv(elevation_csv_filename, index=False)
        print(f"Data saved to '{elevation_csv_filename}' for Gauge ID {gage_id}.")

        if curve_filename and os.path.isfile(curve_filename):
            elevation_curve = pd.read_csv(curve_filename)

            volume_time_series = elevation_to_volume(daily_elevation, elevation_curve, reservoir_name)
        else:
            print(f"Storage curve file for {reservoir_name} not found or not applicable.")
            empty_df = pd.DataFrame(columns=[reservoir_name])
            empty_df.index.name = 'datetime'
            volume_time_series = empty_df
    else:
        print(f"No data found for Gauge ID {gage_id}.")
        empty_df = pd.DataFrame(columns=[reservoir_name])
        empty_df.index.name = 'datetime'
        volume_time_series = empty_df

    all_volume_time_series.append(volume_time_series)

if all_volume_time_series:
    final_volume_df = pd.concat(all_volume_time_series, axis=1, join='outer')

    final_volume_df.sort_index(inplace=True)  # Sort the index to ensure consecutive datetime

    final_csv_filename = 'observed_storage_data.csv'
    final_csv_filename = f'{input_dir}/historic_reservoir_ops/{final_csv_filename}'
    final_volume_df.to_csv(final_csv_filename)
    print(f"Combined volume data saved to '{final_csv_filename}'.")
else:
    print("No volume data to save.")


