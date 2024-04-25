"""Functions useful for working with reservoir nodes and data.
"""

import numpy as np
import pandas as pd

from .directories import model_data_dir, input_dir
from .lists import reservoir_list_nyc

istarf = pd.read_csv(f'{model_data_dir}drb_model_istarf_conus.csv')
def get_reservoir_capacity(reservoir):
    return float(istarf['Adjusted_CAP_MG'].loc[istarf['reservoir'] == reservoir].iloc[0])


def get_nyc_total_capacity():
    nyc_capacitites = [get_reservoir_capacity(reservoir) for reservoir in reservoir_list_nyc]
    return sum(nyc_capacitites)


def load_lower_basin_elevations():
    lower_elevations= pd.read_csv(f'{input_dir}historic_reservoir_ops/lower_basin_reservoir_elevation.csv', 
                            index_col=0, parse_dates=True)
    lower_elevations.index = pd.to_datetime(lower_elevations.index)
    return lower_elevations


def load_lower_basin_storage_curves():
    storage_curves = {}
    for res in ['blueMarsh', 'beltzvilleCombined', 'fewalter']:
        storage_curves[res]= pd.read_csv(f'{input_dir}historic_reservoir_ops/{res}_elevation_storage_curve.csv', sep=',')
    return storage_curves



def load_historic_nyc_storage():
    nyc_hist_storage = pd.read_csv(f'{input_dir}/historic_NYC/NYC_storage_daily_2000-2021.csv', sep=',', index_col=0, parse_dates=True)
    nyc_hist_storage.index = pd.to_datetime(nyc_hist_storage.index)

    # Get total storage
    nyc_hist_storage['NYCTotal'] = nyc_hist_storage.loc[:, reservoir_list_nyc].sum(axis=1)
    return nyc_hist_storage    