"""Functions useful for working with reservoir nodes and data.
"""

import numpy as np
import pandas as pd

from pywrdrb.utils.lists import reservoir_list_nyc
from pywrdrb.path_manager import get_pn_object

pn = get_pn_object()
fname = pn.operational_constants.get_str("istarf_conus.csv")
istarf = pd.read_csv(fname)


def get_reservoir_capacity(reservoir):
    return float(
        istarf["Adjusted_CAP_MG"].loc[istarf["reservoir"] == reservoir].iloc[0]
    )


def get_nyc_total_capacity():
    nyc_capacitites = [
        get_reservoir_capacity(reservoir) for reservoir in reservoir_list_nyc
    ]
    return sum(nyc_capacitites)
