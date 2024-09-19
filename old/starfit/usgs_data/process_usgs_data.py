"""
Trevor Amestoy

Pulls reservoir data from the ResOpsUS database, using IDs in the
DRB reservoir info spreadsheet.
"""

import numpy as np
import pandas as pd
import urllib.request
import os


################################################################################
# Sourcing straight from USGS

# Adapted from Andrew's get_usgs_inflow.py
# Adopted to get BlueMarsh and Beltzville downstream records
###############################################################################

# Gages of interest (downstream of reservoirs)
# Blue marsh: '01470960'

gages = ["01470960", "01449800"]


def download_usgs_gage_data(gages):
    for gage in gages:
        try:
            start_date = "2005-10-01"
            end_date = "2012-10-01"
            url = f"https://nwis.waterdata.usgs.gov/usa/nwis/uv/?cb_00060=on&format=rdb&site_no={gage}&period=&begin_date={start_date}&end_date={end_date}"
            filename = f"usgs_{gage}.txt"
            urllib.request.urlretrieve(url, filename)
        except:
            print("DOWNLOAD FAIL: GAGE " + gage)
    return


def get_usgs_data(gages, daily_average=True):
    """
    gages : list
        gage ids (strings) in a list.
    """

    # Source initial .txt data
    download_usgs_gage_data(gages)

    # clean txt and export to csv
    for gage in gages:
        filename = f"usgs_{gage}.txt"

        data = pd.read_csv(filename, sep="\t", header=None, skiprows=29)
        data.columns = ["agency", "gage_id", "date-time", "timezone", "flow", "unknown"]
        data = data.drop(["agency", "timezone", "unknown"], axis=1)

        # Separate out time from date-time
        data["date"] = pd.to_datetime(data["date-time"]).dt.date
        data = data.drop(["date-time"], axis=1)

        if daily_average:
            data = data.groupby("date").mean()
            data = data.reset_index()

        # Export
        output_file = f"clean_usgs_{gage}.csv"
        data.to_csv(output_file, index=False)

    return


if __name__ == "__main__":
    get_usgs_data(gages)
