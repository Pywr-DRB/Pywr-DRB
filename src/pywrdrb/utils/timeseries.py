"""
Functions for working directly with pd.Series timeseries.

Overview: 
These functions are broadly applicable across the pywrdrb codebase. 


Technical Notes: 
- NA

Links: 
- NA
 
Change Log:
TJA, 2025-05-05, Add docs.
"""

import pandas as pd
import datetime as dt


def subset_timeseries(timeseries, start_date, end_date, end_inclusive=True):
    """
    Take a subset of pd.Series timeseries data between start_date and end_date.

    Parameters
    ----------
    timeseries : pd.Series
        The timeseries data to subset.
    start_date : str or pd.Timestamp
        The start date for the subset. If str, should be in 'YYYY-MM-DD' format.
    end_date : str or pd.Timestamp
        The end date for the subset. If str, should be in 'YYYY-MM-DD' format.
    end_inclusive : bool, optional
        Whether to include the end date in the subset. Default is True.
    
    Returns
    -------
    pd.Series
        The subset of the timeseries data between start_date and end_date.
    """

    data = timeseries.copy()
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    if not end_inclusive:
        end_date = end_date - dt.timedelta(days=1)

    if start_date is not None:
        data = data.loc[start_date:]
    if end_date is not None:
        data = data.loc[:end_date]
    return data


def get_rollmean_timeseries(timeseries, window):
    """
    Calculates the rolling mean of a timeseries for a given window size.

    Parameters
    ----------
    timeseries : pd.Series
            The timeseries data to calculate the rolling mean for.
    window : int
            The window size for the rolling mean.
    
    Returns
    -------
    pd.Series
        The rolling mean transformed timeseries data.
    """

    try:
        datetime = timeseries["datetime"]
        timeseries.drop("datetime", axis=1, inplace=True)
    except:
        pass
    rollmean_timeseries = timeseries.rolling(window=window).mean()
    rollmean_timeseries.iloc[:window] = [
        timeseries.rolling(window=i + 1).mean().iloc[i] for i in range(window)
    ]
    try:
        rollmean_timeseries["datetime"] = datetime
    except:
        pass
    return rollmean_timeseries
