import pandas as pd
import datetime as dt


def subset_timeseries(timeseries, start_date, end_date, end_inclusive=True):
    """
    Take a subset of pd.DataFrame timeseries data between start_date and end_date.
    
    Args:
        timeseries (pd.DataFrame): The timeseries data.
        start_date (str or pd.Timestamp): The start date for the subset.
        end_date (str or pd.Timestamp): The end date for the subset.
        end_inclusive (bool): Whether the end date is inclusive.
    
    Returns:
        data (pd.DataFrame): The subset of the timeseries data.
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
    
    Args:
        timeseries (pd.DataFrame): The timeseries data.
        window (int): The window size for the rolling mean.
        
    Returns:
        rollmean_timeseries (pd.DataFrame): The rolling mean timeseries.
    """
    
    try:
        datetime = timeseries['datetime']
        timeseries.drop('datetime', axis=1, inplace=True)
    except:
        pass
    rollmean_timeseries = timeseries.rolling(window=window).mean()
    rollmean_timeseries.iloc[:window] = [timeseries.rolling(window=i + 1).mean().iloc[i] for i in range(window)]
    try:
        rollmean_timeseries['datetime'] = datetime
    except:
        pass
    return rollmean_timeseries