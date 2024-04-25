"""
TJA
"""

import numpy as np

def transform_intraannual(Y, timestep = 'monthly'):
    """
    Transforms matrix Y into Y' according to methods described in Kirsch et al. (2013)

    Parameters
    ----------
    Y : matrix (n_year x n_time)
        Matrix to be transformed.

    Returns
    -------
    Y_prime : matrix (n_year - 1 x 52 weeks)

    """

    # dimensions
    N, n_col = Y.shape

    # Reorganize Y into Y_prime to preserve interannual correlation
    Y_prime = np.zeros((N-1, n_col))
    flat_Y = Y.flatten()

    if timestep == 'monthly':
        period = 12
        half_period = 6
    elif timestep == 'weekly':
        period = 52
        half_period = 26

    for k in range(1,int(len(flat_Y)/period)):
        Y_prime[(k-1),0:half_period] = flat_Y[(k * half_period) : ((k + 1)*half_period)]
        Y_prime[(k-1),half_period:period] = flat_Y[((k+1) * half_period) : ((k + 2)*half_period)]

    return Y_prime

def transform_daily_df_to_monthly_ndarray(df):

    # Initialize the output array
    n_days, n_sites = df.shape
    n_years = df.index[-1].year - df.index[0].year + 1
    out = np.zeros((n_sites, n_years, 12))

    # Loop over sites and fill in output array
    for i, site in enumerate(df.columns):
        site_data = df[site].values

        # Loop over years
        for j in range(n_years):
            year_start = j * 365
            year_end = min((j + 1) * 365, n_days)

            # Sum daily data into monthly data
            monthly_data = np.zeros(12)
            for k in range(year_start, year_end):
                date = df.index[k]
                monthly_data[date.month - 1] += site_data[k]

            out[i, j, :] = monthly_data
    return out


###############################################################################


def transform_timeseries_df_to_ndarray(df):

    # Initialize the output array
    n_days, n_sites = df.shape
    n_years = df.index[-1].year - df.index[0].year + 1
    out = np.zeros((n_sites, n_years, 365))

    # Loop over sites and fill in output array
    for i, site in enumerate(df.columns):
        site_data = df[site].values

        # Loop over years
        for j in range(n_years):
            year_start = j * 365
            year_end = min((j + 1) * 365, n_days)
            
            # Fill in output array for this year
            year_data = np.zeros(365)
            for k in range(year_start, year_end):
                year_day = k % 365
                year_data[year_day] = site_data[k]

                # If this is Feb 29 in a leap year, fill in next day with same value
                if year_day == 59 and (j * 365 + k) % 1461 == 1460:
                    year_data[60] = site_data[k]
            out[i, j, :] = year_data
    return out