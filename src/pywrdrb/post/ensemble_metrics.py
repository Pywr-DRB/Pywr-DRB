"""
Functions for processing ensemble output data.

Overview
--------
This module provides basic utility functions to compute statistics (mean, median, standard deviation)
from ensemble datasets structured as nested dictionaries, where each realization contains one or 
more pandas DataFrames.

The expected structure is:
    ensemble = {
        'realization_0': pd.DataFrame,
        'realization_1': pd.DataFrame,
        ...
    }

These functions are commonly used for analyzing uncertainty across ensemble simulations,
such as those generated in stochastic reservoir modeling or policy evaluations.

Key Steps
---------
1. Loop through all realizations in the ensemble dictionary.
2. Convert values to float and handle NaNs where appropriate.
3. Aggregate across ensemble members using mean, standard deviation, or median.

Technical Notes
---------------
- Input data must be time-aligned across realizations (identical index).
- Missing values (NaNs) are filled with zeros before computing the ensemble mean.
- Assumes all realizations are formatted as pandas DataFrames of equal structure.
- Median is computed row-wise across ensemble members.

Links
-----
- https://github.com/Pywr-DRB/Pywr-DRB

Change Log
----------
Marilyn Smith, 2025-05-07, Initial module documentation and cleanup.
"""
import numpy as np


def ensemble_mean(ensemble):
    """
    Calculate the mean across ensemble realizations.

    Parameters
    ----------
    ensemble : dict of {str: pd.DataFrame}
        Dictionary of ensemble realizations. Each key corresponds to a realization
        and maps to a DataFrame of time series data.

    Returns
    -------
    pd.DataFrame
        DataFrame of mean values across realizations, indexed by time.

    Notes
    -----
    Missing values (NaNs) are replaced with zeros prior to averaging.
    Assumes all DataFrames are aligned in index and column structure.
    """

    realizations = list(ensemble.keys())
    for i, realization in enumerate(realizations):
        ensemble[realization] = ensemble[realization].astype(float)
        ensemble[realization] = ensemble[realization].fillna(0)

        if i == 0:
            mean = ensemble[realization].copy()
        else:
            mean += ensemble[realization]

    mean = mean / len(realizations)
    return mean


def ensemble_std(ensemble):
    """
    Calculate the standard deviation across ensemble realizations.

    Parameters
    ----------
    ensemble : dict of {str: pd.DataFrame}
        Dictionary of ensemble realizations. Each key corresponds to a realization
        and maps to a DataFrame of time series data.

    Returns
    -------
    pd.DataFrame
        DataFrame of standard deviation values across realizations, indexed by time.

    Notes
    -----
    Standard deviation is computed using the unbiased estimator (N-1 in denominator).
    Assumes all realizations are aligned in index and column structure.
    """

    realizations = list(ensemble.keys())
    for i, realization in enumerate(realizations):
        if i == 0:
            std = ensemble[realization].copy()
        else:
            std += (ensemble[realization] - ensemble_mean(ensemble)) ** 2

    std = np.sqrt(std / (len(realizations) - 1))
    return std


def ensemble_median(ensemble):
    """
    Calculate the median across ensemble realizations.

    Parameters
    ----------
    ensemble : dict of {str: pd.DataFrame}
        Dictionary of ensemble realizations. Each key corresponds to a realization
        and maps to a DataFrame of time series data.

    Returns
    -------
    pd.DataFrame
        DataFrame of median values across realizations, indexed by time.

    Notes
    -----
    Appends all realizations row-wise and then computes the median group-wise
    by index (time). This assumes all DataFrames are aligned in index.
    """

    realizations = list(ensemble.keys())
    for i, realization in enumerate(realizations):
        if i == 0:
            median = ensemble[realization].copy()
        else:
            median = median.append(ensemble[realization])

    median = median.groupby(median.index).median()
    return median
