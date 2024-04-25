"""
Ensemble data is typically going to be formatted in a nested dictionary. 

The dictionary will be structured as:
dict[f'realization_{i}'][variable] = pd.DataFrame

This script contains functions used for processing ensemble data of this format. 
"""

import pandas as pd
import numpy as np

def ensemble_mean(ensemble):
    """
    Calculates the mean of an ensemble of data.
    
    Args:
    ensemble (dict): Dictionary containing the ensemble data.
    
    Returns:
    mean (pd.DataFrame): Mean of the ensemble data.
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
    Calculates the standard deviation of an ensemble of data.
    
    Args:
    ensemble (dict): Dictionary containing the ensemble data.
    
    Returns:
    std (pd.DataFrame): Standard deviation of the ensemble data.
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
    Calculates the median of an ensemble of data.
    
    Args:
    ensemble (dict): Dictionary containing the ensemble data.
    
    Returns:
    median (pd.DataFrame): Median of the ensemble data.
    """
    
    realizations = list(ensemble.keys())
    for i, realization in enumerate(realizations):
        if i == 0:
            median = ensemble[realization].copy()
        else:
            median = median.append(ensemble[realization])
            
    median = median.groupby(median.index).median()
    return median