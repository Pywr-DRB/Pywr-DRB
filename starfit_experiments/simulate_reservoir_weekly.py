"""
Trevor Amestoy
Summer 2022

Simulates reservoir storage and releases, using STARFIT inferred operating
rule parameters.

Operates at a weekly time step (Use the new *_daily version)
"""
import numpy as np
import pandas as pd
from math import pi, sin, cos

def sim_starfit_reservoir_weekly(starfit_df, reservoir_name, inflow, S_initial):
    """
    Parameters:
    ----------
    starfit_df : DataFrame
        A dataframe containing all starfit data for reservoirs in the basin.
    reservoir : str
        The name of the reservoir to be simulated.
    inflow : array [1 x 52-weeks]
        An array containing a timeseries of inflow values into the reservoir;
        weekly time-step.
    S_initial : float
        The initial storage in the reservoir.

    Returns:
    storage : array
        An array of weekly reservoir storage volumes.
    releases : array
        An array of weekly reservoir release volumes.
    """

    # Find the index of the desired reservoir
    res_index = starfit_df.index[starfit_df['reservoir'] == reservoir_name].tolist()

    # Check that reservoir is contained in the starfit_df
    if not res_index:
        print('reservoir_name was not found in starfit_df.\n Check the reservoir_name and try again.\n')
        return

    # Source all starfit data for reservoir of interest in dictionary
    data = starfit_df.iloc[res_index]

    # Define reservoir constant characteristics WEEKLY
    R_max = ((data['Release_max'] + 1) * data['GRanD_MEANFLOW_MGD'] * 7).values
    R_min = ((data['Release_min'] + 1) * data['GRanD_MEANFLOW_MGD'] * 7).values
    I_bar = data['GRanD_MEANFLOW_MGD'].values * 7
    S_cap = data['GRanD_CAP_MG'].values

    # Define the average weekly release function
    def release_harmonic(week):
        R_avg_t = (data['Release_alpha1'] * sin(2 * pi * week/52) +
                 data['Release_alpha2'] * sin(4 * pi * week/52) +
                 data['Release_beta1'] * cos(2 * pi * week/52) +
                 data['Release_beta2'] * cos(4 * pi * week/52))
        return R_avg_t.values[0]

    # Calculate weekly values of the upper NOR bound
    def NOR_weekly_hi(week):
        NOR_hi = (data['NORhi_mu'] + data['NORhi_alpha'] * sin(2*pi*week/52) +
                     data['NORhi_beta'] * cos(2*pi*week/52))

        if (NOR_hi < data['NORhi_min']).bool():
            NOR_hi = data['NORhi_min']
        elif (NOR_hi > data['NORhi_max']).bool():
            NOR_hi = data['NORhi_max']
        return NOR_hi.values

    # Calculate weekly values of the lower NOR bound
    def NOR_weekly_lo(week):
        NOR_lo = (data['NORlo_mu'] + data['NORlo_alpha'] * sin(2*pi*week/52) +
                     data['NORlo_beta'] * cos(2*pi*week/52))

        if (NOR_lo < data['NORlo_min']).bool():
            NOR_lo = data['NORlo_min']
        elif (NOR_lo > data['NORlo_max']).bool():
            NOR_lo = data['NORlo_max']
        return NOR_lo.values

    # Standardize inflow using annual average
    def standardize_inflow(I_t, week):
        return (I_t - I_bar) / I_bar

    # Calculate storage as % of S_cap
    def percent_storage(S_t):
        return (S_t / S_cap)*100

    # Define the weekly release adjustement function
    def release_adjustment(S_hat, week):
        A_t = (S_hat - NOR_weekly_lo(week)) / (NOR_weekly_hi(week) - NOR_weekly_lo(week))
        I_hat = standardize_inflow(inflow[week], week)

        epsilon = (data['Release_c'] + data['Release_p1']*A_t +
                   data['Release_p2']*I_hat)
        return epsilon.values

    # Calculate the conditional target release volume
    def target_release(S_hat, week):
        NOR_hi = NOR_weekly_hi(week)
        NOR_lo = NOR_weekly_lo(week)

        if (S_hat <= NOR_hi) and (S_hat >= NOR_lo):
            target_R = min(I_bar * (release_harmonic(week) +
                                    release_adjustment(S_hat, week))
                           + I_bar, R_max)
            print('in NOR')
        elif (S_hat > NOR_hi):
            target_R = min(S_cap * (S_hat - NOR_hi) + inflow[week], R_max)
            print('above NOR')
        elif (S_hat < NOR_lo):
            target_R = R_min
            print('below NOR')
        return target_R


    # Calculate actual release subject to mass constraints
    def actual_release(target_R, I_t, S_t):
        return max(min(target_R, (I_t + S_t)), (I_t + S_t - S_cap))

    # Initialize matrices
    S = np.zeros_like(inflow)
    S_hat = np.zeros_like(S)
    R = np.zeros_like(inflow)

    # Set initial storage
    S[0] = S_initial
    S_hat[0] = percent_storage(S[0])

    # Simulate at weekly step
    for wk in range(len(inflow) - 1):

        I = inflow[wk]
        S_hat[wk] = percent_storage(S[wk])
        target_R = target_release(S_hat[wk], wk)
        R[wk] = actual_release(target_R, I, S[wk])

        S[wk + 1] = S[wk] + I - R[wk]

    return S, S_hat, R

################################################################################
# Define release, and NOR harmonics for independent use
################################################################################

# Define the average weekly release function
def release_harmonic(reservoir_data, week):
    R_avg_t = (reservoir_data['Release_alpha1'] * sin(2 * pi * week/52) +
             reservoir_data['Release_alpha2'] * sin(4 * pi * week/52) +
             reservoir_data['Release_beta1'] * cos(2 * pi * week/52) +
             reservoir_data['Release_beta2'] * cos(4 * pi * week/52))
    return R_avg_t

# Calculate weekly values of the upper NOR bound
def NOR_weekly_hi(reservoir_data, week):
    NOR_hi = (reservoir_data['NORhi_mu'] + reservoir_data['NORhi_alpha'] * sin(2*pi*week/52) +
                 reservoir_data['NORhi_beta'] * cos(2*pi*week/52))

    if (NOR_hi < reservoir_data['NORhi_min']):
        NOR_hi = reservoir_data['NORhi_min']
    elif (NOR_hi > reservoir_data['NORhi_max']):
        NOR_hi = reservoir_data['NORhi_max']
    return NOR_hi

# Calculate weekly values of the lower NOR bound
def NOR_weekly_lo(reservoir_data, week):
    NOR_lo = (reservoir_data['NORlo_mu'] + reservoir_data['NORlo_alpha'] * sin(2*pi*week/52) +
                 reservoir_data['NORlo_beta'] * cos(2*pi*week/52))

    if (NOR_lo < reservoir_data['NORlo_min']):
        NOR_lo = reservoir_data['NORlo_min']
    elif (NOR_lo > reservoir_data['NORlo_max']):
        NOR_lo = reservoir_data['NORlo_max']
    return NOR_lo
