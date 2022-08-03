"""
Trevor Amestoy
Summer 2022

Simulates reservoir storage and releases, using STARFIT inferred operating
rule parameters.

Operates at a daily time step
"""
import numpy as np
import pandas as pd
from math import pi, sin, cos


def starfit_reservoir_simulation(starfit_df, reservoir_name, inflow, S_initial):
    """
    Simulates reservoir storage and release using STARFIT parameters.
    NOTE: Data must begin on the Oct-1 (start of water year).

    Parameters:
    ----------
    starfit_df : DataFrame
        A dataframe containing all starfit data for reservoirs in the basin.
    reservoir : str
        The name of the reservoir to be simulated.
    inflow : array [1 x 365-days]
        An array containing a timeseries of inflow values into the reservoir;
        daily time-step.
    S_initial : float
        The initial storage in the reservoir.

    Returns:
    storage : array
        An array of daily reservoir storage volumes.
    releases : array
        An array of daily reservoir release volumes.
    """

    # Find the index of the desired reservoir
    res_index = starfit_df.index[starfit_df['reservoir'] == reservoir_name].tolist()

    # Check that reservoir is contained in the starfit_df
    if not res_index:
        print('reservoir_name was not found in starfit_df.\n Check the reservoir_name and try again.\n')
        return

    # Source all starfit data for reservoir of interest in dictionary
    data = starfit_df.iloc[res_index]

    # Define reservoir constant characteristics daily
    R_max = ((data['Release_max'] + 1) * data['GRanD_MEANFLOW_MGD']).values
    R_min = ((data['Release_min'] + 1) * data['GRanD_MEANFLOW_MGD']).values
    I_bar = data['GRanD_MEANFLOW_MGD'].values
    S_cap = data['GRanD_CAP_MG'].values

    # Define the average daily release function
    def release_harmonic(time, timestep = 'daily'):
        if timestep == 'daily':
            time = time/7
        R_avg_t = (data['Release_alpha1'] * sin(2 * pi * (time + 39)/52) +
                 data['Release_alpha2'] * sin(4 * pi * (time + 39)/52) +
                 data['Release_beta1'] * cos(2 * pi * (time + 39)/52) +
                 data['Release_beta2'] * cos(4 * pi * (time + 39)/52))
        return R_avg_t.values[0]

    # Calculate daily values of the upper NOR bound
    def calc_NOR_hi(time, timestep = 'daily'):
        # NOR harmonic is at weekly step
        if timestep == 'daily':
            time = time/7

        NOR_hi = (data['NORhi_mu'] + data['NORhi_alpha'] * sin(2*pi*(time + 39)/52) +
                     data['NORhi_beta'] * cos(2*pi*(time + 39)/52))

        if (NOR_hi < data['NORhi_min']).bool():
            NOR_hi = data['NORhi_min']
        elif (NOR_hi > data['NORhi_max']).bool():
            NOR_hi = data['NORhi_max']
        return (NOR_hi.values/100)

    # Calculate daily values of the lower NOR bound
    def calc_NOR_lo(time, timestep = 'daily'):
        # NOR harmonic is at weekly step
        if timestep == 'daily':
            time = time/7

        NOR_lo = (data['NORlo_mu'] + data['NORlo_alpha'] * sin(2*pi*(time + 39)/52) +
                     data['NORlo_beta'] * cos(2*pi*(time + 39)/52))

        if (NOR_lo < data['NORlo_min']).bool():
            NOR_lo = data['NORlo_min']
        elif (NOR_lo > data['NORlo_max']).bool():
            NOR_lo = data['NORlo_max']
        return (NOR_lo.values/100)

    # Standardize inflow using annual average
    def standardize_inflow(I_t):
        return (I_t - I_bar) / I_bar

    # Calculate storage as % of S_cap
    def percent_storage(S_t):
        return (S_t / S_cap)

    # Define the daily release adjustement function
    def release_adjustment(S_hat, time, timestep = 'daily'):
        A_t = (S_hat - calc_NOR_lo(time, timestep = timestep)) / (calc_NOR_hi(time, timestep = timestep) - calc_NOR_lo(time, timestep = timestep))
        I_hat = standardize_inflow(inflow[time])

        epsilon = (data['Release_c'] + data['Release_p1']*A_t +
                   data['Release_p2']*I_hat)
        return epsilon.values

    # Calculate the conditional target release volume
    def target_release(S_hat, I_t, time, R_previous):
        NOR_hi = calc_NOR_hi(time)
        NOR_lo = calc_NOR_lo(time)

        if (S_hat <= NOR_hi) and (S_hat >= NOR_lo):
            target_R = min(I_bar * (release_harmonic(time) +
                                    release_adjustment(S_hat, time))
                           + I_bar, R_max)
        elif (S_hat > NOR_hi):
            target_R = min(S_cap * (S_hat - NOR_hi) + I_t, R_max)
        else:
            #target_R = R_min

            # EXPERIMENT
            tR = (I_bar * (release_harmonic(time) + release_adjustment(S_hat, time)) + I_bar) * (1 - (NOR_lo - S_hat)/NOR_lo)
            target_R = max(tR, R_min)
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

    # Simulate at daily step
    for d in range(len(inflow) - 1):

        I = inflow[d]
        S_hat[d] = percent_storage(S[d])
        target_R = target_release(S_hat[d], I, d, R[d-1])
        R[d] = actual_release(target_R, I, S[d])

        S[d + 1] = S[d] + I - R[d]

        out = {'storage' : S, 'outflow':R}
        result = pd.DataFrame(out)
    return result

################################################################################
# Define release, and NOR harmonics for independent use
################################################################################


# Define the average daily release function
def release_harmonic(data, time, timestep = 'daily'):
    if timestep == 'daily':
        time = time/7
    R_avg_t = (data['Release_alpha1'] * sin(2 * pi * (time + 39)/52) +
             data['Release_alpha2'] * sin(4 * pi * (time + 39)/52) +
             data['Release_beta1'] * cos(2 * pi * (time + 39)/52) +
             data['Release_beta2'] * cos(4 * pi * (time + 39)/52))
    return R_avg_t

# Calculate daily values of the upper NOR bound
def NOR_hi(data, time, timestep = 'daily'):
    # NOR harmonic is at weekly step
    if timestep == 'daily':
        time = time/7

    NOR_hi = (data['NORhi_mu'] + data['NORhi_alpha'] * sin(2*pi*(time + 39)/52) +
                 data['NORhi_beta'] * cos(2*pi*(time + 39)/52))

    if (NOR_hi < data['NORhi_min']):
        NOR_hi = data['NORhi_min']
    elif (NOR_hi > data['NORhi_max']):
        NOR_hi = data['NORhi_max']
    return NOR_hi/100

# Calculate daily values of the lower NOR bound
def NOR_lo(data, time, timestep = 'daily'):
    # NOR harmonic is at weekly step
    if timestep == 'daily':
        time = time/7

    NOR_lo = (data['NORlo_mu'] + data['NORlo_alpha'] * sin(2*pi*(time + 39)/52) +
                 data['NORlo_beta'] * cos(2*pi*(time + 39)/52))

    if (NOR_lo < data['NORlo_min']):
        NOR_lo = data['NORlo_min']
    elif (NOR_lo > data['NORlo_max']):
        NOR_lo = data['NORlo_max']
    return NOR_lo/100
