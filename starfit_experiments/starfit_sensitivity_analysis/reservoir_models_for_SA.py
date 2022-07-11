"""
Trevor Amestoy
Summer 2022

Simulates reservoir storage and releases, using STARFIT inferred operating
rule parameters.
"""
import numpy as np
import pandas as pd
from math import pi, sin, cos

def sim_reservoir_S(params, reservoir_name = 'blueMarsh'):
    """
    Parameters:
    ----------
    param_samples : obj
        A saltelli sample, from SALib, containing all STARFIT param samples.

    Returns:
    --------
    storage : array
        An array of weekly reservoir storage volumes.
    releases : array
        An array of weekly reservoir release volumes.
    """
    uncertain_params = ['NORhi_alpha', 'NORhi_beta', 'NORhi_max', 'NORhi_min',
        'NORhi_mu', 'NORlo_alpha', 'NORlo_beta', 'NORlo_max', 'NORlo_min',
        'NORlo_mu', 'Release_alpha1', 'Release_alpha2', 'Release_beta1',
        'Release_beta2', 'Release_max', 'Release_min', 'Release_c', 'Release_p1',
        'Release_p2']

    # Load starfit data for DRB reservoirs
    starfit_df = pd.read_csv('../model_data/drb_model_istarf_conus.csv')
    reservoirs = [res for res in starfit_df['reservoir']]

    # Find the index of the desired reservoir
    res_index = starfit_df.index[starfit_df['reservoir'] == reservoir_name].tolist()

    # Check that reservoir is contained in the starfit_df
    if not res_index:
        print('reservoir_name was not found in starfit_df.\n Check the reservoir_name and try again.\n')
        return

    # Source all starfit data for reservoir of interest in dictionary
    full_starfit_params = starfit_df.columns.tolist()
    known_params = []

    for check_param in full_starfit_params:
        if check_param not in uncertain_params:
            known_params.append(check_param)

    # build data frame
    data = pd.DataFrame()
    for c in range(len(known_params)):
        data[known_params[c]] = starfit_df[known_params[c]][res_index].values
    for c in range(len(uncertain_params)):
        data[uncertain_params[c]] = params[c]

    # Define reservoir constant characteristics
    R_max = ((data['Release_max'] + 1) * data['GRanD_MEANFLOW_MGD'] * 7).values
    R_min = ((data['Release_min'] + 1) * data['GRanD_MEANFLOW_MGD'] * 7).values
    I_bar = data['GRanD_MEANFLOW_MGD'].values * 7
    S_cap = data['GRanD_CAP_MG'].values
    S_initial = S_cap * data['NORhi_mu']/100


    ### Generate randome inflow WEEKLY
    n_weeks = 52
    inflow = abs(np.random.normal((I_bar), 50, n_weeks))

    ### Define internal functions
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
            return min(I_bar * (release_harmonic(week) +
                                    release_adjustment(S_hat, week))
                           + I_bar, R_max)
        elif (S_hat > NOR_hi):
            return min(S_cap * (S_hat - NOR_hi) + inflow[week], R_max)
        elif (S_hat < NOR_lo):
            return R_min
        else:
            print('Somehow does not qualify.. check target_R function.')
            print(f'Shat = {S_hat}, and NORs are {NOR_hi} and {NOR_lo}')
            return None



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
    weeks_in_NOR = 0

    # Simulate at weekly step
    for wk in range(len(inflow) - 1):

        I = inflow[wk]
        S_hat[wk] = percent_storage(S[wk])
        R_target = target_release(S_hat[wk], wk+1)
        R[wk] = actual_release(R_target, I, S[wk])

        S[wk + 1] = S[wk] + I - R[wk]

        # Check if currently in NOR
        NOR_hi_t = NOR_weekly_hi(wk)
        NOR_lo_t = NOR_weekly_lo(wk)
        if (S_hat[wk] <= NOR_hi_t) and (S_hat[wk] >= NOR_lo_t):
            weeks_in_NOR += 1

    # Calculate output metric
    out = (weeks_in_NOR / 52) * 100

    return out
