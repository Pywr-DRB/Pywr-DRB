"""
Trevor Amestoy
Summer 2022

Testing reservoir simulator which uses STARFIT rule parameters.

Tests performance of Blue Marsh and Beltzville reservoirs for:
"""

# Core modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Custom simulation and harmonic functions using STARFIT params
from simulate_reservoir_weekly import sim_starfit_reservoir_weekly
from simulate_reservoir_daily import sim_starfit_reservoir_daily
from simulate_reservoir_daily import NOR_hi, NOR_lo, release_harmonic


### Load STARFIT conus data for reservoirs in DRB
starfit = pd.read_csv('../model_data/drb_model_istarf_conus.csv')
reservoirs = [res for res in starfit['reservoir']]


# Simulate weekly then daily operations
for t, time_step in zip([52, 365], ['weekly', 'daily']):

    # Constants
    n_sims = 3
    n_time = t
    
    # Simulate Blue Marsh and Beltzville
    for i in [4,5]:

        # Select one set of STARFIT data at a time
        reservoir_data = starfit.iloc[i]
        
        if i == 4:
            reservoir_lab = 'Blue Marsh'
        elif i == 5:
            reservoir_lab = 'Beltzville'
        

        # Select initial storage with respect to reservoir WEEKLY
        if time_step == 'weekly':
            I_bar = reservoir_data['GRanD_MEANFLOW_MGD'] * 7
            x_lab = 'Week'
            
        elif time_step == 'daily':
            I_bar = reservoir_data['GRanD_MEANFLOW_MGD']
            x_lab = 'Day'

        S_cap = reservoir_data['GRanD_CAP_MG']
        R_max = ((reservoir_data['Release_max']+1) * (I_bar))
        R_min = ((reservoir_data['Release_min']+1) * (I_bar))

        # Test different initial storage
        initial_conditions = [(S_cap * reservoir_data['NORhi_mu']/100),
                              (S_cap * reservoir_data['NORhi_mu']*1.2/100),
                              (S_cap * reservoir_data['NORhi_mu']*0.6/100)]

        ### Generate some inflow data

        constant_inflow_mid = np.ones((n_time))*(I_bar)
        constant_inflow_low = np.ones((n_time))*(I_bar * 0.7)
        constant_inflow_high = np.ones((n_time))*(I_bar * 1.3)
        random_inflow = abs(np.random.normal((I_bar), 200, (n_time)))

        # Select one of the inflow timeseries
        sim_inflow = random_inflow
        #inflow_lab = 'Constant Inflow: Historic Mean'
        inflow_lab = 'Random inflow: ~N(Mean, 200)'

        # Initialize vectors
        R_avg = np.zeros(n_time)
        NOR_hi_harmonic = np.zeros(n_time)
        NOR_lo_harmonic = np.zeros(n_time)

        if time_step == 'weekly':
            for wk in range(n_time):
                R_avg[wk] = release_harmonic(reservoir_data, wk, timestep = time_step)
                NOR_hi_harmonic[wk] = NOR_hi(reservoir_data, wk, timestep = time_step)
                NOR_lo_harmonic[wk] = NOR_lo(reservoir_data, wk, timestep = time_step)
        elif time_step == 'daily':
            for d in range(n_time):
                R_avg[d] = release_harmonic(reservoir_data, d, timestep = time_step)
                NOR_hi_harmonic[d] = NOR_hi(reservoir_data, d, timestep = time_step)
                NOR_lo_harmonic[d] = NOR_lo(reservoir_data, d, timestep = time_step)

        # Initialize vectors for multiple sims
        S = np.zeros((n_sims, n_time))
        S_percent = np.zeros((n_sims, n_time))
        R = np.zeros((n_sims, n_time))

        for s in range(n_sims):
            S_initial = initial_conditions[s]

            if time_step == 'weekly':
                S[s,:], S_percent[s,:], R[s,:] = sim_starfit_reservoir_weekly(starfit, reservoirs[i], sim_inflow, S_initial)
            elif time_step == 'daily':
                S[s,:], S_percent[s,:], R[s,:] = sim_starfit_reservoir_daily(starfit, reservoirs[i], sim_inflow, S_initial)


        # Plot outputs
        x = np.arange(n_time)
        sim_labs = ['%$S_t$ for mean $S_i$', '%$S_t$ for high $S_i$', '%$S_t$ for low $S_i$']

        # Plot the inflow timeseries
        plt.plot(range(t), sim_inflow)
        plt.xlabel(x_lab)
        plt.ylabel('Test Inflow (MGD or MGW)')
        plt.title(inflow_lab)
        plt.show()

        # Seasonal release harmonic
        plt.plot(x, R_avg)
        plt.xlabel('Proportional Release')
        plt.title(f'{reservoir_lab} Reservoir\nAverage Weekly Release Harmonic')
        plt.xlabel(x_lab)
        plt.show()

        # Harmonic NOR
        plt.plot(x, NOR_hi_harmonic, label = 'NOR Bounds', color = 'black', alpha = 0.2, linestyle='dashed')
        plt.plot(x, NOR_lo_harmonic, color = 'black', alpha = 0.2, linestyle='dashed')
        plt.title(f'{reservoir_lab} Reservoir\nNormal Operating Range')
        plt.ylabel('Percent Storage Capacity (%)')
        plt.xlabel(x_lab)
        plt.legend()
        plt.show()

        # Simulated storage with NOR
        for k in range(n_sims):
            plt.plot(x, S[k,:], label = sim_labs[k])
        plt.plot(x, (S_cap * NOR_hi_harmonic/100), color = 'black', alpha = 0.3, label = 'NOR Bounds', linestyle = 'dashed')
        plt.plot(x, (S_cap * NOR_lo_harmonic/100), color = 'black', alpha = 0.3, linestyle = 'dashed')
        plt.title(f'{reservoir_lab} Reservoir\nSimulated Storage:\n {inflow_lab}')
        plt.ylabel('Storage Volume (MG)')
        plt.xlabel(x_lab)
        plt.legend()
        plt.show()

        # Simulated percent storage with NOR
        for k in range(n_sims):
            plt.plot(x, S_percent[k, :], label = sim_labs[k])
        plt.plot(x, NOR_hi_harmonic, label = 'NOR Bounds', color = 'black', alpha = 0.2, linestyle='dashed')
        plt.plot(x, NOR_lo_harmonic, color = 'black', alpha = 0.2, linestyle='dashed')
        plt.title(f'{reservoir_lab} Reservoir\nSimulated Percent Storage:\n {inflow_lab}')
        plt.ylabel('Percent Storage Capacity (%)')
        plt.xlabel(x_lab)
        plt.legend()
        plt.show()

        # Simulated releases
        for k in range(n_sims):
            plt.plot(x[1:], R[k, 1:], label = sim_labs[k])
        plt.plot(x, (np.ones(n_time) * R_max), label = '$R_{max}$',  color = 'black', alpha = 0.2, linestyle='dashed')
        plt.plot(x, (np.ones(n_time) * R_min), label = '$R_{min}$',  color = 'black', alpha = 0.5, linestyle='dashed')
        plt.title(f'{reservoir_lab} Reservoir\nSimulated Release Actions:\n {inflow_lab}')
        plt.ylabel('Release Volume (MGD)')
        plt.xlabel(x_lab)
        plt.legend()
        plt.show()
