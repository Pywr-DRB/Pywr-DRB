"""
Trevor Amestoy
Summer 2022

Testing reservoir simulator which uses STARFIT rule parameters.
"""

# Core modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Custom simulation and harmonic functions using STARFIT params
from simulate_reservoir import sim_starfit_reservoir
from simulate_reservoir import release_harmonic, NOR_weekly_hi, NOR_weekly_lo

### Load STARFIT conus data for reservoirs in DRB
starfit = pd.read_csv('model_data/drb_model_istarf_conus.csv')
reservoirs = [res for res in starfit['reservoir']]

### Generate some inflow data
n_sims = 3
n_weeks = 52
constant_inflow_mid = np.ones((n_weeks))*500
constant_inflow_low = np.ones((n_weeks))*300
constant_inflow_high = np.ones((n_weeks))*800
random_inflow = np.random.normal(500, 200, (n_weeks))


for i in [4,5]:
    
    # Select one set of STARFIT data at a time
    reservoir_data = starfit.iloc[i]

    # Select one of the inflow timeseries
    sim_inflow = random_inflow

    # Select initial storage with respect to reservoir 
    I_bar = reservoir_data['GRanD_MEANFLOW_MGD']
    S_cap = reservoir_data['GRanD_CAP_MG']
    
    
    initial_conditions = [(S_cap * reservoir_data['NORhi_mu']/100), 
                          (S_cap * reservoir_data['NORhi_mu']*1.5/100),
                          (S_cap * reservoir_data['NORhi_mu']*0.6/100)]


    # Initialize vectors
    R_avg = np.zeros(n_weeks)
    NOR_hi_harmonic = np.zeros(n_weeks)
    NOR_lo_harmonic = np.zeros(n_weeks)

    for wk in range(n_weeks):
        R_avg[wk] = release_harmonic(reservoir_data, wk)
        NOR_hi_harmonic[wk] = NOR_weekly_hi(reservoir_data, wk)
        NOR_lo_harmonic[wk] = NOR_weekly_lo(reservoir_data, wk)

    # Initialize vectors for multiple sims
    S = np.zeros((n_sims, n_weeks))
    S_percent = np.zeros((n_sims, n_weeks))
    R = np.zeros((n_sims, n_weeks))
    
    for s in range(n_sims):
        S_initial = initial_conditions[s]
        
        S[s,:], S_percent[s,:], R[s,:] = sim_starfit_reservoir(starfit, reservoirs[i], sim_inflow, S_initial)

    # Plot outputs
    x = np.arange(n_weeks)
    sim_labs = ['Mean $S_i$', 'High $S_i$', 'Low $S_i$']

    
    plt.plot(x, R_avg)
    plt.xlabel('Proportional Release')
    plt.title(f'Average Weekly Release Harmonic: \n{reservoirs[i]} Reservoir')
    plt.xlabel('Week')
    plt.show()
    
    plt.plot(x, NOR_hi_harmonic, label = 'NOR Upper')
    plt.plot(x, NOR_lo_harmonic, label = 'NOR Lower')
    plt.title(f"Normal Operating Range: \n{reservoirs[i]} Reservoir")
    plt.ylabel('Percent Storage Capacity (%)')
    plt.xlabel('Week')
    plt.legend()
    plt.show()
    
    for k in range(n_sims):
        plt.plot(x, S[k,:], label = sim_labs[k])
    plt.title(f"Simulated Storage: \n{reservoirs[i]} Reservoir")
    plt.ylabel('Storage Volume (MG)')
    plt.xlabel('Week')
    plt.legend()
    plt.show()
    
    for k in range(n_sims):
        plt.plot(x, S_percent[k, :], label = sim_labs[k])
    plt.ylabel('Percent Storage Capacity (%)')
    plt.xlabel('Week')
    plt.legend()
    plt.show()
    
    for k in range(n_sims):
        plt.plot(x, R[k, :], label = sim_labs[k])
    plt.ylabel('Release Volume (MGD)')
    plt.xlabel('Week')
    plt.legend()
    plt.show()
