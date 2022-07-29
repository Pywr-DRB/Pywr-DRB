"""
Trevor Amestoy

Compares the pywr-implemented starfit operations with external starfit
simulator.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

from simulate_reservoir_daily import sim_starfit_reservoir_daily as starfit_sim
from simulate_reservoir_daily import NOR_hi, NOR_lo, release_harmonic


### Load pywr model results
with h5py.File('../output_data/drb_output.hdf5', 'r') as f:
    keys = list(f.keys())
    pywr_results = pd.DataFrame({keys[0]: f[keys[0]][:,2]})
    for k in keys[1:]:
        if 'catchment' in k or 'outflow' in k or 'reservoir' in k or 'delLordville' in k or 'delMontague' in k:
            pywr_results[k] = f[k][:,2]
    day = [f['time'][i][0] for i in range(len(f['time']))]
    month = [f['time'][i][2] for i in range(len(f['time']))]
    year = [f['time'][i][3] for i in range(len(f['time']))]
    date = [f'{y}-{m}-{d}' for y,m,d in zip(year, month, day)]
    date = pd.to_datetime(date)
    pywr_results.index = date
pywr_results

# Simulation time period
start_date = '2005-10-01'
end_date = '2012-10-01'



### Load STARFIT conus data for reservoirs in DRB
starfit = pd.read_csv('../model_data/drb_model_istarf_conus.csv')
reservoirs = [res for res in starfit['reservoir']]
reservoir_ids = [id for id in starfit['GRanD_ID']]
reservoir_names = [name for name in starfit['GRanD_NAME']]



consider_reservoirs = ['blueMarsh']
gage_ids = ['01470960']


for reservoir, gage in zip(consider_reservoirs, gage_ids):

    ### Load USGS data
    usgs_data = pd.read_csv(f'./usgs_data/clean_usgs_{gage}.csv')

    ### Load ResOpsUS outlfow data
    resops_data = pd.read_csv(f'./ResOpsUS_data/resops_{reservoir}.csv', sep = ',')

    # Find index corresponding to pywr range
    start_index = [i for i,x in enumerate(resops_data['date'] == start_date) if x][0]
    end_index = [i for i,x in enumerate(resops_data['date'] == end_date) if x][0]

    # Pull specific range and scale
    resops_inflow = resops_data['inflow'][start_index:end_index + 1].values*22.824
    resops_outflow = resops_data['outflow'][start_index:end_index + 1].values*22.824
    resops_inflow[resops_inflow < 0] = 0
    resops_outflow[resops_outflow < 0] = 0

    pywr_inflow = pywr_results[f'catchment_{reservoir}']
    pywr_Si = pywr_results[f'reservoir_{reservoir}'][0]
    pywr_storage = pywr_results[f'reservoir_{reservoir}']
    pywr_outflow = pywr_results[f'outflow_{reservoir}']

    starfit_results = starfit_sim(starfit, reservoir, pywr_inflow, pywr_Si)

    t = range(len(pywr_inflow))

    # Plot comparison of simulated storage
    fig,axs = plt.subplots(2, 1, figsize = (10,5), gridspec_kw={'height_ratios': [3, 1]}, dpi = 300)
    ax = axs[0]
    ax.plot(t, starfit_results['storage'], label = 'starfit', color = 'green')
    ax.plot(t, pywr_storage, label = 'pywr', linestyle = '-', color = 'orange')
    ax.set_title(f'{reservoir}\nSimulated storage (MG)')
    ax.set_ylabel('Storage (MG)')
    ax.legend(bbox_to_anchor = (1.2, 0.5))
    ax.set_xticklabels([])
    ax = axs[1]
    ax.plot(t, pywr_inflow, label = 'Inflow')
    ax.set_xlabel('Days')
    ax.set_ylabel('Inflow (MGD)')
    plt.show()


    # Plot comparison of simulated outflows    
    fig,axs = plt.subplots(2, 1, figsize = (10,5), gridspec_kw={'height_ratios': [3, 1]}, dpi = 300)
    ax = axs[0]
    #ax.plot(t[0:730], resops_outflow[0:730], label = 'ResOpsUS', color = 'pink')
    ax.plot(t[0:730], starfit_results['outflow'][0:730], label = 'starfit', color = 'green')
    ax.plot(t[0:730], pywr_outflow[0:730], label = 'pywr', color = 'orange')
    #ax.set_ylim([0,1000])

    ax.plot(t[0:730], np.ones(730)*555, label = 'Specified $R_{max}$', color = 'grey', linestyle = ':')
    ax.set_title(f'{reservoir}\nSimulated releases (MGD)')
    ax.set_ylabel('Release (MGD)')
    ax.legend(bbox_to_anchor = (1.05, 0.5))
    ax = axs[1]
    ax.plot(t[0:730], pywr_inflow[0:730], label = 'Inflow')
    ax.set_xlabel('Days')
    ax.set_ylabel('Inflow (MGD)')
    plt.show()


    # Plot comparison of inflow timeseries
    fig,axs = plt.subplots(2, 1, figsize = (10,7), gridspec_kw={'height_ratios': [2, 1]}, dpi = 300)
    ax = axs[0]
    ax.plot(t, np.log(resops_inflow), color = 'pink', label = 'ResOps Inflow')
    ax.plot(t, np.log(pywr_inflow), color = 'blue', label = 'Scaled NHM')
    ax.set_title(f'Comparison of inflow data for {reservoir}')
    ax.set_ylabel('Log-Inflow')
    ax.legend(bbox_to_anchor = (1.05, 0.5))
    ax = axs[1]
    ax.plot(t, abs(resops_inflow - pywr_inflow), color = 'maroon')
    ax.set_ylabel('Difference in inflow (MGD)')
    ax.set_xlabel('Days')
    plt.show()

    
    # Comparison of log outflows
    fig,ax = plt.subplots(figsize = (10,7), dpi = 300)
    ax.plot(t, np.log(resops_outflow), color = 'pink', label = 'ResOps Inflow')
    ax.plot(t, np.log(pywr_outflow), color = 'blue', label = 'Scaled NHM')
    #ax.plot(range(len(usgs_data['flow'])), np.log(usgs_data['flow']), label = 'usgs', color = 'lightgreen')
    ax.set_title(f'Comparison of outflow data for {reservoir}')
    ax.set_ylabel('Log-flow')
    ax.set_xlabel('Days')
    ax.legend(bbox_to_anchor = (1.05, 0.5))
    plt.show()

    # Plot comparison of all outflows    
    fig,ax = plt.subplots(1, 1, figsize = (7,3), dpi = 200)
    ax.plot(t[0:730], usgs_data['flow'][0:730], label = 'usgs', color = 'lightgreen')
    ax.plot(t[0:730], resops_outflow[0:730], label = 'ResOpsUS', color = 'pink')
    #ax.plot(t[0:730], starfit_results['outflow'][0:730], label = 'starfit', color = 'green')
    #ax.plot(t[0:730], pywr_outflow[0:730], label = 'pywr', color = 'orange')
    #ax.set_ylim([0,1000])

    ax.plot(t[0:730], np.ones(730)*555, label = 'Specified $R_{max}$', color = 'grey', linestyle = ':')
    ax.set_title(f'{reservoir}\nRecorded outflow vs release (MGD)')
    ax.set_ylabel('Flow (MGD)')
    ax.legend(bbox_to_anchor = (1.35, 0.5))
    ax.set_xlabel('Days')
    plt.show()
