#############################################################################

# Marilyn Simth Summer 2024 Flow Duration Curve Plotting
# This script is used to plot the flow duration curve for a specific reservoir.
# The flow duration curve is a plot of the flow rate against the exceedance probability.


#############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataretrieval import nwis


def fetch_nwis_data(site_number, start_date, end_date):
    """
    Fetches daily streamflow data from NWIS.
    
    Parameters:
    site_number (str): The USGS site number to fetch data from.
    start_date (str): The start date for the data.
    end_date (str): The end date for the data.
    
    Returns:
    pd.Series: The average daily discharge data.
    """
    parameter_code = '00060'  # Discharge
    daily_streamflow = nwis.get_dv(sites=site_number, parameterCd=parameter_code, start=start_date, end=end_date)
    if daily_streamflow:
        data = daily_streamflow[0]
        # Ensure 'datetime' is in the index
        data.index = pd.to_datetime(data.index)
        # Set the 'datetime' as index if not already
        if 'datetime' in data.columns:
            data.set_index('datetime', inplace=True)
        # Extract the discharge data
        return data['00060_Mean']
    else:
        print(f"No data retrieved for site {site_number}")
        return None


def plot_flow_duration_curve(reservoir_downstream_gages, reservoir, model, start_date=None, end_date=None, 
                             end_inclusive=False, colordict=None, save_fig=False, fig_dir=None, fetch_nwis=True):
    """
    Plots the flow duration curve for a specific reservoir.
    
    Parameters:
    reservoir_downstream_gages (dict): Dictionary containing reservoir downstream gages data for each model.
    reservoir (str): The name of the reservoir to plot the flow duration curve for.
    model (str): The model to use for plotting.
    start_date (str or None): The start date for the data to plot.
    end_date (str or None): The end date for the data to plot.
    end_inclusive (bool): Whether to include the end date in the data subset.
    colordict (dict or None): Dictionary of colors for plotting.
    save_fig (bool): Whether to save the figure.
    fig_dir (str or None): The directory to save the figure in.
    fetch_nwis (bool): Whether to fetch data from NWIS for reservoirs not in the model.
    
    Returns:
    None
    """
    
    # Default colors if not provided
    if colordict is None:
        colordict = {'obs': 'black', model: 'blue'}
    
    # Function to get the correct modeled data
    def get_fig_data(data_dict, model, node):
        if node in data_dict[model][0]:
            data = subset_timeseries(data_dict[model][0][node], start_date, end_date)
        else:
            print(f'get_fig_data() not set for node {node}')
            data = None
        return data

    # Function to get the correct observed data
    def get_fig_data_obs(data_dict, model, node):
        if node in data_dict[model]:
            data = subset_timeseries(data_dict[model][node], start_date, end_date)
        else:
            print(f'get_fig_data_obs() not set for node {node}')
            data = None
        return data
    
    # Function to resample the data based on the selected frequency
    def resample_data(data, frequency):
        if frequency == 'daily':
            return data
        elif frequency == 'weekly':
            return data.resample('W').mean()
        elif frequency == 'annual':
            return data.resample('A').mean()
        else:
            raise ValueError("Invalid frequency. Choose from 'daily', 'weekly', or 'annual'.")

    # Define the NWIS site numbers
    nwis_sites = {
        'prompton': '01430000',
        'ontelaunee': '01470761',
        'stillCreek': '01469500'
    }
    
    # Fetch NWIS data if needed
    obs = get_fig_data_obs(reservoir_downstream_gages, 'obs', reservoir)
    if obs is None and fetch_nwis:
        site_number = nwis_sites.get(reservoir, None)
        if site_number:
            obs = fetch_nwis_data(site_number, start_date, end_date)
    
    modeled = get_fig_data(reservoir_downstream_gages, model, reservoir)
    
    if modeled is None:
        print("Error: Modeled data for the specified reservoir is not available.")
        return

    if obs is None and not fetch_nwis:
        print(f"No observed data available for {reservoir} and NWIS data fetch is not enabled.")

    # Define the frequencies and corresponding titles
    frequencies = ['daily', 'weekly', 'annual']
    titles = ['Daily', 'Weekly', 'Annual']

    # Plotting the flow duration curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    def plot_exceedance(data, ax, color, label, **kwargs):
        df = data.sort_values()[::-1]
        exceedance = (np.arange(1., len(df) + 1.) / len(df) * 100)
        ax.plot(exceedance, df, color=color, label=label, **kwargs)

    for ax, frequency, title in zip(axes, frequencies, titles):
        modeled_resampled = resample_data(modeled, frequency)
        
        # Plot the modeled data
        plot_exceedance(modeled_resampled, ax, color=colordict[model], label=model, alpha=1, zorder=1)
        
        # Plot the observed data if available
        if obs is not None:
            obs_resampled = resample_data(obs, frequency)
            plot_exceedance(obs_resampled, ax, color=colordict['obs'], label='Observed', alpha=1, zorder=2)
        
        ax.set_xlabel('Exceedence (%)', fontsize=12)
        if ax == axes[0]:
            ax.set_ylabel('Flow', fontsize=12)
        ax.set_yscale('log')
        ax.set_title(f'{title} Flow Duration Curve', fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    fig.suptitle(f'Flow Duration Curves for {reservoir}', fontsize=16)
    axes[0].legend()
    
    if save_fig:
        if fig_dir is not None:
            fig.savefig(f'{fig_dir}flow_duration_curve_{reservoir}_{model}_all.png', bbox_inches='tight', dpi=300)
        else:
            fig.savefig(f'flow_duration_curve_{reservoir}_{model}_all.png', bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

# Helper function to subset time series data
def subset_timeseries(data, start_date, end_date, end_inclusive=False):
    if start_date is not None:
        data = data[data.index >= start_date]
    if end_date is not None:
        if end_inclusive:
            data = data[data.index <= end_date]
        else:
            data = data[data.index < end_date]
    return data

