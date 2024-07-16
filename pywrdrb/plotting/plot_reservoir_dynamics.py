#############################################################################

# Marilyn Simth Summer 2024 Plot reservoir dynamics 
# This script is used to plot the inflow, storage, and reservoir releases for a specific node using major data.

#############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pywrdrb.utils.directories import model_data_dir
from pywrdrb.plotting.styles import base_model_colors, paired_model_colors, scatter_model_markers
import matplotlib.ticker as ticker

# Function to calculate sin and cos components
def sinNpi(day, N, first_month='Oct'):
    if first_month == 'Oct':
        return np.sin(N * np.pi * (day) / 52)
    elif first_month == 'Jan':
        return np.sin(N * np.pi * (day + 39) / 52)

def cosNpi(day, N, first_month='Oct'):
    if first_month == 'Oct':
        return np.cos(N * np.pi * (day) / 52)
    elif first_month == 'Jan':
        return np.cos(N * np.pi * (day + 39) / 52)

# Function to calculate NOR upper bound
def calc_NOR_hi(data, times, timestep='daily', first_month='Oct'):
    if timestep == 'daily':
        times = times / 7
    
    NOR_hi = []
    for time in times:
        NOR_hi.append(data['NORhi_mu'] + data['NORhi_alpha'] * sinNpi(time, 2, first_month=first_month) +
                      data['NORhi_beta'] * cosNpi(time, 2, first_month=first_month))
    
    NOR_hi = np.array(NOR_hi)
    NOR_hi = np.where(NOR_hi < data['NORhi_min'], data['NORhi_min'], NOR_hi)
    NOR_hi = np.where(NOR_hi > data['NORhi_max'], data['NORhi_max'], NOR_hi)
    
    return NOR_hi / 100

# Function to calculate NOR lower bound
def calc_NOR_lo(data, times, timestep='daily', first_month='Oct'):
    if timestep == 'daily':
        times = times / 7
    
    NOR_lo = []
    for time in times:
        NOR_lo.append(data['NORlo_mu'] + data['NORlo_alpha'] * sinNpi(time, 2, first_month=first_month) +
                      data['NORlo_beta'] * cosNpi(time, 2, first_month=first_month))
    
    NOR_lo = np.array(NOR_lo)
    NOR_lo = np.where(NOR_lo < data['NORlo_min'], data['NORlo_min'], NOR_lo)
    NOR_lo = np.where(NOR_lo > data['NORlo_max'], data['NORlo_max'], NOR_lo)
    
    return NOR_lo / 100

# Function to get reservoir capacity from ISTARF data
def get_reservoir_capacity(reservoir, modified=False):
    istarf = pd.read_csv(f'{model_data_dir}drb_model_istarf_conus.csv', index_col=0)
    if modified:
        return float(istarf.loc[f'modified_{reservoir}', 'Adjusted_CAP_MG'])
    else:
        return float(istarf.loc[reservoir, 'Adjusted_CAP_MG'])

# Function to retrieve Starfit parameters for a reservoir
def get_starfit_params(reservoir, modified=False):
    istarf = pd.read_csv(f'{model_data_dir}drb_model_istarf_conus.csv', index_col=0)
    if modified:
        return istarf.loc[f'modified_{reservoir}']
    else:
        return istarf.loc[reservoir]

# Function to subset time series data
def subset_timeseries(data, start_date, end_date, end_inclusive=False):
    if start_date:
        data = data.loc[start_date:]
    if end_date:
        end_date = pd.Timestamp(end_date)  # Convert end_date to Timestamp
        if end_inclusive:
            data = data.loc[:end_date]
        else:
            data = data.loc[:end_date - pd.Timedelta(days=1)]  # Subtract one day for non-inclusive end
    return data

# Main function to plot reservoir dynamics
def plot_reservoir_dynamics(inflows, storage, releases, node, model,
                            start_date=None, end_date=None, end_inclusive=False,
                            colordict=None, save_fig=False, fig_dir=None,
                            log_scale=False, plot_percent=True, plot_NOR=True, plot_observed=False):
    """
    Plots the inflow, storage, and reservoir releases for a specific node using major data.

    Parameters:
    inflows (dict): Dictionary containing inflow data for each model.
    storage (dict): Dictionary containing storage data for each model.
    releases (dict): Dictionary containing reservoir release data for each model.
    node (str): The node to plot the data for.
    model (str): The model to use for plotting.
    start_date (str or None): The start date for the data to plot.
    end_date (str or None): The end date for the data to plot.
    end_inclusive (bool): Whether to include the end date in the data subset.
    colordict (dict or None): Dictionary of colors for plotting.
    save_fig (bool): Whether to save the figure.
    fig_dir (str or None): The directory to save the figure in.
    log_scale (bool): Whether to use a logarithmic scale for the y-axis.
    plot_percent (bool): Whether to plot percent capacity with NOR bounds.
    plot_NOR (bool): Whether to plot the NOR bounds for Starfit reservoirs.
    plot_observed (bool): Whether to plot observed data if available.

    Returns:
    None
    """

    # Default colors if not provided
    if colordict is None:
        colordict = {model: 'blue'}

    # Function to get the correct data
    def get_fig_data(data_dict, model, node):
        if node in data_dict[model][0]:
            data = subset_timeseries(data_dict[model][0][node], start_date, end_date)
        else:
            print(f'get_fig_data() not set for node {node}')
            data = None
        return data

    # Retrieve data for inflows, storage, and releases
    inflow_data = get_fig_data(inflows, model, node)
    storage_data = get_fig_data(storage, model, node)
    release_data = get_fig_data(releases, model, node)

    # Function to get the correct observed data
    def get_fig_data_obs(data_dict, model, node):
        if node in data_dict[model]:
            data = subset_timeseries(reservoir_downstream_gages[model][node], start_date, end_date)
        else:
            print(f'get_fig_data() not set for node {node}')
            data = None
        return data

    # Retrieve observed data if available
    if plot_observed:
        observed_inflow = get_fig_data_obs(inflows, 'obs', node)
        observed_storage = get_fig_data_obs(storage, 'obs', node)
        observed_release = get_fig_data_obs(releases, 'obs', node)

    # Check if any data is missing
    if any(data is None for data in [inflow_data, storage_data, release_data]):
        print(f"Cannot make plot for node: {node} - Data not available.")
        return

    # Plotting inflows, storage, and releases in a 3-panel plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot inflows
    if inflow_data is not None:
        if log_scale:  # Always allow log scale option for inflows
            ax1.plot(inflow_data.index, np.log(inflow_data.values), color=colordict[model], label='Log Inflow')
            ax1.set_yscale('log')
            ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))
            ax1.set_ylabel('Log Inflow (log MCM)')
        else:
            ax1.plot(inflow_data.index, inflow_data.values, color=colordict[model], label='Inflow')
            ax1.set_ylabel('Inflow (MCM)')
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.grid(True)
    else:
        print(f"Cannot make plot for node: {node} - Inflow Data not available.")

    # Plot observed inflows if available
    if plot_observed:
        if observed_inflow is not None:
            if log_scale:
                ax1.plot(observed_inflow.index, np.log(observed_inflow.values), color='grey', linestyle='--', label='Observed Log Inflow')
                ax1.set_yscale('log')
                ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))
            else:
                ax1.plot(observed_inflow.index, observed_inflow.values, color='grey', linestyle='--', label='Observed Inflow')
            ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            print(f"Observed Inflow data not available for node: {node}.")

    # Plot storage based on user choice
    if storage_data is not None:
        if plot_percent:
            storage_capacity = get_reservoir_capacity(node)
            percent_capacity = storage_data.values / storage_capacity * 100
            ax2.plot(storage_data.index, percent_capacity, color='green', label='Percent Capacity')

            if plot_NOR:
                times = storage_data.index.day_of_year
                starfit_data = get_starfit_params(node)
                NOR_hi = calc_NOR_hi(starfit_data, times)
                NOR_lo = calc_NOR_lo(starfit_data, times)

                ax2.fill_between(storage_data.index, NOR_lo * 100, NOR_hi * 100, color='yellow', alpha=0.3, label='NOR Range')

            ax2.set_ylabel('Percent Capacity (%)')

        elif log_scale and not plot_NOR:  # Log option only if plot_percent is false and plot_NOR is false
            ax2.plot(storage_data.index, np.log(storage_data.values), color=colordict[model], label='Log Storage')
            ax2.set_ylabel('Log Storage (log MCM)')
        else:  # Actual storage plot
            ax2.plot(storage_data.index, storage_data.values, color=colordict[model], label='Storage')
            ax2.set_ylabel('Storage (MCM)')

        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.grid(True)
    else:
        print(f"Cannot make plot for node: {node} - Storage Data not available.")
        
    # Plot observed storage if available
    if plot_observed:
        if observed_storage is not None:
            if plot_percent:
                storage_capacity = get_reservoir_capacity(node)
                percent_capacity = observed_storage.values / storage_capacity * 100
                ax2.plot(observed_storage.index, percent_capacity, color='grey', linestyle='--', label='Observed Percent Capacity')

            if plot_NOR:
                times = observed_storage.index.day_of_year
                starfit_data = get_starfit_params(node)
                NOR_hi = calc_NOR_hi(starfit_data, times)
                NOR_lo = calc_NOR_lo(starfit_data, times)

                ax2.fill_between(observed_storage.index, NOR_lo * 100, NOR_hi * 100, color='yellow', alpha=0.3, label='Observed NOR Range')

            ax2.set_ylabel('Percent Capacity (%)')

        elif log_scale and not plot_NOR:  # Log option only if plot_percent is false and plot_NOR is false
            ax2.plot(observed_storage.index, np.log(observed_storage.values), color='grey', linestyle='--', label='Observed Log Storage')
            ax2.set_ylabel('Log Storage (log MCM)')
        else:  # Actual observed storage plot
            ax2.plot(observed_storage.index, observed_storage.values, color='grey', linestyle='--', label='Observed Storage')
            ax2.set_ylabel('Storage (MCM)')

        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.grid(True)
    else:
        print(f"Observed Storage data not available for node: {node}.")


    # Plot releases
    if release_data is not None:
        if log_scale:  # Always allow log scale option for releases
            ax3.plot(release_data.index, np.log(release_data.values), color=colordict[model], label='Log Releases')
            ax3.set_yscale('log')
            ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))
            ax3.set_ylabel('Log Releases (log MCM)')
        else:
            ax3.plot(release_data.index, release_data.values, color=colordict[model], label='Releases')
            ax3.set_ylabel('Releases (MCM)')
        ax3.set_xlabel('Date')
        ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax3.grid(True)
    else:
        print(f"Cannot make plot for node: {node} - Release Data not available.")

    # Plot observed releases if available
    if plot_observed:
        if observed_release is not None:
            if log_scale:
                ax3.plot(observed_release.index, np.log(observed_release.values), color='grey', linestyle='--', label='Observed Log Releases')
                ax3.set_yscale('log')
                ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))    
            else:
                ax3.plot(observed_release.index, observed_release.values, color='grey', linestyle='--', label='Observed Release')
            ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            print(f"Observed Release data not available for node: {node}.")

    # Set a title for the entire figure
    fig.suptitle(f"Reservoir Dynamics for {node}", fontsize=16)

    plt.tight_layout()

    # Save the figure if required
    if save_fig and fig_dir:
        plt.savefig(f"{fig_dir}/{node}_reservoir_dynamics.png")

    plt.show()

# Example usage
#plot_reservoir_dynamics(inflows, storages, reservoir_releases, 'blueMarsh', 'pywr_obs_pub_nhmv10_ObsScaled', colordict = paired_model_colors,
#                         save_fig=False, fig_dir=False, log_scale=True, plot_percent=True, plot_NOR=True, plot_observed=False)
