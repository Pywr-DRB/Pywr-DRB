"""
TJA
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns


def plot_flow_ranges(Q_h, Q_s, 
                        y_scale = 'log',
                        colors = ['grey', 'orange'], 
                        all_realizations = True,
                        realization = 0,
                        site = 0,
                        figsize = (7,3),
                        title_addon = ' ',
                        plot_medians=True):

    n_sites, n_years, n_timesteps = Q_h.shape
    n_reals = Q_s.shape[0]

    h_max = np.max(Q_h[site,:,:], axis = 0)
    h_min = np.min(Q_h[site,:,:], axis = 0)

    if all_realizations:
        s_max = np.max(np.max(Q_s[:, site, :,:], axis = 1), axis = 0)
        s_min = np.min(np.min(Q_s[:, site, :,:], axis = 1), axis = 0)
        syn_label = f'Synthetic Range Over All Realizations'
    elif not all_realizations:
        s_max = np.max(Q_s[realization, site, :,:], axis = 0)
        s_min = np.min(Q_s[realization, site, :,:], axis = 0)
        n_reals = 1
        syn_label = f'Synthetic Range for 1 Realization'

    if y_scale == 'log':
        y_lab_p2 = 'Log-Flow (cms)'
    else:
        y_lab_p2 = 'Flow (cms)'

    if n_timesteps > 300:
        x_lab = 'Day of the Year (Jan-Dec)'
        y_lab_p1 = 'Daily'
    elif n_timesteps == 12:
        x_lab = 'Month of the Year (Jan-Dec)'
        y_lab_p1 = 'Monthly'
    else:
        print('Error with timestep size...')
        return

    plt.figure(figsize = figsize, dpi=150)

    xs = np.arange(n_timesteps)
    plt.fill_between(xs, h_min, h_max, color = colors[0], label = 'Historic Range', alpha = 0.5)
    plt.fill_between(xs, s_min, s_max, color = colors[1], label = syn_label, alpha = 0.5)
    
    # plot median flows
    if plot_medians:
        h_med = np.median(Q_h[site,:,:], axis = 0)
        s_med = np.median(Q_s[:, site, :,:], axis = 1)
        plt.plot(xs, h_med, color = colors[0], label = 'Historic Median', alpha = 1, linewidth = 2)
        plt.plot(xs, s_med.T, color = colors[1], label = 'Synthetic Median', alpha = 1, linewidth = 2)
    
    plt.yscale(y_scale)
    plt.ylabel(f'{y_lab_p1} { y_lab_p2}')
    plt.xlabel(x_lab)
    plt.legend()
    plt.title(f'Range of Flows Across Years \n for Historic & Synthetic Timeseries at One Site \n {title_addon}')
    plt.show()
    return plt


def plot_flow_ranges(Qh, Qs, 
                     timestep = 'daily',
                     units = 'cms', y_scale = 'log',
                     savefig = False, fig_dir = '.',
                     figsize = (7,5), colors = ['black', 'orange'],
                     title_addon = ""):
    """Plots the range of flow for historic and syntehtic streamflows for a specific timestep scale.
     
    Args:
        Qh (pd.Series): Historic daily streamflow timeseries. Index must be pd.DatetimeIndex. 
        Qs (pd.DataFrame): Synthetic daily streamflow timeseries realizations. Each column is a unique realization. Index must be pd.DatetimeIndex.
        timestep (str, optional): The timestep which data should be aggregated over. Defaults to 'daily'. Options are 'daily', 'weekly', or 'monthly'.
        units (str, optional): Streamflow units, for axis label. Defaults to 'cms'.
        y_scale (str, optional): Scale of the y-axis. Defaults to 'log'.
        savefig (bool, optional): Allows for png to be saved to fig_dir. Defaults to False.
        fig_dir (str, optional): Location of saved figure output. Defaults to '.' (working directory).
        figsize (tuple, optional): The figure size. Defaults to (4,4).
        colors (list, optional): List of two colors for historic and synthetic data respectively. Defaults to ['black', 'orange'].
        title_addon (str, optional): Text to be added to the end of the title. Defaults to "".
    """
 
    # Assert formatting matches expected
    assert(type(Qh.index) == pd.DatetimeIndex), 'Historic streamflow (Qh) should have pd.DatatimeIndex.'
    assert(type(Qs.index) == pd.DatetimeIndex), 'Synthetic streamflow (Qh) should have pd.DatatimeIndex.'
 
    # Handle conditional datetime formatting
    if timestep == 'daily':
        h_grouper = Qh.index.dayofyear
        s_grouper = Qs.index.dayofyear
        x_lab = 'Day of the Year (Jan-Dec)'
    elif timestep == 'monthly':
        h_grouper = Qh.index.month
        s_grouper = Qs.index.month
        x_lab = 'Month of the Year (Jan-Dec)'
    elif timestep == 'weekly':
        h_grouper = pd.Index(Qh.index.isocalendar().week, dtype = int)
        s_grouper = pd.Index(Qs.index.isocalendar().week, dtype = int)
        x_lab = 'Week of the Year (Jan-Dec)'
    else:
        print('Invalid timestep input. Options: "daily", "monthly", "weekly".')
        return
 
    # Find flow ranges
    s_max = Qs.groupby(s_grouper).max().max(axis=1)
    s_min = Qs.groupby(s_grouper).min().min(axis=1)
    s_median = Qs.groupby(s_grouper).median().median(axis=1)
    h_max = Qh.groupby(h_grouper).max()
    h_min = Qh.groupby(h_grouper).min()
    h_median = Qh.groupby(h_grouper).median()
   
    ## Plotting  
    fig, ax = plt.subplots(figsize = figsize, dpi=150)
    xs = h_max.index
    ax.fill_between(xs, s_min, s_max, color = colors[1], label = 'Synthetic Range', alpha = 0.5)
    ax.plot(xs, s_median, color = colors[1], label = 'Synthetic Median')
    ax.fill_between(xs, h_min, h_max, color = colors[0], label = 'Historic Range', alpha = 0.3)
    ax.plot(xs, h_median, color = colors[0], label = 'Historic Median')
     
    ax.set_yscale(y_scale)
    ax.set_ylabel(f'{timestep.capitalize()} Flow ({units})', fontsize=12)
    ax.set_xlabel(x_lab, fontsize=12)
    ax.legend(ncols = 2, fontsize = 10, bbox_to_anchor = (0, -.5, 1.0, 0.2), loc = 'upper center')    
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_title(f'{timestep.capitalize()} Streamflow Ranges\nHistoric & Synthetic Timeseries at One Location\n{title_addon}')
    plt.tight_layout()
     
    if savefig:
        plt.savefig(f'{fig_dir}/flow_ranges_{timestep}.png', dpi = 150)
    return

##################################################

def find_continuous_FDC(data, quants):
    flows = np.quantile(data, quants, axis = 0)
    return flows


##################################################

def plot_fdc_ranges(Qh, Qs, 
                    units = 'cms', y_scale = 'log',
                    savefig = False, fig_dir = '.',
                    fname_addon = '',
                    figsize = (5,5), colors = ['black', 'orange'],                   
                    title_text = "",
                    ax=None):
    """Plots the range and aggregate flow duration curves for historic and synthetic streamflows.
     
    Args:
        Qh (pd.Series): Historic daily streamflow timeseries. Index must be pd.DatetimeIndex. 
        Qs (pd.DataFrame): Synthetic daily streamflow timeseries realizations. Each column is a unique realization. Index must be pd.DatetimeIndex.
        units (str, optional): Streamflow units, for axis label. Defaults to 'cms'.
        y_scale (str, optional): Scale of the y-axis. Defaults to 'log'.
        savefig (bool, optional): Allows for png to be saved to fig_dir. Defaults to False.
        fig_dir (str, optional): Location of saved figure output. Defaults to '.' (working directory).
        figsize (tuple, optional): The figure size. Defaults to (4,4).
        colors (list, optional): List of two colors for historic and synthetic data respectively. Defaults to ['black', 'orange'].
        title_text (str, optional): Text for the title. Defaults to "".
        fname_addon (str, optional): Text to be added to the end of the filename. Defaults to "".
    """
     
    ## Assertions
    assert(type(Qs) == pd.DataFrame), 'Synthetic streamflow should be type pd.DataFrame.'
    assert(type(Qh.index) == pd.DatetimeIndex), 'Historic streamflow (Qh) should have pd.DatatimeIndex.'
    assert(type(Qs.index) == pd.DatetimeIndex), 'Synthetic streamflow (Qh) should have pd.DatatimeIndex.'
 
     
    # Calculate FDCs for total period and each realization
    nonexceedance = np.linspace(0.0001, 0.9999, 50)
    s_total_fdc = np.quantile(Qs.values.flatten(), nonexceedance)
    h_total_fdc = np.quantile(Qh.values.flatten(), nonexceedance) 
     
    s_fdc_max = np.zeros_like(nonexceedance)
    s_fdc_min = np.zeros_like(nonexceedance)
    h_fdc_max = np.zeros_like(nonexceedance)
    h_fdc_min = np.zeros_like(nonexceedance)
 
    annual_synthetics = Qs.groupby(Qs.index.year)
    annual_historic = Qh.groupby(Qh.index.year)
 
    for i, quant in enumerate(nonexceedance):
            s_fdc_max[i] = annual_synthetics.quantile(quant).max().max()
            s_fdc_min[i] = annual_synthetics.quantile(quant).min().min()
            h_fdc_max[i] = annual_historic.quantile(quant).max()
            h_fdc_min[i] = annual_historic.quantile(quant).min()
     
    ## Plotting
    is_subplot = False if ax is None else True
    if ax is None:
        fig, ax = plt.subplots(figsize = figsize, dpi=150)
     
    #for quant in syn_fdc_quants:
    ax.fill_between(nonexceedance, s_fdc_min, s_fdc_max, color = colors[1], label = 'Synthetic Annual FDC Range', alpha = 0.5)
    ax.fill_between(nonexceedance, h_fdc_min, h_fdc_max, color = colors[0], label = 'Historic Annual FDC Range', alpha = 0.3)
 
    ax.plot(nonexceedance, s_total_fdc, color = colors[1], label = 'Synthetic Total FDC', alpha = 1)
    ax.plot(nonexceedance, h_total_fdc, color = colors[0], label = 'Historic Total FDC', alpha = 1, linewidth = 2)
 
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_yscale(y_scale)
    ax.set_ylabel(f'Flow ({units})')
    ax.set_xlabel('Non-Exceedance Probability')
    ax.legend(fontsize= 10)
    ax.grid(True, linestyle='--', linewidth=0.5)
     
    ax.set_title(title_text)
    if savefig:
        plt.savefig(f'{fig_dir}/flow_duration_curves_{fname_addon}.png', dpi=200)
    return ax


###


def plot_correlation_heatmap(Qh, Qs_i,
                        timestep = 'daily',
                        savefig = False, fig_dir = '.',
                        figsize = (8,4), color_map = 'BuPu',
                        cbar_on = True):
    """Plots the side-by-side heatmaps of flow correlation between sites for historic and synthetic multi-site data.
     
    Args:
        Qh (pd.DataFrame): Historic daily streamflow timeseries at many different locations. The columns of Qh and Qs_i must match, and index must be pd.DatetimeIndex. 
        Qs (pd.DataFrame): Synthetic daily streamflow timeseries realizations. Each column is a unique realization. The columns of Qh and Qs_i must match, and index must be pd.DatetimeIndex.
        timestep (str, optional): The timestep which data should be aggregated over. Defaults to 'daily'. Options are 'daily', 'weekly', or 'monthly'.
        savefig (bool, optional): Allows for png to be saved to fig_dir. Defaults to False.
        fig_dir (str, optional): Location of saved figure output. Defaults to '.' (working directory).
        figsize (tuple, optional): The figure size. Defaults to (4,4).
        color_map (str, optional): The colormap used for the heatmaps. Defaults to 'BuPu.
        cbar_on (bool, optional): Indictor if the colorbar should be shown or not. Defaults to True.
    """
     
    ## Assertions
    assert(type(Qh) == type(Qs_i) and (type(Qh) == pd.DataFrame)), 'Both inputs Qh and Qs_i should be pd.DataFrames.'
    assert(np.mean(Qh.columns == Qs_i.columns) == 1.0), 'Historic and synthetic data should have same columns.'
     
    if timestep == 'monthly':
        Qh = Qh.resample('MS').sum()
        Qs_i = Qs_i.resample('MS').sum()
    elif timestep == 'weekly':
        Qh = Qh.resample('W-SUN').sum()
        Qs_i = Qs_i.resample('W-SUN').sum()
     
    ## Plotting
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi = 250)
     
    # Heatmap of historic correlation
    h_correlation = np.corrcoef(Qh.values.transpose())
    sns.heatmap(h_correlation, ax = ax1, 
                square = True, annot = False,
                xticklabels = 5, yticklabels = 5, 
                cmap = color_map, cbar = False, vmin = 0, vmax = 1)
 
    # Synthetic correlation
    s_correlation = np.corrcoef(Qs_i.values.transpose())
    sns.heatmap(s_correlation, ax = ax2,
                square = True, annot = False,
                xticklabels = 5, yticklabels = 5, 
                cmap = color_map, cbar = False, vmin = 0, vmax = 1)
     
    for axi in (ax1, ax2):
        axi.set_xticklabels([])
        axi.set_yticklabels([])
    ax1.set(xlabel = 'Site i', ylabel = 'Site j', title = 'Historic Streamflow')
    ax2.set(xlabel = 'Site i', ylabel = 'Site j', title = 'Synthetic Realization')
    title_string = f'Pearson correlation across different streamflow locations\n{timestep.capitalize()} timescale\n'
         
    # Adjust the layout to make room for the colorbar
    if cbar_on:
        plt.tight_layout(rect=[0, 0, 0.9, 0.85])
        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.6])  # [left, bottom, width, height]
        cbar = fig.colorbar(ax1.collections[0], cax=cbar_ax)
        cbar.set_label("Pearson Correlation")
    plt.suptitle(title_string, fontsize = 12)
     
    if savefig:
        plt.savefig(f'{fig_dir}/spatial_correlation_comparison_{timestep}.png', dpi = 250)
    plt.show()
    return


###

