import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, norm

def confidence_interval_pearson(r, n, alpha=0.05):
    """
    Calculate the confidence interval for a Pearson correlation coefficient using Fisher's Z transformation.
    
    Args:
    - r (float): Pearson correlation coefficient.
    - n (int): Sample size.
    - alpha (float): Significance level for two-tailed test, default 0.05 for 95% CI.
    
    Returns:
    - ci (tuple): Lower and upper bounds of the confidence interval for the correlation coefficient.
    """
    # Fisher Z transformation
    r_z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z = norm.ppf(1 - alpha / 2)
    
    z_low = r_z - z * se
    z_high = r_z + z * se
    
    # Transform back to correlation
    r_low = np.tanh(z_low)
    r_high = np.tanh(z_high)
    
    return (r_low, r_high)



def plot_autocorrelation(Qh, Qs, lag_range, 
                         timestep = 'daily',
                         savefig = False, fig_dir = '.',
                         figsize=(7, 5), colors = ['black', 'orange'],
                         alpha = 0.3):
    """Plot autocorrelation of historic and synthetic flow over some range of lags.
 
    Args:
        Qh (pd.Series): Historic daily streamflow timeseries. Index must be pd.DatetimeIndex. 
        Qs (pd.DataFrame): Synthetic daily streamflow timeseries realizations. Each column is a unique realization. Index must be pd.DatetimeIndex.
        lag_range (iterable): A list or range of lag values to be used for autocorrelation calculation.
         
        timestep (str, optional): The timestep which data should be aggregated over. Defaults to 'daily'. Options are 'daily', 'weekly', or 'monthly'.
        savefig (bool, optional): Allows for png to be saved to fig_dir. Defaults to False.
        fig_dir (str, optional): Location of saved figure output. Defaults to '.' (working directory).
        figsize (tuple, optional): The figure size. Defaults to (4,4).
        colors (list, optional): List of two colors for historic and synthetic data respectively. Defaults to ['black', 'orange'].
        alpha (float, optional): The opacity of synthetic data. Defaults to 0.3.
    """
     
    ## Assertions
    assert(type(Qs) == pd.DataFrame), 'Synthetic streamflow should be type pd.DataFrame.'
    assert(type(Qh.index) == pd.DatetimeIndex), 'Historic streamflow (Qh) should have pd.DatatimeIndex.'
    assert(type(Qs.index) == pd.DatetimeIndex), 'Synthetic streamflow (Qh) should have pd.DatatimeIndex.'
 
    if timestep == 'monthly':
        Qh = Qh.resample('MS').sum()
        Qs = Qs.resample('MS').sum()
        time_label = 'months'
    elif timestep == 'weekly':
        Qh = Qh.resample('W-SUN').sum()
        Qs = Qs.resample('W-SUN').sum()
        time_label = f'weeks'
    elif timestep == 'daily':
        time_label = f'days'
         
    # Calculate autocorrelations
    autocorr_h = np.zeros(len(lag_range))
    confidence_autocorr_h = np.zeros((2, len(lag_range)))
    for i, lag in enumerate(lag_range):
        n = len(Qh) - lag
        h_corr = pearsonr(Qh.values[:-lag], Qh.values[lag:])
        autocorr_h[i] = h_corr[0]
        ci_low, ci_high = confidence_interval_pearson(h_corr[0], n)
        confidence_autocorr_h[0,i] = h_corr[0] - ci_low
        confidence_autocorr_h[1,i] = ci_high - h_corr[0]
     
    autocorr_s = np.zeros((Qs.shape[1], len(lag_range)))
    for i, realization in enumerate(Qs.columns):
        autocorr_s[i] = [pearsonr(Qs[realization].values[:-lag], Qs[realization].values[lag:])[0] for lag in lag_range]
 
 
    ## Plotting
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
 
    # Plot autocorrelation for each synthetic timeseries
    for i, realization in enumerate(Qs.columns):
        if i == 0:
            ax.plot(lag_range, autocorr_s[i], alpha=1, color = colors[1], label=f'Synthetic Realization')
        else:
            ax.plot(lag_range, autocorr_s[i], color = colors[1], alpha=alpha, zorder = 1)
        ax.scatter(lag_range, autocorr_s[i], alpha=alpha, color = 'orange', zorder = 2)
 
    # Plot autocorrelation for the historic timeseries
    ax.plot(lag_range, autocorr_h, color=colors[0], linewidth=2, label='Historic', zorder = 3)
    ax.scatter(lag_range, autocorr_h, color=colors[0], zorder = 4)
    ax.errorbar(lag_range, autocorr_h, yerr=confidence_autocorr_h, 
                color='black', capsize=4, alpha=0.75, label ='Historic 95% CI')
 
    # Set labels and title
    ax.set_xlabel(f'Lag ({time_label})', fontsize=12)
    ax.set_ylabel('Autocorrelation (Pearson)', fontsize=12)
    ax.set_title('Autocorrelation Comparison\nHistoric Timeseries vs. Synthetic Ensemble', fontsize=14)
 
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', linewidth=0.5)
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.92)
 
    if savefig:
        plt.savefig(f'{fig_dir}/autocorrelation_plot_{timestep}.png', dpi=200, bbox_inches='tight')
    plt.show()
    return