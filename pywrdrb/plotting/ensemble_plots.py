"""
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize, LogNorm, ListedColormap
import hydroeval as he
import h5py
import datetime as dt



    
def get_ensemble_error_metrics(ensemble_results, observed_results, models, nodes):
    """
    Generate error metrics (NSE, KGE, correlation, bias, etc.) for 
    each realization in an ensemble at a specific node for specific model.

    Args:
        ensemble_results (dict): A dictionary containing dataframes of results.
        observed_results (pandas.DataFrame): A dataframe containing observed flow data.
        models (list): A list of model names (str) to compute error metrics for.
        nodes (list): A list of node names (str) to compute error metrics for.

    Returns:
        pd.DataFrame: A dataframe containing error metrics for the specified models and nodes.
    """
    assert('realization_' in list(ensemble_results[models[0]].keys())[0]), 'ensemble_results[model] must be a dict of Dfs with keys of the form "realization_#"'
    
    ensemble_results_metrics = {}
    
    ### compile error across models/nodes/ensemble/metrics
    for r in ensemble_results[m].keys():
        for j, node in enumerate(nodes):
            obs = observed_results[node]
            for i, m in enumerate(models):

                modeled = ensemble_results[m][r][node]

                ### only do models with nonzero entries (eg remove some weap)
                if np.sum(modeled) > 0:
                    ### get kge & nse
                    kge, r, alpha, beta = he.evaluator(he.kge, modeled, obs)
                    nse = he.evaluator(he.nse, modeled, obs)
                    logkge, logr, logalpha, logbeta = he.evaluator(he.kge, modeled, obs, transform='log')
                    lognse = he.evaluator(he.nse, modeled, obs, transform='log')

                    ### get Kolmogorov-Smirnov Statistic, & metric is 1 minus this (1 in ideal case, 0 in worst case)
                    kss, _ = stats.ks_2samp(modeled, obs)
                    kss = 1 - kss

                    resultsdict = {'nse': nse[0], 'kge': kge[0], 'r': r[0], 'alpha': alpha[0], 'beta': beta[0],
                                'lognse': lognse[0], 'logkge': logkge[0], 'logr': logr[0], 'logalpha': logalpha[0],
                                'logbeta': logbeta[0], 'kss': kss}

                    resultsdict['node'] = node
                    resultsdict['model'] = m
                    if i == 0 and j == 0:
                        results_metrics = pd.DataFrame(resultsdict, index=[0])
                    else:
                        results_metrics = results_metrics.append(pd.DataFrame(resultsdict, index=[0]))

        ensemble_results_metrics[r] = results_metrics.reset_index(inplace=True, drop=True)
        
    return results_metrics



def make_polar_plot(data, metric_names, ideal_score, filename, sub_title,
                    metric_mins,
                    r_max = 1.5,
                    inner_r = 0.5,
                    normalize = False,
                    cmap = 'rainbow',
                    color_by = 0,
                    brush_by = 0,
                    brush_condition = 'under',
                    brush_threshold = 1,
                    brush_alpha = 1,
                    scale_ideal = False,
                    plot_spokes = True,
                    buffer = 0.0,
                    cut_negatives = True,
                    show_legend = True,
                    figsize = (10,10),
                    line_width = 1,
                    line_alpha = 0.1):

    # Checks
    assert(data.shape[1] == len(metric_names)), 'Number of data columns != number of metric names.'
    #assert(len(ideal_score) == len(metric_names)), 'Length of ideal scores != number of metric names.'

    n_obs, n_metrics = data.shape
    n_spokes = n_metrics
    theta =  np.linspace(0, 2*np.pi, n_spokes)

    # Find the minimum and maximum achieved objective values
    data_mins = data.min(axis = 0)
    data_maxs = data.max(axis = 0)

    # Create a normalized data set
    if normalize:
        scaler = MinMaxScaler()
        scale_model = scaler.fit(np.vstack((data, metric_mins)))
        norm_data = scale_model.transform(data)
        norm_ideal = scale_model.transform(ideal_score)
    else:
        norm_data = data.copy()
        norm_ideal = ideal_score.copy()

    # Remove color_by and add the 1st metric to end
    stacked_norm_data = np.hstack((np.delete(norm_data, color_by, axis = 1), np.delete(norm_data, color_by, axis = 1)[:,0:1]))
    stacked_data = np.hstack((np.delete(data, color_by, axis = 1), np.delete(data, color_by, axis = 1)[:,0:1]))
    stacked_ideal = np.hstack((np.delete(norm_ideal, color_by, axis = 1), np.delete(norm_ideal, color_by, axis = 1)[:,0:1]))

    # Define the radial data - scaled according to norms
    r_data = np.zeros_like(stacked_norm_data)

    if scale_ideal:
        shift_ideal = np.ones(n_metrics) / norm_ideal
        shift_ideal = np.hstack((np.delete(shift_ideal, color_by), np.delete(shift_ideal, color_by)[0]))
        r_max = r_max + inner_r + buffer # + max(shift_ideal)
        for i in range(n_spokes):
            if shift_ideal[i] > 0:
                r_data[:,i] = (stacked_norm_data[:,i]) * shift_ideal[i] + inner_r + buffer
                stacked_ideal[:,i] = (stacked_ideal[:,i]) * shift_ideal[i] + inner_r + buffer
            else:
                r_data[:,i] = (stacked_norm_data[:,i]) + 1 + inner_r+ buffer
                stacked_ideal[:,i] = (stacked_ideal[:,i]) + 1 + inner_r+ buffer

    else:
        r_data = stacked_norm_data + inner_r + buffer
        stacked_ideal = stacked_ideal + inner_r + buffer
        r_max = r_max + inner_r + buffer


    if cut_negatives:
        r_data[np.argwhere(r_data<0)] = 0

    # Initialize plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    cmap = plt.cm.get_cmap(cmap)

    # Plot inner and outer ring
    ax.plot(theta, np.array([r_max]*n_spokes), color = 'grey', alpha = 0.01)
    ax.plot(theta, np.array([inner_r]*n_spokes), color = 'grey', alpha = 0.3)

    # Plot spokes
    if plot_spokes:
        for s in range(n_spokes):
            ax.plot(np.array([theta[s], theta[s]]), np.array([inner_r, max(r_data[:,s])]), color = 'grey', alpha = 0.3)
            ax.plot(np.array([theta[s], theta[s]]), np.array([max(r_data[:,s]), r_max]), color = 'grey', alpha = 0.3, linestyle = 'dashed')


    # Plot all observations
    brush_counter = 0
    for i in range(n_obs):
        if brush_condition == 'under':
            brush_header = f'Brush criteria: {metric_names[brush_by]} < {brush_threshold}'
            if data[i,brush_by] <= brush_threshold:
                a = brush_alpha
                ci = cmap(norm_data[i, color_by])
                brush_counter += 1
            else:
                a = line_alpha
                ci = 'k'
        elif brush_condition == 'over':
            brush_header = f'Brush criteria: {metric_names[brush_by]} > {brush_threshold}'
            if data[i,brush_by] >= brush_threshold:
                a = brush_alpha
                ci = cmap(norm_data[i, color_by])
                brush_counter += 1
            else:
                a = line_alpha
                ci = 'k'
        else:
            print('Invalid brush_condition. Options are "under" or "over".')

        ax.plot(theta, r_data[i, :], c = ci, linewidth = line_width, alpha = a)

    # Plot ideal
    ax.plot(theta, stacked_ideal[0,:], c = 'k', linewidth = 2, linestyle = 'dashed', label = 'Ideal')

    # Add colorbar
    cb = plt.cm.ScalarMappable(cmap = cmap)
    cb.set_array([data_mins[color_by], data_maxs[color_by]])
    cbar = fig.colorbar(cb, anchor = (2.5, 0), pad = 0.05)
    cbar.ax.set_ylabel(metric_names[color_by], fontsize = 16)

    # Add legend
    if show_legend == True:
        ax.legend(bbox_to_anchor = (1.2, 1))

    # Make radial labels
    spoke_maxs = np.max(stacked_data, axis = 0)
    spoke_labs = np.delete(metric_names, color_by)
    #outter_radial_labels = [f'{spoke_maxs[i]}\n{spoke_labs[i]}' for i in range(len(spoke_labs))]
    outter_radial_labels = np.delete(metric_names, color_by)

    # Add text for brush condition
    brush_text = str(brush_header + f'\nn = {brush_counter} of {n_obs}')
    ax.text((3/2)*np.pi, r_max+0.4*r_max, brush_text, verticalalignment='bottom', horizontalalignment='center',
            color='k', fontsize=14)

    # Add lower bound values
    actual_minimums = np.min(r_data, axis = 0)
    spoke_min_labels = np.delete(metric_mins, color_by)
    for s in range(n_spokes - 1):
        ax.text(theta[s], inner_r - 0.3*inner_r, f'{spoke_min_labels[s]:.1f}', horizontalalignment = 'center',
                verticalalignment='center', color = 'k', fontsize = 10)

    # Graphic features
    ax.set_rmax(r_max)
    ax.set_rticks([])  # Less radial ticks
    ax.spines['polar'].set_visible(False)
    ax.set_rlabel_position(-50.5)  # Move radial labels away from plotted line
    ax.set_title(f'Prediction Performance Metrics\n{sub_title}', va='bottom', fontsize = 15)
    ax.set_xticklabels(outter_radial_labels, fontsize = 16)
    ax.set_xticks(theta)
    ax.grid(False)
    fig.set_size_inches(figsize)
    fig.set_dpi(200)
    plt.show()
    return plt
