"""
Contains all plotting functions used for Pywr-DRB model assessments, including:


"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
from scipy import stats
import sys

import hydroeval as he
import h5py
import datetime as dt

from pywrdrb.pywr_drb_node_data import upstream_nodes_dict, downstream_node_lags, immediate_downstream_nodes_dict

# Custom modules
from pywrdrb.utils.constants import cms_to_mgd, cm_to_mg, cfs_to_mgd
from pywrdrb.utils.lists import reservoir_list, reservoir_list_nyc, majorflow_list, reservoir_link_pairs
from pywrdrb.utils.directories import input_dir, fig_dir, output_dir, model_data_dir

from pywrdrb.plotting.styles import base_model_colors, model_hatch_styles, paired_model_colors, scatter_model_markers




### function to return subset of dates for timeseries data
def subset_timeseries(timeseries, start_date, end_date):
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

    if start_date is not None:
        timeseries = timeseries.loc[start_date:]
    if end_date is not None:
        timeseries = timeseries.loc[:end_date]
    return timeseries



### 3-part figure to visualize flow: timeseries, scatter plot, & flow duration curve. Can plot observed plus 1 or 2 modeled series.
def plot_3part_flows(results, models, node, 
                     colordict = paired_model_colors, markerdict = scatter_model_markers, start_date=None, end_date=None,
                     uselog=False, save_fig=True, fig_dir = fig_dir):
    """
    Plots a 3-part figure to visualize flow data, including a timeseries plot, a scatter plot, and a flow duration curve.
    
    Args:
        results (dict): A dictionary containing the flow data, including observed and modeled flows.
        models (list): A list of model names to plot. It can contain one or two model names.
        node (str): The name of the node or location for which the flows are plotted.
        colordict (dict, optional): A dictionary mapping model names to color codes for line and scatter plots.
            Defaults to paired_model_colors.
        markerdict (dict, optional): A dictionary mapping model names to marker codes for scatter plots.
            Defaults to scatter_model_markers.
        uselog (bool, optional): Determines whether logarithmic scale is used for plotting. Defaults to False.
        save_fig (bool, optional): Determines whether to save the figure as a PNG file. Defaults to True.
        fig_dir (str, optional): The directory to save the figure. Defaults to fig_dir.

    Returns:
        None
    """
    
    use2nd = True if len(models) > 1 else False
    fig = plt.figure(figsize=(16, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=(2, 1, 1), wspace=0.25, hspace=0.3)

    obs = subset_timeseries(results['obs'][node], start_date, end_date)

    ### first fig: time series of observed & modeled flows
    ax = fig.add_subplot(gs[0, 0])
    for i, m in enumerate(models):
        if use2nd or i == 0:
            ### first plot time series of observed vs modeled
            modeled = subset_timeseries(results[m][node], start_date, end_date)

            if i == 0:
                ax.plot(obs, label='observed', color=colordict['obs'])
            ax.plot(modeled, label=m, color=colordict[m])

            ### get error metrics
            if uselog:
                kge, r, alpha, beta = he.evaluator(he.kge, modeled, obs, transform='log')
                nse = he.evaluator(he.nse, modeled, obs, transform='log')
            else:
                kge, r, alpha, beta = he.evaluator(he.kge, modeled, obs)
                nse = he.evaluator(he.nse, modeled, obs)
            nse, kge, r, alpha, beta = round(nse[0], 2), round(kge[0], 2), round(r[0], 2), round(alpha[0], 2), round(beta[0], 2)

            ### clean up fig
            if i == 0:
                coords = (0.04, 0.94)
            else:
                coords = (0.04, 0.88)
            ax.annotate(f'NSE={nse}; KGE={kge}: r={r}, relvar={alpha}, bias={beta}', xy=coords, xycoords=ax.transAxes,
                        color=colordict[m])
            ax.legend(loc='right')
            ax.set_ylabel('Daily flow (MGD)')
            ax.set_xlabel('Date')
            if uselog:
                ax.semilogy()

    ### second fig: scatterplot of observed vs modeled flow
    ax = fig.add_subplot(gs[0, 1])
    for i, m in enumerate(models):
        ### now add scatter of observed vs modeled
        if use2nd or i == 0:
            ### first plot time series of observed vs modeled
            modeled = subset_timeseries(results[m][node], start_date, end_date)

            ax.scatter(obs, modeled, alpha=0.25, zorder=2, color=colordict[m], marker='x' if 'pywr' in m else 'o')
            diagmax = min(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([0, diagmax], [0, diagmax], color='k', ls='--')
            if uselog:
                ax.loglog()
            ax.set_xlabel('Observed flow (MGD)')
            ax.set_ylabel('Modeled flow (MGD)')

    ### third fig: flow duration curves
    ax = fig.add_subplot(gs[0, 2])
    for i, m in enumerate(models):
        if use2nd or i == 0:
            ### now add exceedance plot
            def plot_exceedance(data, ax, color, **kwargs):
                df = data.sort_values()
                exceedance = np.arange(1., len(df) + 1.) / len(df)
                ax.plot(exceedance, df, color=color, **kwargs)

            modeled = subset_timeseries(results[m][node], start_date, end_date)

            plot_exceedance(obs, ax, color = colordict['obs'])
            ax.semilogy()
            ax.set_xlabel('Non-exceedence')
            ax.set_ylabel('Daily flow (log scale, MGD)')

            plot_exceedance(modeled, ax, color = colordict[m])

    # plt.show()
    if save_fig:
        if use2nd:
            fig.savefig(f'{fig_dir}streamflow_3plots_{models[0]}_{models[1]}_{node}.png', bbox_inches='tight', dpi = 250)
        else:
            fig.savefig(f'{fig_dir}streamflow_3plots_{models[0]}_{node}.png', bbox_inches='tight', dpi = 250)
        plt.close()
    return


### 
def plot_weekly_flow_distributions(results, models, node, colordict= paired_model_colors, fig_dir = fig_dir,
                                   start_date=None, end_date=None):
    """
    Plot distributions (range and median) of weekly flows for 1 or 2 model simulation results.
        
    Args:
        results (dict): A dictionary containing the flow data, including observed and modeled flows.
        models (list): A list of model names to plot. It can contain one or two model names.
        node (str): The name of the node or location for which the flows are plotted.
        colordict (dict, optional): A dictionary mapping model names to color codes for line and scatter plots.
            Defaults to paired_model_colors.
        markerdict (dict, optional): A dictionary mapping model names to marker codes for scatter plots.
            Defaults to scatter_model_markers.
        fig_dir (str, optional): The directory to save the figure. Defaults to fig_dir.

    Returns:
        None
    """
    use2nd = True if len(models) > 1 else False

    fig = plt.figure(figsize=(16, 4))
    gs = fig.add_gridspec(1, 2, wspace=0.15, hspace=0.3)

    obs = subset_timeseries(results['obs'][node], start_date, end_date)

    obs_resample = obs.resample('W').sum()
    nx = len(obs_resample.groupby(obs_resample.index.week).max())
    ymax = obs_resample.groupby(obs_resample.index.week).max().max()
    ymin = obs_resample.groupby(obs_resample.index.week).min().min()
    for i, m in enumerate(models):
        modeled = subset_timeseries(results[m][node], start_date, end_date)
        modeled_resample = modeled.resample('W').sum()
        ymax = max(ymax, modeled_resample.groupby(modeled_resample.index.week).max().max())
        ymin = min(ymin, modeled_resample.groupby(modeled_resample.index.week).min().min())
    
    ### first plot time series of observed vs modeled, real scale
    for i, m in enumerate(models):
        
        if i == 0:
            ax = fig.add_subplot(gs[0, 0])
            ax.fill_between(np.arange(1, (nx+1)), obs_resample.groupby(obs_resample.index.week).max(),
                            obs_resample.groupby(obs_resample.index.week).min(), color=colordict['obs'], alpha=0.4)
            ax.plot(obs_resample.groupby(obs_resample.index.week).mean(), label='observed', color=colordict['obs'])

        modeled = subset_timeseries(results[m][node], start_date, end_date)
        modeled_resample = modeled.resample('W').sum()
        ax.fill_between(np.arange(1, (nx+1)), modeled_resample.groupby(modeled_resample.index.week).max(),
                        modeled_resample.groupby(modeled_resample.index.week).min(), color=colordict[m], alpha=0.4)
        ax.plot(modeled_resample.groupby(modeled_resample.index.week).mean(), label=m, color=colordict[m])

        ax.legend(loc='upper right')
        ax.set_ylabel('Weekly flow (MGW)')
        ax.set_xlabel('Week')
        ax.set_ylim([-0.1 * ymax, ymax * 1.1])

    ### now repeat, log scale
    for i, m in enumerate(models):
        if i == 0:
            ax = fig.add_subplot(gs[0, 1])
            ax.fill_between(np.arange(1, (nx+1)), obs_resample.groupby(obs_resample.index.week).max(),
                            obs_resample.groupby(obs_resample.index.week).min(), color=colordict['obs'], alpha=0.4)
            ax.plot(obs_resample.groupby(obs_resample.index.week).mean(), label='observed', color=colordict['obs'])

        modeled = subset_timeseries(results[m][node], start_date, end_date)
        modeled_resample = modeled.resample('W').sum()
        ax.fill_between(np.arange(1, (nx+1)), modeled_resample.groupby(modeled_resample.index.week).max(),
                        modeled_resample.groupby(modeled_resample.index.week).min(), color=colordict[m], alpha=0.4)
        ax.plot(modeled_resample.groupby(modeled_resample.index.week).mean(), label=m, color=colordict[m])

        ax.set_ylim([max(ymin * 0.5, 0.01), ymax * 1.5])
        ax.set_xlabel('Week')

        ax.semilogy()

    # plt.show()
    if use2nd:
        fig.savefig(f'{fig_dir}streamflow_weekly_{models[0]}_{models[1]}_{node}.png', bbox_inches='tight', dpi = 250)
    else:
        fig.savefig(f'{fig_dir}streamflow_weekly_{models[0]}_{node}.png', bbox_inches='tight', dpi = 250)
    plt.close()
    return




###
def get_error_metrics(reservoir_downstream_gages, major_flows, models, nodes, start_date=None, end_date=None):
    """
    Generate error metrics (NSE, KGE, correlation, bias, etc.) for a specific model and node.

    Args:
        results (dict): A dictionary containing dataframes of results.
        models (list): A list of model names (str) to compute error metrics for.
        nodes (list): A list of node names (str) to compute error metrics for.

    Returns:
        pd.DataFrame: A dataframe containing error metrics for the specified models and nodes.
    """
    ### compile error across models/nodes/metrics
    for j, node in enumerate(nodes):
        if node in reservoir_list:
            results = reservoir_downstream_gages
        else:
            results = major_flows

        obs = subset_timeseries(results['obs'][node], start_date, end_date)
        for i, m in enumerate(models):
            modeled = subset_timeseries(results[m][node], start_date, end_date)

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

    results_metrics.reset_index(inplace=True, drop=True)
    return results_metrics




### radial plots across diff metrics/reservoirs/models.
### following galleries here https://www.python-graph-gallery.com/circular-barplot-with-groups
def plot_radial_error_metrics(results_metrics, radial_models, nodes, useNonPep = True, useweap = True, usepywr = True,
                              usemajorflows=False, fig_dir = fig_dir,
                              colordict = base_model_colors, hatchdict = model_hatch_styles):
    """
    Plot radial error metrics for different models, nodes, and metrics.

    Args:
        results_metrics (pd.DataFrame): Dataframe containing error metrics.
        radial_models (list): List of model names (str) to include in the plot.
        nodes (list): List of node names (str) to include in the plot.
        useNonPep (bool): Whether to include non-pepacton nodes in the plot (default: True).
        useweap (bool): Whether to include WEAP models in the plot (default: True).
        usepywr (bool): Whether to include PyWR models in the plot (default: True).
        usemajorflows (bool): Whether to use major flows in the plot (default: False).
        fig_dir (str): Directory to save the generated figure (default: fig_dir).
        colordict (dict): Dictionary mapping model names to colors (default: base_model_colors).
        hatchdict (dict): Dictionary mapping model names to hatch styles (default: model_hatch_styles).

    Returns:
        None
    """

    fig, axs = plt.subplots(2, 4, figsize=(16, 8), subplot_kw={"projection": "polar"})

    metrics = ['nse', 'kge', 'r', 'alpha', 'beta', 'kss', 'lognse', 'logkge']

    nodelabeldict = {'pepacton': 'Pep', 'cannonsville': 'Can', 'neversink': 'Nev', 'prompton': 'Pro', 'assunpink': 'AspRes',\
                    'beltzvilleCombined': 'Bel', 'blueMarsh': 'Blu', 'mongaupeCombined': 'Mgp', 'fewalter': 'FEW',\
                    'delLordville':'Lor', 'delMontague':'Mtg', 'delTrenton':'Tre', 'outletAssunpink':'Asp', \
                     'outletSchuylkill':'Sch'}
    titledict = {'nse': 'NSE', 'kge': 'KGE', 'r': 'Correlation', 'alpha': 'Relative STD', 'beta': 'Relative Bias',
                 'kss': 'K-S Statistic', 'lognse': 'LogNSE', 'logkge': 'LogKGE'}

    for k, metric in enumerate(metrics):
        row = k % 2
        col = int(k / 2)
        ax = axs[row, col]

        pad = 1
        groups = len(nodes)
        angles = np.linspace(0, 2 * np.pi, len(nodes) * len(radial_models) + pad * groups, endpoint=False)
        values = np.maximum(np.minimum(results_metrics[metric], 3), -1) - 1

        labels = [node + '_' + model for node, model in zip(results_metrics['node'], results_metrics['model'])]
        colors = [colordict[model] for model in results_metrics['model']]
        if not usepywr:
            if not useweap:
                mask = [m in radial_models[-2:] for m in results_metrics['model']]
            else:
                mask = [m in radial_models[-3:] for m in results_metrics['model']]
            colors = [v if m else 'none' for m, v in zip(mask, colors)]
        if not useNonPep:
            mask = [r == 'pepacton' for r in results_metrics['node']]
            colors = [v if m else 'none' for m, v in zip(mask, colors)]

        edges = ['w' for model in results_metrics['model']]
        # edges = [edgedict[model] for model in results_metrics['model']]
        hatches = [hatchdict[model] for model in results_metrics['model']]

        width = 2 * np.pi / len(angles)

        ### Obtaining the right indexes is now a little more complicated
        offset = 0
        idxs = []
        groups_size = [len(radial_models)] * len(nodes)
        for size in groups_size:
            idxs += list(range(offset + pad, offset + size + pad))
            offset += size + pad

        ### Remove all spines
        ax.set_frame_on(False)

        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        ### Set limits for radial (y) axis. The negative lower bound creates the whole in the middle.
        if metric in ['fdcm', 'r', 'kss']:
            ax.set_ylim(-1, 0.2)
            yrings = [-1, -0.25, -0.5, -0.75, 0]
        elif metric in ['alpha', 'beta']:
            #         ax.set_ylim(-1, 2.2)
            yrings = [-1, -0.5, 0, 0.5, 1]
        elif metric in ['kge', 'nse', 'logkge', 'lognse']:
            ax.set_ylim(-2, 0.2)
            yrings = [-2, -1.5, -1, -0.5, 0]

        # Add reference lines
        x2 = np.linspace(0, 2 * np.pi, num=50)
        for j, y in enumerate(yrings):
            if y == 0:
                ax.plot(x2, [y] * 50, color="#333333", lw=1.5, zorder=3)

            ax.plot(x2, [y] * 50, color="0.8", lw=0.8, zorder=1)
            if (np.abs(y - int(y)) < 0.001):
                ax.text(0, y, round(y + 1, 2), color="#333333", fontsize=12, fontweight="bold", ha="left", va="center")

        for j in range(groups):
            angle = 2 * np.pi / groups * j
            ax.plot([angle, angle], [yrings[0], yrings[-1] + 0.1], color='0.8', zorder=1)

        ### Add bars
        ax.bar(angles[idxs], values, width=width, linewidth=0.5, color=colors, hatch=hatches, edgecolor=edges, zorder=2)

        ### customization to add group annotations
        offset = 0
        for j, node in enumerate(nodes):
            # Add line below bars
            x1 = np.linspace(angles[offset + pad], angles[offset + size + pad - 1], num=50)

            # Add text to indicate group
            wedge = 360 / len(nodes)
            rotation = -90 + wedge / 2 + wedge * j
            if j >= 3:
                rotation += 180
            if useNonPep or node == 'pepacton':
                fontcolor = "#333333"
            else:
                fontcolor = "w"

            ax.text(
                np.mean(x1), ax.get_ylim()[1] + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]), nodelabeldict[node],
                color=fontcolor, fontsize=14,
                ha="center", va="center", rotation=rotation
            )

            offset += size + pad

        ax.text(np.pi / 2, ax.get_ylim()[1] + 0.18 * (ax.get_ylim()[1] - ax.get_ylim()[0]), titledict[metric],
                color="#333333", fontsize=16,
                fontweight="bold", ha="center", va="center")

    legend_elements = []
    legend_elements.append(Line2D([0], [0], color='none', label='models'))
    for m in radial_models[::-1]:
        if usepywr or m in radial_models[-2:] or (useweap and m == radial_models[-3]):
            legend_elements.append(Patch(facecolor=colordict[m], edgecolor='w', label=m, hatch=hatchdict[m]))
        else:
            legend_elements.append(Patch(facecolor='w', edgecolor='w', label=m, hatch=hatchdict[m]))

    leg = plt.legend(handles=legend_elements, loc='center', bbox_to_anchor=(1.5, 1.1), frameon=False)
    for i, text in enumerate(leg.get_texts()):
        if not usepywr and i > 3:
            text.set_color("w")
        elif not useweap and i > 2:
            text.set_color('w')

    if usemajorflows:
        filename_mod = 'combinedNodeTypes'
    else:
        filename_res = 'allRes' if useNonPep else 'pep'
        if usepywr:
            filename_mod = 'allMod_withPywr'
        elif useweap:
            filename_mod = 'allMod_withoutPywr'
        else:
            filename_mod = 'NhmNwm_withoutPywr'
        filename_mod = filename_res + '_' + filename_mod

    # plt.show()
    fig.savefig(f'{fig_dir}/radialMetrics_{filename_mod}.png', bbox_inches='tight', dpi=300)
    plt.close()
    return


###
def get_RRV_metrics(results, models, nodes, start_date=None, end_date=None):
    """
    Calculate measures of reliability, resilience, and vulnerability based on Hashimoto et al. (1982) WRR.

    Args:
        results (dict): Dictionary containing model results for different nodes.
        models (list): List of model names (str) to include in the analysis.
        nodes (list): List of node names (str) to include in the analysis.

    Returns:
        pd.DataFrame: DataFrame containing reliability, resiliency, and vulnerability metrics for each model and node.
    """
    thresholds = {'delMontague': 1131.05, 'delTrenton': 1938.950669}  ### FFMP flow targets (MGD)
    eps = 1e-9
    thresholds = {k: v - eps for k, v in thresholds.items()}
    for j, node in enumerate(nodes):
        for i, m in enumerate(models):
            modeled = subset_timeseries(results[m][node], start_date, end_date)

            ### only do models with nonzero entries (eg remove some weap)
            if np.sum(modeled) > 0:

                ### reliability is the fraction of time steps above threshold
                reliability = (modeled > thresholds[node]).mean()
                ### resiliency is the probability of recovering to above threshold if currently under threshold
                if reliability < 1 - eps:
                    resiliency = np.logical_and((modeled.iloc[:-1] < thresholds[node]).reset_index(drop=True), \
                                                (modeled.iloc[1:] >= thresholds[node]).reset_index(drop=True)).mean() / \
                                 (1 - reliability)
                else:
                    resiliency = np.nan
                ### vulnerability is the expected maximum severity of a failure event
                if reliability > eps:
                    max_shortfalls = []
                    max_shortfall = 0
                    in_event = False
                    for i in range(len(modeled)):
                        v = modeled.iloc[i]
                        if v < thresholds[node]:
                            in_event = True
                            s = thresholds[node] - v
                            max_shortfall = max(max_shortfall, s)
                        else:
                            if in_event:
                                max_shortfalls.append(max_shortfall)
                                in_event = False
                    vulnerability = np.mean(max_shortfalls)
                else:
                    vulnerability = np.nan

                resultsdict = {'reliability': reliability, 'resiliency': resiliency, 'vulnerability': vulnerability}

                resultsdict['node'] = node
                resultsdict['model'] = m
                try:
                    rrv_metrics = rrv_metrics.append(pd.DataFrame(resultsdict, index=[0]))
                except:
                    rrv_metrics = pd.DataFrame(resultsdict, index=[0])

    rrv_metrics.reset_index(inplace=True, drop=True)
    return rrv_metrics




### 
def plot_rrv_metrics(rrv_metrics, rrv_models, nodes, fig_dir = fig_dir,
                     colordict = base_model_colors, hatchdict = model_hatch_styles):
    """
    Plot histograms of reliability, resiliency, and vulnerability for different models and nodes.

    Args:
        rrv_metrics (pd.DataFrame): DataFrame containing reliability, resiliency, and vulnerability metrics.
        rrv_models (list): List of model names (str) to include in the plot.
        nodes (list): List of node names (str) to include in the plot.
        fig_dir (str): Directory to save the figure (optional).
        colordict (dict): Dictionary mapping model names to color codes (optional).
        hatchdict (dict): Dictionary mapping model names to hatch styles (optional).

    Returns:
        None
    """

    fig, axs = plt.subplots(2, 3, figsize=(16, 8))

    metrics = ['reliability','resiliency','vulnerability']
    
    for n, node in enumerate(nodes):
        for k, metric in enumerate(metrics):
            ax = axs[n, k]

            colors = [colordict[model] for model in rrv_models]
            hatches = [hatchdict[model] for model in rrv_models]
            heights = [rrv_metrics[metric].loc[np.logical_and(rrv_metrics['node']==node, rrv_metrics['model']==model)].iloc[0] for model in rrv_models]
            positions = range(len(heights))

            ### Add bars
            ax.bar(positions, heights, width=0.8, linewidth=0.5, color=colors, hatch=hatches, edgecolor='w', zorder=2)

            ax.set_xlim([-0.5, positions[-1]+0.5])
            if k == 0:
                ax.set_ylim([0.8, 1.])
            if n>0:
                ax.set_xticks(positions, rrv_models, rotation=90)
            else:
                ax.set_xticks(positions, ['']*len(positions))
                ax.set_title(metric)

    legend_elements = []
    legend_elements.append(Line2D([0], [0], color='none', label='models'))
    for m in rrv_models:
        legend_elements.append(Patch(facecolor=colordict[m], edgecolor='w', label=m, hatch=hatchdict[m]))
    leg = plt.legend(handles=legend_elements, loc='center', bbox_to_anchor=(1.5, 1.1), frameon=False)

    fig.savefig(f'{fig_dir}/rrv_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    return



def plot_flow_contributions(reservoir_releases, major_flows, inflows, model, node, start_date=None, end_date=None,
                            upstream_nodes_dict = upstream_nodes_dict,
                            downstream_node_lags= downstream_node_lags, 
                            reservoir_list = reservoir_list,
                            log_flows=True,
                            smoothing=True, smoothing_window=7,
                            fig_dir = fig_dir,
                            ):
    """
    Plot flow contributions at a specific node for a given model.

    Args:
        reservoir_releases (dict): Dictionary of reservoir releases data for different models.
        major_flows (dict): Dictionary of major flows data.
        model (str): Name of the model.
        node (str): Name of the node.
        start_date (str): Start date of the plot in 'YYYY-MM-DD' format.
        end_date (str): End date of the plot in 'YYYY-MM-DD' format.
        upstream_nodes_dict (dict): Dictionary mapping nodes to their upstream contributing nodes (optional).
        reservoir_list (list): List of reservoir names (optional).
        majorflow_list (list): List of major flow names (optional).
        percentage_flow (bool): Whether to plot flow contributions as percentages (optional).
        plot_target (bool): Whether to plot the flow target line (optional).
        fig_dir (str): Directory to save the figure (optional).
        input_dir (str): Directory to load input data (optional).

    Returns:
        None
    """

    mainstem_nodes= ['delLordville', 'delMontague', 'delDRCanal', 'delTrenton',
                     'outletAssunpink', 'outletSchuylkill']
    
    # Get contributions
    contributing = upstream_nodes_dict[node]
    non_nyc_reservoirs = [i for i in contributing if (i in reservoir_list) and (i not in reservoir_list_nyc)]
    
    use_releases = [i for i in contributing if i in reservoir_list]
    use_inflows = [i for i in contributing if (i in mainstem_nodes)]
    if node == 'delMontague':
        use_inflows.append('delMontague')

    title_text = 'Contributing flows at Trenton' if (node == 'delTrenton') else 'Contributing flows at Montague'
    if node == 'delMontague':
        target = 1750*cfs_to_mgd
    elif node == 'delTrenton':
        target = 3000*cfs_to_mgd
    else:
        print('Invalid node specification. Options are "delMontague" and "delTrenton"')

    ## Pull just contributing data
    release_contributions = reservoir_releases[model][use_releases]
    inflow_contributions = inflows[model][use_inflows]
    contributions = pd.concat([release_contributions, inflow_contributions], axis=1)

    # Impose lag
    for c in upstream_nodes_dict[node][::-1]:
        if c in contributions.columns:
            lag= downstream_node_lags[c]
            downstream_node = immediate_downstream_nodes_dict[c]
            
            while downstream_node not in ['delDRCanal', 'delTrenton', 'output_del']:
                if node == 'delDRCanal':
                    break
                lag += downstream_node_lags[downstream_node]
                downstream_node = immediate_downstream_nodes_dict[downstream_node]
                
            if lag > 0:
                contributions[c].iloc[lag:] = contributions[c].iloc[:-lag]
            
    contributions = subset_timeseries(contributions, start_date, end_date)
    
    # Get total sim and obs flow    
    total_obs_node_flow = subset_timeseries(major_flows['obs'][node], start_date, end_date)
    total_sim_node_flow = subset_timeseries(major_flows[model][node], start_date, end_date)
    
    #TODO: Find source of unaccounted flow
    unaccounted_flow = (total_sim_node_flow - contributions.sum(axis=1)).divide(total_sim_node_flow, axis=0)*100 
    
    contributions = contributions.divide(total_sim_node_flow, axis =0) * 100
    contributions[contributions<0] = 0
    
    ## Plotting
    nyc_color = 'midnightblue'
    other_reservoir_color = 'darkcyan'
    upstream_inflow_color = 'lightsteelblue'
    obs_flow_color = 'red'

    fig, axes = plt.subplots(nrows=2, ncols=1, 
                           figsize=(8, 5), dpi =200,
                           sharex=True, 
                           gridspec_kw={'height_ratios': [1, 1.5], 'wspace': 0.05})
    ax1= axes[0]
    ax2= axes[1]
    
    ts = contributions.index
    
    B = contributions[use_inflows].sum(axis=1) + unaccounted_flow
    C = contributions[non_nyc_reservoirs].sum(axis=1) + B
    D = contributions[reservoir_list_nyc].sum(axis=1) + C
    if smoothing:
        B = B.rolling(window=smoothing_window).mean()
        C = C.rolling(window=smoothing_window).mean()
        D = D.rolling(window=smoothing_window).mean()
        
        smooth_total_sim_node_flow = total_sim_node_flow.rolling(window=7).mean()
    
    # Total flows and target flow
    ax1.hlines(target, ts[0], ts[-1], linestyle = 'dotted', color = 'maroon', alpha = 0.85, label = f'Flow target {target:.0f} (MGD)')
    ax1.plot(ts, smooth_total_sim_node_flow.loc[ts], color = 'dodgerblue', label = 'Sim. Flow')
    ax1.plot(ts, total_obs_node_flow.loc[ts], color = 'black', ls='dashed', label = 'Obs. Flow')
    # ax1.fill_between(ts, total_sim_node_flow.loc[ts], target, where=(total_sim_node_flow.loc[ts] < target), color='red', alpha=0.5)
    
    ax1.set_ylabel('Flow (MGD)', fontsize=14)
    if log_flows:
        ax1.set_yscale('log')
    ax1.set_ylim([1000,100000])

    # plot percent contribution
    # ax.fill_between(ts, A, color = node_inflow_color, label = 'Direct node inflow')

    ax2.fill_between(ts, B, color = upstream_inflow_color, label = 'Unmanaged Flows')
    ax2.fill_between(ts, C, B, color = other_reservoir_color, label = 'Non-NYC Reservoir Releases')
    ax2.fill_between(ts, D, C, color = nyc_color, label = 'NYC Reservoir Releases')
    
    ax2.set_ylabel('Contributions (%)', fontsize=14)
    ax2.set_ylim([0,100])
    
    # Create legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.0, 0.9))

    title = f'{fig_dir}/flow_contributions_{node}_{model}_{contributions.index.year[0]}_{contributions.index.year[-1]}'

    plt.xlim([contributions.index[0], contributions.index[-1]])
    plt.xlabel('Date')
    fig.align_labels()
    plt.savefig(f'{title}.png', bbox_inches='tight', dpi=300)
    plt.close()
    return



def compare_inflow_data(inflow_data, nodes, models, start_date = None, end_date = None, fig_dir = fig_dir):
    """Generates a boxplot comparison of inflows are specific nodes for different datasets.

    Args:
        inflow_data (dict): Dictionary containing pd.DataFrames with inflow data. 
        nodes (list): List of nodes with inflows.
        fig_dir (str, optional): Folder to save figures. Defaults to 'figs/'.
    """

    results = {}
    for m in models:
        results[m] = subset_timeseries(inflow_data[m].loc[:,nodes], start_date, end_date)
        results[m] = results[m].assign(Dataset=m)

    cdf = pd.concat([results[m] for m in models])
    mdf = pd.melt(cdf, id_vars=['Dataset'], var_name=['Node'])
    mdf.value.name = "Inflow (MGD)"
    
    plt.figure(figsize=(15,7))
    ax = sns.boxplot(x="Node", y="value", hue="Dataset", data=mdf, 
                    showfliers=False, linewidth=1.2, saturation=0.8)
    ax.set(ylim=(1, 100000))
    ax.tick_params(axis='x', rotation=90)    
    for patch in ax.artists:
        r,g,b,a = patch.get_facecolor()
        patch.set_edgecolor((0,0,0,.0))
        patch.set_facecolor((r,g,b,.0))
    plt.yscale('log')
    plt.savefig(f'{fig_dir}inflow_comparison_boxplot.png', bbox_inches='tight', dpi=250)
    plt.close()
    return




def plot_combined_nyc_storage(storages, releases, all_drought_levels, models,
                      start_date = '1999-10-01',
                      end_date = '2010-05-31',
                      colordict = base_model_colors,
                      use_percent = True,
                      plot_observed=True, plot_sim=True,
                      add_ffmp_levels=True, ffmp_levels_to_plot=[2,5],
                      plot_drought_levels = True, 
                      smooth_releases=True, smooth_window=7,
                      plot_releases = True, 
                      fig_dir=fig_dir, filename_addon=""):
    """
    Plot simulated and observed combined NYC reservoir storage.

    Args:
        storages (dict): Dictionary of storage results from `get_pywr_results`.
        releases (dict): Dictionary of release data.
        models (list): List of models to plot.
        start_date (str): Start date of the plot in 'YYYY-MM-DD' format.
        end_date (str): End date of the plot in 'YYYY-MM-DD' format.
        colordict (dict): Dictionary mapping model names to colors (optional).
        use_percent (bool): Whether to plot storage as percentages of capacity (optional).
        plot_drought_levels (bool): Whether to plot drought levels (optional).
        plot_releases (bool): Whether to plot releases (optional).

    Returns:
        None
    """



    ffmp_level_colors = ['blue', 'blue', 'blue', 'cornflowerblue', 'green', 'darkorange', 'maroon']
    drought_cmap = ListedColormap(ffmp_level_colors, N=7)
    norm = plt.Normalize(0, 6)
    
    ### get reservoir storage capacities
    istarf = pd.read_csv(f'{model_data_dir}drb_model_istarf_conus.csv')
    def get_reservoir_capacity(reservoir):
        return float(istarf['Adjusted_CAP_MG'].loc[istarf['reservoir'] == reservoir].iloc[0])
    capacities = {r: get_reservoir_capacity(r) for r in reservoir_list_nyc}
    capacities['combined'] = sum([capacities[r] for r in reservoir_list_nyc])

    historic_storage = pd.read_csv(f'{input_dir}/historic_NYC/NYC_storage_daily_2000-2021.csv', sep=',', index_col=0)
    historic_storage.index = pd.to_datetime(historic_storage.index)
    historic_storage = subset_timeseries(historic_storage, start_date, end_date)

    historic_release = pd.read_excel(f'{input_dir}/historic_NYC/Pep_Can_Nev_releases_daily_2000-2021.xlsx', index_col=0)
    historic_release.index = pd.to_datetime(historic_release.index)
    historic_release = historic_release.iloc[:,:3]
    historic_release = subset_timeseries(historic_release, start_date, end_date) * cfs_to_mgd
    historic_release.columns = ['pepacton','cannonsville','neversink']
    historic_release['Total'] = historic_release.sum(axis=1)

    ### add seasonal min FFMP releases (table 3 https://webapps.usgs.gov/odrm/documents/ffmp/Appendix_A_FFMP-20180716-Final.pdf)
    historic_release['FFMP_min_release'] = 95 * cfs_to_mgd
    historic_release['FFMP_min_release'].loc[[m in (6,7,8) for m in historic_release.index.month]] = 190 * cfs_to_mgd

    # model_names = [m[5:] for m in models]
    drought_levels = pd.DataFrame()
    for model in models:
        drought_levels[model] = subset_timeseries(all_drought_levels[model]['nyc'], start_date, end_date)
    
    # Create figure with m subplots
    n_subplots = 3 if plot_releases else 2
    
    fig = plt.figure(figsize=(8, 5), dpi=200)
    gs = gridspec.GridSpec(nrows=n_subplots, ncols=2, width_ratios=[15, 1], height_ratios=[1, 3, 2], wspace=0.05)
    
    ## Plot drought levels
    if plot_drought_levels:
        ax1 = fig.add_subplot(gs[0, 0])
        ax_cbar = fig.add_subplot(gs[0, 1])
        sns.heatmap(drought_levels.transpose(), cmap = drought_cmap,  
                    ax = ax1, norm=norm,
                    cbar_ax = ax_cbar, cbar_kws = dict(use_gridspec=False))
        ax1.set_xticklabels([])
        ax1.set_xticks([])
        ax1.set_yticklabels([])
        ax1.set_ylabel('FFMP\nLevel', fontsize=12)
            
    # Invert the colorbar
    if ax_cbar is not None:
        ax_cbar.invert_yaxis()  

    ## Plot combined storage
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.grid(True, which='major', axis='y')
    for m in models:
        if use_percent:
            sim_data = subset_timeseries(storages[m][reservoir_list_nyc].sum(axis=1), start_date, end_date)/capacities['combined']*100
            hist_data = subset_timeseries(historic_storage['Total'], start_date, end_date)/capacities['combined']*100
            ylab = f'Storage\n(% Useable)'
        else:
            sim_data = subset_timeseries(storages[m][reservoir_list_nyc].sum(axis=1), start_date, end_date)
            hist_data = subset_timeseries(historic_storage['Total'], start_date, end_date)
            ylab = f'Combined NYC Reservoir Storage (MG)'
        if plot_sim:
            ax2.plot(sim_data, color=colordict[m], label=f'{m}')
    if plot_observed:
        ax2.plot(hist_data, color=colordict['obs'], label=f'Observed')
    datetime = sim_data.index
    
    if add_ffmp_levels:
        # Load profiles
        level_profiles = pd.read_csv(f'{model_data_dir}drb_model_dailyProfiles.csv', sep=',')
        level_profiles = level_profiles.transpose()
        level_profiles.columns= level_profiles.iloc[0]
        level_profiles=level_profiles[1:]
        # Format to make datetime
        level_profiles.index=pd.to_datetime(level_profiles.index+f'-1944', 
                                            format='%d-%b-%Y')
        for l in ffmp_levels_to_plot:
            d_emergency=pd.DataFrame(data= level_profiles[f'level{l}']*100,
                                    index=pd.date_range('1944-01-01', end_date))
            first_year_data = d_emergency[d_emergency.index.year == 1944]
            day_of_year_to_value = {day.day_of_year: value for day, value in zip(first_year_data.index, first_year_data[f'level{l}'])}
            d_emergency.columns=[f'level{l}']
            
            d_emergency[f'level{l}'] = d_emergency.apply(lambda row: day_of_year_to_value[row.name.day_of_year] if np.isnan(row[f'level{l}']) else row[f'level{l}'], axis=1)
        
            # Plot
            ax2.plot(subset_timeseries(d_emergency, start_date, end_date),
                     color=drought_cmap(l),ls='dashed', zorder=1, alpha = 0.3,
                     label= f'FFMP L{l}')
    
    ax2.set_ylabel(ylab, fontsize = 12)
    ax2.yaxis.set_label_coords(-0.1, 0.5) # Set y-axis label position
    ax2.set_ylim([0, 110])
    ax2.set_xticklabels([])
    ax2.set_xlim([start_date, end_date])
    
    # Plot releases
    ax3 = fig.add_subplot(gs[2,0])
    if plot_sim:
        for m in models:
            sim_data = subset_timeseries(releases[m][reservoir_list_nyc].sum(axis=1), start_date, end_date)
            sim_data.index = datetime
            
            if smooth_releases:
                rd_rolling= sim_data.rolling(window=smooth_window).mean().values
                rd_rolling[0:smooth_window]= sim_data.values[0:smooth_window]
                rd_rolling[-smooth_window:]= sim_data.values[-smooth_window:]
                
                ax3.plot(sim_data.index, rd_rolling, color = colordict[m], label = m, lw = 1)
            else:
                ax3.plot(sim_data.index, sim_data, color = colordict[m], label = m, lw = 1)

    if plot_observed:
        ax3.plot(historic_release['Total'], color = colordict['obs'], label=f'Observed', 
                lw = 1, zorder=3)
    ax3.plot(historic_release['FFMP_min_release'], color ='black', ls =':', zorder=3,
            label = f'FFMP Min. Allowable Combined Release\nAt Drought Level 5')

    ax3.set_yscale('log')
    ax3.set_ylim([10, 10000])
    ax3.yaxis.set_label_coords(-0.1, 0.5)
    ax3.set_ylabel('Releases\n(MGD)', fontsize = 12)
    ax3.set_xlabel('Date', fontsize = 12)
    ax3.set_xlim([start_date, end_date])

    # Create legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles3, labels3 = ax3.get_legend_handles_labels()
    handles = handles1 + handles2 + handles3
    labels = labels1 + labels2 + labels3
    plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.5))
    
    plt.xlabel('Date')

    # plt.legend(loc = 'upper left', bbox_to_anchor=(0., -0.5), ncols=2)
    plt.tight_layout()
    fig.align_labels()
    plt.suptitle('Combined NYC Reservoir Operations\nSimulated & Observed')
    plt.savefig(f'{fig_dir}NYC_reservoir_ops_{start_date.strftime("%Y")}_{end_date.strftime("%Y")}{filename_addon}.png', dpi=250)
    # plt.show()
    return