"""
Contains all plotting functions used for Pywr-DRB model assessments, including:


"""
import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import gridspec
from matplotlib import cm
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize, LogNorm, ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy import stats
import sys


import hydroeval as he
import h5py
import datetime as dt

from pywrdrb.pywr_drb_node_data import upstream_nodes_dict, downstream_node_lags, immediate_downstream_nodes_dict

# Custom modules
from pywrdrb.utils.constants import cms_to_mgd, cm_to_mg, cfs_to_mgd
from pywrdrb.utils.lists import reservoir_list, reservoir_list_nyc, majorflow_list, reservoir_link_pairs, seasons_dict,\
    drbc_lower_basin_reservoirs

from pywrdrb.utils.directories import input_dir, fig_dir, output_dir, model_data_dir, spatial_data_dir
from pywrdrb.make_model import get_reservoir_max_release
from pywrdrb.plotting.styles import base_model_colors, model_hatch_styles, paired_model_colors, scatter_model_markers, \
    node_label_dict, node_label_full_dict, model_label_dict, month_dict, model_linestyle_dict


dpi=400


### function to return subset of dates for timeseries data
def subset_timeseries(timeseries, start_date, end_date, end_inclusive=True):
    data = timeseries.copy()
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    if not end_inclusive:
        end_date = end_date - dt.timedelta(days=1)

    if start_date is not None:
        data = data.loc[start_date:]
    if end_date is not None:
        data = data.loc[:end_date]
    return data



def plot_3part_flows_hier(reservoir_downstream_gages, major_flows, nodes, models, colordict = paired_model_colors,
                            markerdict = scatter_model_markers, start_date=None, end_date=None, end_inclusive=False,
                            uselog=False, save_fig=True, fig_dir = fig_dir):

    use2nd = True if len(models) > 1 else False
    alpha_lines = 1
    alpha_dots = 0.25
    fontsize = 8

    labels = [['a)','b)','c)'],
              ['d)', 'e)', 'f)'],
              ['g)', 'h)', 'i)']
              ]

    fig = plt.figure(figsize=(8, 5.2))
    gs = fig.add_gridspec(3,3, width_ratios=(2, 1, 1), wspace=0.35, hspace=0.25)

    ### function to get correct data for each row of figure
    def get_fig_data(model, node):
        if node in ['cannonsville','NYCAgg']:
            ### cannonsville only
            data = subset_timeseries(reservoir_downstream_gages[model][node], start_date, end_date,
                                     end_inclusive=end_inclusive)

        elif node in ['delMontague','delTrenton']:
            ### trenton
            data = subset_timeseries(major_flows[model][node], start_date, end_date,
                                     end_inclusive=end_inclusive)
        else:
            print(f'get_fig_data() not set for node {node}')
            data = ''
        return data


    for row, node in enumerate(nodes):

        ### get observed data for comparison
        obs = get_fig_data('obs', node)

        ### first fig: time series of observed & modeled flows
        ax = fig.add_subplot(gs[row, 0])
        for i, m in enumerate(models):
            if use2nd or i == 0:
                ### first plot time series of observed vs modeled
                modeled = get_fig_data(m, node)

                if i == 0:
                    ax.plot(obs, label=model_label_dict['obs'], color=colordict['obs'], zorder=2, alpha=alpha_lines)
                ax.plot(modeled, label=model_label_dict[m], color=colordict[m], zorder=1, alpha=alpha_lines)

                ax.set_ylabel('Daily Flow (MGD)', fontsize=fontsize)

                if uselog:
                    ax.semilogy()
                ax.set_xlim([start_date, end_date])
                ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', va='top', fontsize=fontsize)

        ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
        if row == 2:
            ax.legend(frameon=False, loc='upper center', bbox_to_anchor = (1.1, -0.35), ncols=3, fontsize=fontsize)
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)
        else:
            ax.set_xticklabels(['']*len(ax.get_xticklabels()), fontsize=fontsize)

        ax.annotate(labels[row][0], xy=(0.02, 0.95), xycoords='axes fraction', fontsize=fontsize, weight='bold',
                    va='top', ha='left')


        ### second fig: scatterplot of observed vs modeled flow
        ax = fig.add_subplot(gs[row, 1])
        for i, m in enumerate(models):
            ### now add scatter of observed vs modeled
            if use2nd or i == 0:
                ### first plot time series of observed vs modeled
                modeled = get_fig_data(m, node)
                ax.scatter(obs, modeled, alpha=alpha_dots, zorder=2, color=colordict[m], marker=markerdict[m])
                diagmax = min(ax.get_xlim()[1], ax.get_ylim()[1])
                ax.plot([0, diagmax], [0, diagmax], color=colordict['obs'], ls='--')

                ax.loglog()
                if row == 2:
                    ax.set_xlabel('Observed Flow (MGD)', fontsize=fontsize)
                ax.set_ylabel('Modeled Flow (MGD)', fontsize=fontsize)

        ax.annotate(labels[row][1], xy=(0.05, 0.95), xycoords='axes fraction', fontsize=fontsize, weight='bold',
                    va='top', ha='left')

        ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)

        ### third fig: flow duration curves
        ax = fig.add_subplot(gs[row, 2])
        def plot_exceedance(data, ax, color, label, **kwargs):
            df = data.sort_values()[::-1]
            exceedance = (np.arange(1., len(df) + 1.) / len(df) * 100)
            ax.plot(exceedance, df, color=color, label=label, **kwargs)

        for i, m in enumerate(models):
            if use2nd or i == 0:
                modeled = get_fig_data(m, node)
                if i == 0:
                    plot_exceedance(obs, ax, color = colordict['obs'], label=model_label_dict['obs'],
                                    alpha=alpha_lines, zorder=2)
                    ax.semilogy()
                    if row == 2:
                        ax.set_xlabel('Exceedence (%)', fontsize=fontsize)
                    ax.set_ylabel('Daily Flow (MGD)', fontsize=fontsize)

                plot_exceedance(modeled, ax, color = colordict[m], label=model_label_dict[m],
                                alpha = alpha_lines, zorder=1)

        ax.annotate(labels[row][2], xy=(0.07, 0.95), xycoords='axes fraction', fontsize=fontsize, weight='bold',
                    va='top', ha='left')

        ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)

    # plt.show()
    if save_fig:
        if use2nd:
            fig.savefig(f'{fig_dir}streamflow_3plots_{models[0]}_{models[1]}_hier_' + \
                f'{modeled.index.year[0]}_{modeled.index.year[-1]}.png', bbox_inches='tight', dpi = dpi)
        else:
            fig.savefig(f'{fig_dir}streamflow_3plots_{models[0]}_hier_' + \
                f'{modeled.index.year[0]}_{modeled.index.year[-1]}.png', bbox_inches='tight', dpi = dpi)
        plt.close()
    return








###
def get_error_metrics(reservoir_downstream_gages, major_flows, models, nodes, start_date=None, end_date=None,
                      end_inclusive=False):
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
        if node in reservoir_list + ['NYCAgg']:
            results = reservoir_downstream_gages
        else:
            results = major_flows

        for i, m in enumerate(models):
            resultsdict = {}
            for timescale in ['D','M']:
                obs = subset_timeseries(results['obs'][node], start_date, end_date, end_inclusive=end_inclusive)
                modeled = subset_timeseries(results[m][node], start_date, end_date, end_inclusive=end_inclusive)
                if timescale == 'M':
                    obs = obs.resample('M').mean()
                    modeled = modeled.resample('M').mean()

                ### get kge & nse
                kge, r, alpha, beta = he.evaluator(he.kge, modeled, obs)
                nse = he.evaluator(he.nse, modeled, obs)
                logkge, logr, logalpha, logbeta = he.evaluator(he.kge, modeled, obs, transform='log')
                lognse = he.evaluator(he.nse, modeled, obs, transform='log')

                ### autocorrelation, 1 & 7 day
                autocorr1_obs_log = np.corrcoef(np.log(obs.iloc[1:]), np.log(obs.iloc[:-1]))[0,1]
                autocorr1_mod_log = np.corrcoef(np.log(modeled.iloc[1:]), np.log(modeled.iloc[:-1]))[0,1]
                rel_autocorr1 = autocorr1_mod_log / autocorr1_obs_log
                autocorr7_obs_log = np.corrcoef(np.log(obs.iloc[7:]), np.log(obs.iloc[:-7]))[0,1]
                autocorr7_mod_log = np.corrcoef(np.log(modeled.iloc[7:]), np.log(modeled.iloc[:-7]))[0,1]
                rel_autocorr7 = autocorr7_mod_log / autocorr7_obs_log
                ### relative roughness, measured as std of diffed series. Do this in log space too.
                ### https://stats.stackexchange.com/questions/24607/how-to-measure-smoothness-of-a-time-series-in-r
                rel_roughness_log = np.std(np.log(modeled.iloc[1:].values) - np.log(modeled.iloc[:-1].values)) / \
                                    np.std(np.log(obs.iloc[1:].values) - np.log(obs.iloc[:-1].values))

                ### get horizontal FDC match metric (1 minus max horiz distance between FDCs, i.e., Kolmogorov-Smirnov Statistic - 1 in ideal case, 0 in worst case)
                kss, _ = stats.ks_2samp(modeled, obs)
                fdc_match_horiz = 1 - kss
                ### get vertical FDC match metric (1 minus max vertical distance between FDCs, normalized by range of FDCs, in log space. Also 1 in ideal case, 0 in worst case)
                obs_ordered = np.log(np.sort(obs))
                modeled_ordered = np.log(np.sort(modeled))
                fdc_range = max(obs_ordered.max(), modeled_ordered.max()) - min(obs_ordered.min(), modeled_ordered.min())
                fdc_match_vert = 1 - np.abs(obs_ordered - modeled_ordered).max() / fdc_range

                ### FDC slope relative bias. follow Yan et al JAMES SI, but switch to relative value so 1 is best, like bias/std.
                ### Do this in log space, even though Yan et al doesnt do that. seems like real space is just measuring size of higher quantile.
                sfdc_2575_obs_log = (np.log(np.quantile(obs, 0.75)) - np.log(np.quantile(obs, 0.25))) / 0.5
                sfdc_0595_obs_log = (np.log(np.quantile(obs, 0.95)) - np.log(np.quantile(obs, 0.05))) / 0.9
                sfdc_MinMax_obs_log = (np.log(obs.max()) - np.log(obs.min()))
                sfdc_2575_mod_log = (np.log(np.quantile(modeled, 0.75)) - np.log(np.quantile(modeled, 0.25))) / 0.5
                sfdc_0595_mod_log = (np.log(np.quantile(modeled, 0.95)) - np.log(np.quantile(modeled, 0.05))) / 0.9
                sfdc_MinMax_mod_log = (np.log(modeled.max()) - np.log(modeled.min()))
                sfdc_relBias_2575_log = sfdc_2575_mod_log / sfdc_2575_obs_log
                sfdc_relBias_0595_log = sfdc_0595_mod_log / sfdc_0595_obs_log
                sfdc_relBias_MinMax_log = sfdc_MinMax_mod_log / sfdc_MinMax_obs_log

                ### relative biases in different seasons
                relbiases = {'Annual': modeled.mean() / obs.mean()}
                season_ts = [seasons_dict[m] for m in obs.index.month]
                seasons = ['DJF','MAM','JJA','SON']
                for season in seasons:
                    bools = np.array([s == season for s in season_ts])
                    relbiases[season] = modeled.loc[bools].mean() / obs.loc[bools].mean()

                ### relative biases in different quantiles of observed series
                qgroups = [(0,5),(5,25),(25,75),(75,95),(95,100)]
                for qgroup in qgroups:
                    if qgroup[1] == 100:
                        bools = obs >= np.quantile(obs, qgroup[0]/100)
                    elif qgroup[0] == 0:
                        bools = obs < np.quantile(obs, qgroup[1]/100)
                    else:
                        bools = np.logical_and(obs >= np.quantile(obs, qgroup[0]/100),
                                               obs < np.quantile(obs, qgroup[1]/100))

                    relbiases[f'q{qgroup[0]}-{qgroup[1]}'] = modeled.loc[bools].mean() / obs.loc[bools].mean()
                ### rel bias in min/max flow
                relbiases['qMin'] = modeled.min() / obs.min()
                relbiases['qMax'] = modeled.max() / obs.max()

                resultsdict_inner = {'nse': nse[0], 'kge': kge[0], 'r': r[0], 'alpha': alpha[0], 'beta': beta[0],
                                   'lognse': lognse[0], 'logkge': logkge[0], 'logr': logr[0], 'logalpha': logalpha[0],
                                   'logbeta': logbeta[0],
                                     'rel_autocorr1': rel_autocorr1, 'rel_autocorr7': rel_autocorr7,
                                     'rel_roughness_log': rel_roughness_log,
                                     'fdc_match_horiz': fdc_match_horiz, 'fdc_match_vert': fdc_match_vert,
                                     'sfdc_relBias_2575_log': sfdc_relBias_2575_log,
                                     'sfdc_relBias_0595_log': sfdc_relBias_0595_log,
                                     'sfdc_relBias_MinMax_log': sfdc_relBias_MinMax_log,
                                     'bias_Annual': relbiases['Annual']}

                for key in seasons + ['qMin','qMax']:
                    resultsdict_inner[f'bias_{key}'] = relbiases[key]
                for qgroup in qgroups:
                    resultsdict_inner[f'bias_q{qgroup[0]}-{qgroup[1]}'] = relbiases[f'q{qgroup[0]}-{qgroup[1]}']

                for k,v in resultsdict_inner.items():
                    resultsdict[f'{timescale}_{k}'] = v

            resultsdict['node'] = node
            resultsdict['model'] = m
            if i == 0 and j == 0:
                results_metrics = pd.DataFrame(resultsdict, index=[0])
            else:
                results_metrics = pd.concat([results_metrics, pd.DataFrame(resultsdict, index=[0])])

    results_metrics.reset_index(inplace=True, drop=True)
    return results_metrics






### gridded version of multimetric/multisite/multimodel comparison of error metrics wrt observed record
def plot_gridded_error_metrics(results_metrics, models, nodes, start_date, end_date, figstage=0, fig_dir = fig_dir):
    """

    """
    ### figstage: 0 = Smaller subset of locations & models, with daily metrics
    ###           1 = Full set of locations & models, with daily metrics
    ###           2 = Full set of locations & models, with monthly metrics
    ### for figstage==0 (subset of nodes & models for smaller fig for main text figure), use larger font
    if figstage == 0:
        fontsize = 8
    else:
        fontsize = 7

    fig, ax = plt.subplots(1,1, figsize=(9,9))

    # metrics = ['nse', 'kge', 'r', 'alpha', 'beta', 'kss', 'lognse', 'logkge', 'sfdc_relBias_2575', 'sfdc_relBias_0595']
    # metrics = [f'{timescale}_{metric}' for metric in metrics for timescale in ('D','M')]
    if figstage < 2:
        metrics = ['D_nse', 'D_kge', 'D_r', 'D_fdc_match_horiz', 'D_fdc_match_vert',
                   'D_sfdc_relBias_2575_log', 'D_sfdc_relBias_0595_log', 'D_alpha',
                   'D_rel_autocorr1', 'D_rel_roughness_log',
                   'D_bias_Annual', 'D_bias_DJF', 'D_bias_MAM', 'D_bias_JJA', 'D_bias_SON',
                   'D_bias_qMin', 'D_bias_q0-5', 'D_bias_q5-25', 'D_bias_q25-75', 'D_bias_q75-95',
                   'D_bias_q95-100', 'D_bias_qMax']
    else:
        metrics = ['M_nse', 'M_kge', 'M_r', 'M_fdc_match_horiz', 'M_fdc_match_vert',
                   'M_sfdc_relBias_2575_log', 'M_sfdc_relBias_0595_log', 'M_alpha',
                   'M_rel_autocorr1', 'M_rel_roughness_log',
                   'M_bias_Annual', 'M_bias_DJF', 'M_bias_MAM', 'M_bias_JJA', 'M_bias_SON',
                   'M_bias_qMin', 'M_bias_q0-5', 'M_bias_q5-25', 'M_bias_q25-75', 'M_bias_q75-95',
                   'M_bias_q95-100', 'M_bias_qMax']

    # metric_label_dict = {'nse': 'Nash-Sutcliffe Efficiency', 'kge': 'Kling-Gupta Efficiency', 'r': 'Correlation', 'alpha': 'Relative STD', 'beta': 'Relative Bias',
    #                      'kss': 'Kolmogorov-Smirnov Metric', 'lognse': 'Log Nash-Sutcliffe Efficiency', 'logkge': 'Log Kling-Gupta Efficiency'}
    metric_label_dict = {'nse': 'NSE', 'kge': 'KGE', 'r': 'Corr.', 'alpha': 'Rel. STD', 'beta': 'Rel. Bias',
                         'fdc_match_horiz': 'FDC Horiz. Match', 'fdc_match_vert': 'FDC Vert. Match', 'lognse': 'Log NSE', 'logkge': 'Log KGE',
                         'sfdc_relBias_2575':'FDC Q25-75 Slope Rel. Bias',
                         'sfdc_relBias_0595':'FDC Q5-95 Slope Rel. Bias',
                         'sfdc_relBias_2575_log': 'Log FDC Q25-75 Slope Rel. Bias',
                         'sfdc_relBias_0595_log': 'Log FDC Q5-95 Slope Rel. Bias',
                         'bias_Annual':'Annual Rel. Bias', 'bias_DJF':'DJF Rel. Bias', 'bias_MAM':'MAM Rel. Bias',
                         'bias_JJA':'JJA Rel. Bias', 'bias_SON':'SON Rel. Bias',
                         'bias_qMin': 'QMin Rel. Bias', 'bias_q0-5': 'Q0-5 Rel. Bias', 'bias_q5-25': 'Q5-25 Rel. Bias',
                         'bias_q25-75': 'Q25-75 Rel. Bias', 'bias_q75-95': 'Q75-95 Rel. Bias',
                         'bias_q95-100': 'Q95-100 Rel. Bias', 'bias_qMax': 'QMax Rel. Bias'}

    timescale_label_dict = {'D': 'Daily', 'M':'Monthly'}
    metric_label_dict = {f'{timescale}_{k}': f'{timescale_label_dict[timescale]} {v}' \
                         for k,v in metric_label_dict.items() for timescale in ('D','M')}

    labels = ['a)','b)','c)','d)','e)','f)','g)','h)','i)']

    ### individual colormap for each metric
    sm_dict = {}
    vrange_dict = {}
    scale_dict = {}
    for metric in metrics:
        vmin = results_metrics[metric].min()
        vmax = results_metrics[metric].max()
        ### linear color palette for relative bias type metrics
        # if vmax > 1:
        #     ### set vmin & vmax to be centered at 1, with widest allowed limits of (0,2)
        #     vabs = max(abs(vmin-1), abs(vmax-1))
        #     vmin = - min(vabs, 1) + 1
        #     vmax = min(vabs, 1) + 1
        #     vrange_dict[metric] = (vmin, vmax)
        #     # ### now expand slightly for colormap so that colors aren't too dark to read text
        #     range = vmax - vmin
        #     vmax += range * 0.1
        #     vmin -= range * 0.1
        #     ### separate norm/cmap for above and below 1
        #     cmaps = [cm.get_cmap('Reds_r'), cm.get_cmap('Purples')]
        #     ## note: Purples palette starts off slower than Reds (more white), so add a bit of bias here to even out colors
        #     norms = [mpl.colors.Normalize(vmin=vmin, vmax=1), mpl.colors.Normalize(vmin=0.93, vmax=vmax)]

        ### log color palette for relative bias type metrics
        if vmax > 1:
            ### set vmin & vmax to be centered at 1, with widest allowed limits of (0,2)
            vabs_log2 = max(abs(np.log2(vmin)), abs(np.log2(vmax)))
            vmin_log2 = -vabs_log2
            vmax_log2 = vabs_log2
            vrange_dict[metric] = (vmin_log2, vmax_log2)
            # ### now expand slightly for colormap so that colors aren't too dark to read text
            range = vmax_log2 - vmin_log2
            vmax_log2 += range * 0.12
            vmin_log2 -= range * 0.12
            ### separate norm/cmap for above and below 1
            cmaps = [cm.get_cmap('Greens_r'), cm.get_cmap('Purples')]
            ## note: Purples palette starts off slower than Reds (more white), so add a bit of buffer here to even out colors
            norms = [mpl.colors.Normalize(vmin=vmin_log2, vmax=0), mpl.colors.Normalize(vmin=-0.05 * vmax_log2,
                                                                                        vmax=vmax_log2)]
            scale_dict[metric] = 'log2'

        else:
            ### for metrics with no values >1, set max at 0. widest allowable limits of (-1,1).
            vmax = 1
            # vmin = max(vmin, -1)
            vrange_dict[metric] = (vmin, vmax)
            # ### now expand slightly for colormap so that colors aren't too dark to read text
            vmin -= (vmax - vmin) * 0.15
            ### norm/cmap
            cmaps = [cm.get_cmap('Reds_r')]
            norms = [mpl.colors.Normalize(vmin=vmin, vmax=vmax)]
            scale_dict[metric] = 'linear_max1'

        sm_dict[metric] = [cm.ScalarMappable(cmap=cmap, norm=norm) for cmap, norm in zip(cmaps, norms)]

    ### Add metric-specific colorbars annotate metrics
    num_gradations = 200
    # has_upper_extension = False

    ax.annotate(labels[0], xy=(-1.3, 105.3), va='top', ha='left', weight='bold', fontsize=fontsize)
    for x, metric in enumerate(metrics):
        vs = np.arange(vrange_dict[metric][0], vrange_dict[metric][1]+0.001,
                           (vrange_dict[metric][1]-vrange_dict[metric][0]) / num_gradations)
        dy = 3 / num_gradations
        ys = np.arange(101.5, 104.5+0.001, dy)
        for v,y in zip(vs, ys):
            v_color = min(max(v, vrange_dict[metric][0]), vrange_dict[metric][1])
            if scale_dict[metric] == 'linear_max1':
                cmap_idx = 0 if v_color <= 1 else 1
            elif scale_dict[metric] == 'log2':
                cmap_idx = 0 if v_color <= 0 else 1

            c = sm_dict[metric][cmap_idx].to_rgba(v_color)
            box = [Rectangle((x, y), 1, dy)]
            pc = PatchCollection(box, facecolor=c, lw=0.0, zorder=1)
            ax.add_collection(pc)
            ### annotation in square based on metric range
            if scale_dict[metric] == 'linear_max1':
                dy_dict = {ys[0]: -0.5, ys[-1]: 0.4,
                           ys[np.argmin(np.abs(vs-1))]: 0.4 if ys[-1] == ys[np.argmin(np.abs(vs-1))] \
                               else -0.5 if ys[0] == ys[np.argmin(np.abs(vs-1))] else 0}
            elif scale_dict[metric] == 'log2':
                dy_dict = {ys[0]: -0.5, ys[-1]: 0.4,
                           ys[np.argmin(np.abs(vs))]: 0.4 if ys[-1] == ys[np.argmin(np.abs(vs))] \
                               else -0.5 if ys[0] == ys[np.argmin(np.abs(vs))] else 0}
            if y in dy_dict:
                v_annot = 2 ** v_color if scale_dict[metric] == 'log2' else v_color
                ax.annotate(text=f'{v_annot:.2f}', xy=(x + 0.5, y + dy_dict[y]), ha='center', va='center',
                            fontsize=fontsize, color='k', annotation_clip=False, zorder=3)

        ### add outer boundary
        for lines in [[(x,x), (101.5,104.5)],[(x+1,x+1), (101.5,104.5)]]:
            ax.plot(lines[0], lines[1], lw=0.5, color='0.8', zorder=2)

        #
        # ### now add triangular extension for metrics that have values beyond colorscale limits
        # if results_metrics[metric].min() < vrange_dict[metric][0]:
        #     v_color = vrange_dict[metric][0]
        #     cmap_idx = 0 if v_color <= 1 else 1
        #     ax.fill_between([x, x+0.5, x+1], [101+dy, 100, 101+dy], [101+dy, 101+dy, 101+dy],
        #                     lw=0, alpha=1, color=sm_dict[metric][cmap_idx].to_rgba(v_color))
        #     ax.annotate(text=f'{results_metrics[metric].min():.2f}', xy=(x + 0.5, 100), ha='center', va='top',
        #                 fontsize=fontsize, color='k', annotation_clip=False)
        #     ### add outer boundary
        #     for lines in ([(x, x+0.5), (101, 100)], [(x+0.5, x+1), (100, 101)]):
        #         ax.plot(lines[0], lines[1], lw=0.5, color='0.8', zorder=2)
        # else:
        ### add outer boundary
        for lines in [[(x, x+1), (101.5, 101.5)]]:
            ax.plot(lines[0], lines[1], lw=0.5, color='0.8', zorder=2)
        #
        # if results_metrics[metric].max() > vrange_dict[metric][1]:
        #     v_color = vrange_dict[metric][1]
        #     cmap_idx = 0 if v_color <= 1 else 1
        #     ax.fill_between([x, x+0.5, x+1], [104-dy, 105, 104-dy], [104-dy, 104-dy, 104-dy],
        #                     lw=0, alpha=1, color=sm_dict[metric][cmap_idx].to_rgba(v_color))
        #     ax.annotate(text=f'{results_metrics[metric].max():.2f}', xy=(x + 0.5, 105), ha='center', va='bottom',
        #                 fontsize=fontsize, color='k', annotation_clip=False)
        #     has_upper_extension = True
        #     ### add outer boundary
        #     for lines in ([(x, x+0.5), (104, 105)], [(x+0.5, x+1), (105, 104)]):
        #         ax.plot(lines[0], lines[1], lw=0.5, color='0.8', zorder=2)
        # else:
        ### add outer boundary
        for lines in [[(x, x+1), (104.5, 104.5)]]:
            ax.plot(lines[0], lines[1], lw=0.5, color='0.8', zorder=2)


    ### annotate color scale
    ax.annotate(text='Color\nScale', xy=(-0.2, 103), rotation=90, ha='right', va='center', ma='center',
                fontsize=fontsize+1, color='k', annotation_clip=False)
    # upper_text_y = 106 if has_upper_extension else 105
    # for x, metric in enumerate(metrics):
    #     ax.annotate(text=metric_label_dict[metric], xy=(x+0.3, upper_text_y), rotation=30, ha='left', va='bottom',
    #                 fontsize=fontsize, color='k', annotation_clip=False)


    ### loop over models, nodes, & metrics. Color squares based on metrics, plus annotations.
    y = 100
    for i,model in enumerate(models):
        y -= 1
        if i==4:
            y -= 1
        ax.annotate(labels[i+1], xy=(-1.3, y-0.2), va='top', ha='left', weight='bold', fontsize=fontsize)

        for j,node in enumerate(nodes):
            y -= 1
            for x,metric in enumerate(metrics):
                ### color square based on metric value
                v = results_metrics[metric].loc[np.logical_and(results_metrics['node']==node,
                                                               results_metrics['model']==model)].values[0]
                if scale_dict[metric] == 'log2':
                    v = np.log2(v)
                v_color = min(max(v, vrange_dict[metric][0]), vrange_dict[metric][1])

                if scale_dict[metric] == 'linear_max1':
                    cmap_idx = 0 if v_color <= 1 else 1
                elif scale_dict[metric] == 'log2':
                    cmap_idx = 0 if v_color <= 0 else 1

                c = sm_dict[metric][cmap_idx].to_rgba(v_color)
                box = [Rectangle((x,y), 1, 1)]
                pc = PatchCollection(box, facecolor=c, edgecolor='0.8', lw=0.5)
                ax.add_collection(pc)
                ### annotation in square based on metric value
                v_annot = 2 ** v if scale_dict[metric] == 'log2' else v
                ax.annotate(text=f'{v_annot:.2f}', xy=(x+0.5, y+0.45), ha='center', va='center',
                            fontsize=fontsize, color='k', annotation_clip=False)
            ### annotate location
            ax.annotate(text=node_label_full_dict[node], xy=(x+1.1, y+0.5), ha='left', va='center',
                        fontsize=fontsize, color='k', annotation_clip=False)
        ### annotate model
        ax.annotate(text=model_label_dict[model].replace('-','-\n'),
                    xy=(-0.2, y+len(nodes)/2), rotation=90, ha='right', va='center', ma='center',
                    fontsize=fontsize+1, color='k', annotation_clip=False)

    ### annotate metric numbers
    for x, metric in enumerate(metrics):
        ax.annotate(text=str(x+1), xy=(x+0.5, 106), ha='center', va='bottom',
                    fontsize=fontsize+1, color='k', annotation_clip=False)
    ax.annotate(text='Error Metric', xy=(10, 107.25), ha='center', va='bottom',
                fontsize=fontsize+1, color='k', annotation_clip=False)

    ### clean up
    ax.set_xlim([-1.7,x+4])
    ax.set_ylim([y-0.7,107.75])
    ax.spines[['right', 'left', 'bottom', 'top']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ### add boundaries around sections
    box = [Rectangle((-1.5, 99.5+0.5), 26.25, 6), Rectangle((-1.5, 99.5-28.25), 26.25, 28.25),
           Rectangle((-1.5, 99.5-28.5*2), 26.25, 28.25)]
    pc = PatchCollection(box, facecolor='none', edgecolor='k', lw=1)
    ax.add_collection(pc)

    fig.savefig(f'{fig_dir}/griddedErrorMetrics_{start_date.year}_{end_date.year}_{figstage}.png',
                bbox_inches='tight', dpi=dpi)
    plt.close()
    return









def plot_combined_nyc_storage(storages, ffmp_level_boundaries, models, colordict = paired_model_colors,
                      start_date = '1999-10-01', end_date = '2010-05-31', fig_dir=fig_dir):
    """

    """

    fig, ax = plt.subplots(1,1,figsize=(7,3))
    fontsize = 8

    ### get reservoir storage capacities
    istarf = pd.read_csv(f'{model_data_dir}drb_model_istarf_conus.csv')
    def get_reservoir_capacity(reservoir):
        return float(istarf['Adjusted_CAP_MG'].loc[istarf['reservoir'] == reservoir].iloc[0])
    capacities = {r: get_reservoir_capacity(r) for r in reservoir_list_nyc}
    capacities['combined'] = sum([capacities[r] for r in reservoir_list_nyc])

    historic_storage = pd.read_csv(f'{input_dir}/historic_NYC/NYC_storage_daily_2000-2021.csv', sep=',', index_col=0)
    historic_storage.index = pd.to_datetime(historic_storage.index)
    historic_storage = subset_timeseries(historic_storage['Total'], start_date, end_date)
    historic_storage *= 100/capacities['combined']

    ffmp_level_boundaries = subset_timeseries(ffmp_level_boundaries, start_date, end_date) * 100
    ffmp_level_boundaries['level1a'] = 100.


    ### First plot FFMP levels as background color
    levels = [f'level{l}' for l in ['1a','1b','1c','2','3','4','5']]
    level_labels = ['Flood A', 'Flood B', 'Flood C', 'Normal', 'Drought Watch', 'Drought Warning', 'Drought Emergency']

    level_colors = [cm.get_cmap('Blues')(v) for v in [0.3, 0.2, 0.1]] +\
                    ['papayawhip'] +\
                    [cm.get_cmap('Reds')(v) for v in [0.1, 0.2, 0.3]]
    level_alpha = [0.7]*3 + [0.7] + [0.7]*3
    x = ffmp_level_boundaries.index
    for i in range(len(levels)):
        y0 = ffmp_level_boundaries[levels[i]]
        if i == len(levels)-1:
            y1 = 0.
        else:
            y1 = ffmp_level_boundaries[levels[i+1]]
        ax.fill_between(x, y0, y1, color=level_colors[i], lw=0.2, edgecolor='k',
                        alpha=level_alpha[i], zorder=1, label=level_labels[i])


    ax.plot(historic_storage, color='k', ls = ':', label=model_label_dict['obs'], zorder=3)
    line_colors = [colordict[m] for m in models]
    for m,c in zip(models,line_colors):
        modeled_storage = subset_timeseries(storages[m][reservoir_list_nyc], start_date, end_date).sum(axis=1)
        modeled_storage *= 100/capacities['combined']
        ax.plot(modeled_storage, color='k', ls='-', zorder=2, lw=1.5)
        ax.plot(modeled_storage, color=c, ls='-', label=model_label_dict[m], zorder=2, lw=1.2)


    ### clean up figure
    ax.set_xlim([start_date, end_date])
    ax.set_ylabel('Combined NYC Storage (%)', fontsize=fontsize)
    ax.set_ylim([0,110])
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncols=4, fontsize=fontsize)

    ### save fig
    plt.savefig(f'{fig_dir}NYC_storages_' + \
                f'{ffmp_level_boundaries.index.year[0]}_{ffmp_level_boundaries.index.year[-1]}.png',
                bbox_inches='tight', dpi=dpi)

    return






def plot_combined_nyc_storage_vs_diversion(storages, ffmp_level_boundaries, ibt_demands, ibt_diversions, models,
                                           customer, shortfall_metrics, colordict = paired_model_colors,
                                           start_date = '1999-10-01', end_date = '2010-05-31', fig_dir=fig_dir):
    """

    """

    fig, axs= plt.subplots(6,1,figsize=(6,7), gridspec_kw={'height_ratios':[2,1,1,1,1,1], 'hspace':0.1})

    fontsize = 8
    labels = ['a)','b)','c)','d)','e)','f)']

    ### subplot a: Reservoir modeled storages, no observed
    ax = axs[0]
    ### get reservoir storage capacities
    istarf = pd.read_csv(f'{model_data_dir}drb_model_istarf_conus.csv')
    def get_reservoir_capacity(reservoir):
        return float(istarf['Adjusted_CAP_MG'].loc[istarf['reservoir'] == reservoir].iloc[0])
    capacities = {r: get_reservoir_capacity(r) for r in reservoir_list_nyc}
    capacities['combined'] = sum([capacities[r] for r in reservoir_list_nyc])

    ffmp_level_boundaries = subset_timeseries(ffmp_level_boundaries, start_date, end_date) * 100
    ffmp_level_boundaries['level1a'] = 100.

    ### First plot FFMP levels as background color
    levels = [f'level{l}' for l in ['1a','1b','1c','2','3','4','5']]
    level_labels = ['Flood A', 'Flood B', 'Flood C', 'Normal', 'Drought Watch', 'Drought Warning', 'Drought Emergency']

    level_colors = [cm.get_cmap('Blues')(v) for v in [0.3, 0.2, 0.1]] +\
                    ['papayawhip'] +\
                    [cm.get_cmap('Reds')(v) for v in [0.1, 0.2, 0.3]]
    level_alpha = [0.7]*3 + [0.7] + [0.7]*3
    x = ffmp_level_boundaries.index
    for i in range(len(levels)):
        y0 = ffmp_level_boundaries[levels[i]]
        if i == len(levels)-1:
            y1 = 0.
        else:
            y1 = ffmp_level_boundaries[levels[i+1]]
        ax.fill_between(x, y0, y1, color=level_colors[i], lw=0.2, edgecolor='k',
                        alpha=level_alpha[i], zorder=1, label=level_labels[i])

    line_colors = [colordict[m] for m in models]
    for m,c in zip(models,line_colors):
        modeled_storage = subset_timeseries(storages[m][reservoir_list_nyc], start_date, end_date).sum(axis=1)
        modeled_storage *= 100/capacities['combined']
        ax.plot(modeled_storage, color='k', ls='-', zorder=2, lw=2)
        ax.plot(modeled_storage, color=c, ls='-', label=model_label_dict[m], zorder=2, lw=1.6)

    ### clean up figure
    ax.set_xlim([start_date, end_date + datetime.timedelta(days=-1)])
    ax.set_xticks(ax.get_xticks(), ['']*len(ax.get_xticks()))
    ax.set_yticklabels(ax.get_yticklabels(), fontsize = fontsize)
    ax.set_ylabel('Combined NYC\nStorage (%)', fontsize=fontsize)
    ax.set_ylim([0,100])
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.44, 1.04), fontsize=fontsize, ncols=4)
    # ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02,0.4), fontsize=fontsize)
    ax.annotate(labels[0], xy=(0.005, 0.025), xycoords='axes fraction', ha='left', va='bottom', weight='bold',
                fontsize=fontsize)


    ### subfigure b: IBT demands
    ax = axs[1]
    customer_key = 'demand_nyc' if customer == 'nyc' else 'demand_nj' if customer == 'nj' else 'error'
    dems = subset_timeseries(ibt_demands[m][customer_key], start_date, end_date) + 0.01  ### add a small amt to avoid dividing by zero
    ax.plot(dems, color='k')

    ### also plot monthly rolling average
    dems_rolling = dems.rolling(30, center=True).mean()
    ax.plot(dems_rolling, color='0.6')
    ax.set_ylabel('Demand\n(MGD)', fontsize=fontsize)
    ax.set_xlim(axs[0].get_xlim())
    ax.set_xticks(ax.get_xticks(), ['']*len(ax.get_xticks()))
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)

    ax.annotate(labels[1], xy=(0.005, 0.95), xycoords='axes fraction', ha='left', va='top', weight='bold',
                fontsize=fontsize)

    ### subfigure c-f IBT demand satisfaction for each of 4 pywr models
    ymin = 100
    for i in range(len(models)):
        ax = axs[2+i]

        ### add shortfall events
        m = models[i-2]
        event_starts = shortfall_metrics[customer][m]['event_starts']
        event_ends = shortfall_metrics[customer][m]['event_ends']
        event_labels = ['Shortfall Event' if i==0 else '' for i in range(len(event_starts))]
        for event_start, event_end, label in zip(event_starts, event_ends, event_labels):
            ax.fill_between([event_start, event_end], [0,0], [110,110], color='lightcoral', alpha=1, label=label)

        customer_key = 'delivery_nyc' if customer == 'nyc' else 'delivery_nj' if customer == 'nj' else 'error'
        divs = subset_timeseries(ibt_diversions[m][customer_key], start_date, end_date) + 0.01  ### add a small amt to match dems
        divs = divs.divide(dems) * 100
        ymin = min(ymin, divs.min())
        ax.plot(divs, color='k', label='Daily Satisfied')

        ### monthly rolling average
        divs_rolling = divs.rolling(30, center=True).mean()
        ax.plot(divs_rolling, color='0.6', label='30-Day Avg. Satisfied')

        ax.set_ylabel('Demand\nSatisfied (%)', fontsize=fontsize)
        if i == len(models)-1:
            ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.3), fontsize=fontsize, ncols=3)

    ### clean up, standardize y ranges
    yrange = 100 - ymin
    ymin = max(ymin - 0.15 * yrange, 0)
    ymax = 100 + 0.15 * yrange
    for i in range(2, 6):
        ax = axs[i]
        ax.set_ylim([ymin, ymax])
        ax.set_xlim(axs[0].get_xlim())
        if i < 5:
            ax.set_xticks(ax.get_xticks(), [''] * len(ax.get_xticks()))
        else:
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)

        ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
        ax.annotate(labels[i], xy=(0.005, 0.05), xycoords='axes fraction', ha='left', va='bottom', weight='bold',
                    fontsize=fontsize)



    ### save fig
    plt.savefig(f'{fig_dir}storages_diversions_{customer}_' + \
                f'{ffmp_level_boundaries.index.year[0]}_{ffmp_level_boundaries.index.year[-1]}.png',
                bbox_inches='tight', dpi=dpi)

    return













def plot_combined_nyc_storage_vs_minflows(storages, ffmp_level_boundaries, major_flows, lower_basin_mrf_contributions,
                                          mrf_targets, reservoir_releases, downstream_release_targets,
                                          base_model, shortfall_metrics, colordict = paired_model_colors,
                                           start_date = '1999-10-01', end_date = '2010-05-31', fig_dir=fig_dir):
    """

    """

    fig, axs= plt.subplots(5,1,figsize=(6, 7), gridspec_kw={'height_ratios':[2,1,1,1,1], 'hspace':0.1})

    fontsize = 8
    labels = ['a)','b)','c)','d)','e)']
    pywr_model = 'pywr_'+base_model

    ### subplot a: Reservoir modeled storages, no observed
    ax = axs[0]
    ### get reservoir storage capacities
    istarf = pd.read_csv(f'{model_data_dir}drb_model_istarf_conus.csv')
    def get_reservoir_capacity(reservoir):
        return float(istarf['Adjusted_CAP_MG'].loc[istarf['reservoir'] == reservoir].iloc[0])
    capacities = {r: get_reservoir_capacity(r) for r in reservoir_list_nyc}
    capacities['combined'] = sum([capacities[r] for r in reservoir_list_nyc])

    ffmp_level_boundaries = subset_timeseries(ffmp_level_boundaries, start_date, end_date) * 100
    ffmp_level_boundaries['level1a'] = 100.

    ### First plot FFMP levels as background color
    levels = [f'level{l}' for l in ['1a','1b','1c','2','3','4','5']]
    level_labels = ['Flood A', 'Flood B', 'Flood C', 'Normal', 'Drought Watch', 'Drought Warning', 'Drought Emergency']

    level_colors = [cm.get_cmap('Blues')(v) for v in [0.3, 0.2, 0.1]] +\
                    ['papayawhip'] +\
                    [cm.get_cmap('Reds')(v) for v in [0.1, 0.2, 0.3]]
    level_alpha = [0.7]*3 + [0.7] + [0.7]*3
    x = ffmp_level_boundaries.index
    for i in range(len(levels)):
        y0 = ffmp_level_boundaries[levels[i]]
        if i == len(levels)-1:
            y1 = 0.
        else:
            y1 = ffmp_level_boundaries[levels[i+1]]
        ax.fill_between(x, y0, y1, color=level_colors[i], lw=0.2, edgecolor='k',
                        alpha=level_alpha[i], zorder=1, label=level_labels[i])

    color_pywr_model = 'pywr_nhmv10_withObsScaled'
    line_color = colordict[color_pywr_model]
    modeled_storage = subset_timeseries(storages[pywr_model][reservoir_list_nyc], start_date, end_date).sum(axis=1)
    modeled_storage *= 100/capacities['combined']
    ax.plot(modeled_storage, color='k', ls='-', zorder=2, lw=2)
    ax.plot(modeled_storage, color=line_color, ls='-', label=f'{model_label_dict[pywr_model]} Storage', zorder=2, lw=1.6)

    ### clean up figure
    ax.set_xlim([start_date, end_date + datetime.timedelta(days=-1)])
    ax.set_xticks(ax.get_xticks(), ['']*len(ax.get_xticks()), fontsize=fontsize)
    ax.set_yticks(ax.get_yticks(), ax.get_yticklabels(), fontsize=fontsize)
    ax.set_ylabel('Combined NYC Storage (%)', fontsize=fontsize)
    ax.set_ylim([0,100])
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.04), ncols=4, fontsize=fontsize)
    # ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02,0.5), fontsize=fontsize)
    ax.annotate(labels[0], xy=(0.005, 0.025), xycoords='axes fraction', ha='left', va='bottom', weight='bold',
                fontsize=fontsize)


    ### now show base & pywr shortfall events for Montague then Trenton
    for i, mrf in enumerate(['delMontague', 'delTrenton']):
        ### First do shortfall events with base model
        ax = axs[1 + i*2]

        ### first plot fraction of static/max min flow satisfied on right y axis
        target = subset_timeseries(mrf_targets[pywr_model][f'mrf_target_{mrf}'], start_date, end_date)
        target_max = np.ones(len(target)) * target.max()
        flow = subset_timeseries(major_flows[base_model][mrf], start_date, end_date)

        satisfaction = np.minimum(flow.divide(target_max) * 100, np.ones(len(flow)) * 100)
        leftlinecolor = 'k'
        ax.plot(satisfaction, color=leftlinecolor, label='Daily Value')

        ### monthly rolling average
        satisfaction_rolling = satisfaction.rolling(30, center=True).mean()
        ax.plot(satisfaction_rolling, color='0.6', label='30-Day Avg.')

        ylims = [0, 110]# if mrf == 'delMontague' else [40, 105]
        yticks = [0, 50, 100]
        ax.set_ylim(ylims)
        ax.set_yticks(yticks)
        ax.set_ylabel('Normal\nMin. Flow\nSatisfied (%)', fontsize=fontsize, color=leftlinecolor)
        ax.set_yticks(ax.get_yticks(), ax.get_yticklabels(), fontsize=fontsize, color=leftlinecolor)
        ax.tick_params(axis='y', colors=leftlinecolor)
        ax.set_ylim(ylims)

        ### add shortfall events
        event_starts = shortfall_metrics[mrf][base_model]['event_starts']
        event_ends = shortfall_metrics[mrf][base_model]['event_ends']
        event_labels = ['Shortfall Event' if i==0 else '' for i in range(len(event_ends))]
        for event_start, event_end, event_label in zip(event_starts, event_ends, event_labels):
            ax.fill_between([event_start, event_end], [ylims[0]] * 2, [ylims[1]] * 2, color='lightcoral', alpha=1,
                            label=event_label)

        ### clean up
        ax.annotate(labels[1 + i*2], xy=(0.005, 0.05), xycoords='axes fraction', ha='left', va='bottom', weight='bold',
                     fontsize=fontsize, color='k')
        ax.set_xlim(axs[0].get_xlim())
        ax.set_xticks(ax.get_xticks(), [''] * len(ax.get_xticks()), fontsize=fontsize)




        ### now repeat with pywr model, and also add time varying min flow constraint
        ax = axs[2 + i*2]

        yticks = [0, 50, 100]
        ax.set_ylim(ylims)
        ax.set_yticks(yticks, yticks, fontsize=fontsize, color=leftlinecolor)
        ax.set_ylabel('Normal\nMin. Flow\nSatisfied (%)', fontsize=fontsize, color=leftlinecolor)
        ax.tick_params(axis='y', colors=leftlinecolor)
        ax.set_ylim(ylims)

        ### add shortfall events
        event_starts = shortfall_metrics[mrf][pywr_model]['event_starts']
        event_ends = shortfall_metrics[mrf][pywr_model]['event_ends']

        for event_start, event_end in zip(event_starts, event_ends):
            ax.fill_between([event_start, event_end], [ylims[0]] * 2, [ylims[1]] * 2, color='lightcoral', alpha=1,
                            label='Shortfall Event' if event_start == event_starts[0] else '')

        ###  plot fraction of static/max min flow satisfied on right y axis
        flow = subset_timeseries(major_flows[pywr_model][mrf], start_date, end_date)

        ### for Trenton, include BLue Marsh FFMP releases in Trenton Equiv flow
        if mrf == 'delTrenton':
            lower_basin_mrf = subset_timeseries(lower_basin_mrf_contributions[pywr_model], start_date, end_date)
            lower_basin_mrf.columns = [c.split('_')[-1] for c in lower_basin_mrf.columns]
            # acct for lag at blue marsh so it can be added to trenton equiv flow.
            for c in ['blueMarsh']:
                lag = downstream_node_lags[c]
                downstream_node = immediate_downstream_nodes_dict[c]
                while downstream_node != 'output_del':
                    lag += downstream_node_lags[downstream_node]
                    downstream_node = immediate_downstream_nodes_dict[downstream_node]
                if lag > 0:
                    lower_basin_mrf[c].iloc[lag:] = lower_basin_mrf[c].iloc[:-lag]
            flow += lower_basin_mrf['blueMarsh']

        ### calculate minimum flow threshold satisfaction
        satisfaction = np.minimum(flow.divide(target_max) * 100, np.ones(len(flow)) * 100)

        releases = subset_timeseries(reservoir_releases[pywr_model], start_date, end_date)
        dstargets = subset_timeseries(downstream_release_targets[pywr_model], start_date, end_date)

        leftlinecolor = 'k'
        ax.plot(satisfaction, color=leftlinecolor, label='Daily Satisfied')

        ### monthly rolling average
        satisfaction_rolling = satisfaction.rolling(30, center=True).mean()
        ax.plot(satisfaction_rolling, color='0.6', label='30-Day Avg. Satisfied')


        ### now plot dynamic min flow target on right y axis
        ax2 = ax.twinx()
        twincolor = 'cornflowerblue'
        ax2.plot(target, color=twincolor, ls='-', label='Dynamic Target')

        ax2.set_ylabel('Dynamic\nMin. Flow\nTarget (MGD)', fontsize=fontsize, rotation=270,
                       labelpad=30, color=twincolor)
        ax2.annotate(labels[2 + i*2], xy=(0.005, 0.05), xycoords='axes fraction', ha='left', va='bottom', weight='bold',
                     fontsize=fontsize, color='k')
        yticks = [0, 500, 1000] if mrf == 'delMontague' else [0, 1000, 2000]
        ax2.set_yticks(yticks)
        ax2.tick_params(axis='y', colors=twincolor)

        ax.set_xlim(axs[0].get_xlim())
        if mrf == 'delMontague':
            ax.set_xticks(ax.get_xticks(), [''] * len(ax.get_xticks()), fontsize=fontsize)
        else:
            ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), fontsize=fontsize)

        ### set right axis to correspond with left
        ax2.set_ylim([target_max[0] * ylims[0]/100, target_max[0] * ylims[1]/100])

        ### bottom legend
        if i == 1:
            ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.36, -0.3), fontsize=fontsize, ncols=3)
            ax2.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.99, -0.3), fontsize=fontsize, ncols=3)

    ### save fig
    plt.savefig(f'{fig_dir}storages_minflows_{base_model}_{pywr_model}_' + \
                f'{ffmp_level_boundaries.index.year[0]}_{ffmp_level_boundaries.index.year[-1]}.png',
                bbox_inches='tight', dpi=dpi)

    return





def plot_lowflow_exceedances(reservoir_downstream_gages, major_flows, lower_basin_mrf_contributions, models, nodes,
                             start_date=None, end_date=None, colordict = paired_model_colors):
    fontsize = 8
    windows = [1, 7, 30, 90, 365]#, 365*5]

    ### 6x3 grid of rolling avg windows (rows) & nodes (cols)
    fig, axs = plt.subplots(len(windows), len(nodes), figsize=(8,7), gridspec_kw={'wspace':0.2, 'hspace':0.2})

    ### set y ticks manually, default is terrible for some reason
    yticks = [[(0, 0.1, 0.2), (0, 0.25, 0.5), (0, 2, 4)],
              [(0, 0.1, 0.2), (0, 0.25, 0.5), (0, 2, 4)],
              [(0, 0.1, 0.2, 0.3), (0, 0.25, 0.5, 0.75), (0, 2, 4)],
              [(0, 0.2, 0.4), (0, 0.5, 1), (0, 4, 8)],
              [(0, 0.4, 0.8), (0, 0.5, 1, 1.5), (0, 4, 8)]
              ]
    labels = [['a)','f)','k)'],
              ['b)', 'g)', 'l)'],
              ['c)', 'h)', 'm)'],
              ['d)', 'i)', 'n)'],
              ['e)', 'j)', 'o)'],
              ]
    ### compile low flow metrics across models/nodes/metrics
    for col, node in enumerate(nodes):

        if node in reservoir_list + ['NYCAgg']:
            results = reservoir_downstream_gages
        else:
            results = major_flows

        for j, window in enumerate(windows):
            row = j
            ax = axs[row, col]
            for i, m in enumerate(models):
                modeled = subset_timeseries(results[m][node], start_date, end_date) / 1000 # convert to BGD for smaller labels

                ### for Trenton & Pywr models, include BLue Marsh FFMP releases in Trenton Equiv flow
                if 'pywr' in m and node == 'delTrenton':
                    lower_basin_mrf_contributions = subset_timeseries(lower_basin_mrf_contributions[m],
                                                                      start_date, end_date) / 1000
                    lower_basin_mrf_contributions.columns = [c.split('_')[-1] for c in
                                                             lower_basin_mrf_contributions.columns]
                    # acct for lag at blue marsh so it can be added to trenton equiv flow.
                    for c in ['blueMarsh']:
                        lag = downstream_node_lags[c]
                        downstream_node = immediate_downstream_nodes_dict[c]
                        while downstream_node != 'output_del':
                            lag += downstream_node_lags[downstream_node]
                            downstream_node = immediate_downstream_nodes_dict[downstream_node]
                        if lag > 0:
                            lower_basin_mrf_contributions[c].iloc[lag:] = lower_basin_mrf_contributions[c].iloc[:-lag]

                    modeled += lower_basin_mrf_contributions['blueMarsh']

                ### now get rolling average & plot xQn statistics
                modeled_rollingavg = modeled.rolling(window).mean()
                modeled_rollingavg_annmin = modeled_rollingavg.resample('A').min().values
                modeled_rollingavg_annmin_ordered = np.sort(modeled_rollingavg_annmin)[::-1]
                modeled_rollingavg_annmin_ordered = modeled_rollingavg_annmin_ordered[np.logical_not(np.isnan(modeled_rollingavg_annmin_ordered))]
                x = np.arange(len(modeled_rollingavg_annmin_ordered))
                return_periods = ((len(modeled_rollingavg_annmin_ordered) + 1) / (x + 1))[::-1]
                exceedances = np.arange(1, len(modeled_rollingavg_annmin_ordered) + 1) / len(modeled_rollingavg_annmin_ordered) * 100
                ax.plot(x, modeled_rollingavg_annmin_ordered, color = 'k', lw=2)
                ax.plot(x, modeled_rollingavg_annmin_ordered, color = colordict[m], lw=1.6, label=model_label_dict[m])
                ### print 7Q10
                if window == 1:
                    print(f'{node}, {m}, 1Q1= {np.interp(1, return_periods, modeled_rollingavg_annmin_ordered)}')
                    print(f'{node}, {m}, 1Q30= {np.interp(30, return_periods, modeled_rollingavg_annmin_ordered)}')
                if window == 7:
                    print(f'{node}, {m}, 7Q10= {np.interp(10, return_periods, modeled_rollingavg_annmin_ordered)}')

            ### bottom x ticks based on exceedance
            ticks_T_bottom = [0, 25, 50, 75, 100]
            ticks_X_bottom = [np.interp(T, exceedances, x) for T in ticks_T_bottom]

            if j == len(windows)-1:
                ax.set_xlabel('Exceedance (%)', fontsize=fontsize)
                ax.set_xticks(ticks_X_bottom, ticks_T_bottom, fontsize=fontsize-1)
                if col == 1:
                    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.4), ncols=4, fontsize=fontsize)
            else:
                ax.set_xticks(ticks_X_bottom, [''] * len(ticks_X_bottom), fontsize=fontsize)

            ###set top x tick locations based on return intervals. & labels etc.
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ticks_T_top = [1, 2, 5, 10, 30]
            ticks_X_top = [np.interp(T, return_periods, x) for T in ticks_T_top]

            if j == 0:
                # ax.set_title(node_label_full_dict[node], fontsize=fontsize)
                ax2.set_xlabel('Return Period (years)', fontsize=fontsize)
                ax2.set_xticks(ticks_X_top, ticks_T_top, fontsize=fontsize-1)

            else:
                ax2.set_xticks(ticks_X_top, ['']*len(ticks_X_top), fontsize=fontsize)


            if col == 0:
                ax.set_ylabel(f'Annual Min\n{window}-Day-Avg\nDaily Flow\n(BGD)', fontsize=fontsize)

            ax.set_ylim([0, ax.get_ylim()[1]])
            ax.set_yticks(yticks[row][col], yticks[row][col], fontsize=fontsize)
            ax.annotate(labels[row][col], xy=(0.98,0.95), xycoords='axes fraction', fontsize=fontsize, weight='bold',
                        va='top', ha='right')

    fig.savefig(f'{fig_dir}/lowFlowReturnPeriods_{start_date.year}_{end_date.year}.png',
                bbox_inches='tight', dpi=dpi)
    plt.close()
    return







def plot_NYC_release_components_combined(storages, ffmp_level_boundaries, nyc_release_components,
                                         lower_basin_mrf_contributions, reservoir_releases, reservoir_downstream_gages,
                                         major_flows, inflows, diversions, consumptions, base_model, node,
                                         colordict = base_model_colors, start_date = None, end_date = None,
                                         use_log=False, use_observed=False, fig_dir=fig_dir):

    fig, axs = plt.subplots(3,1,figsize=(7,7), 
                            gridspec_kw={'hspace':0.1},
                            sharex=True)
    fontsize = 8
    labels = ['a)','b)','c)']

    pywr_model = 'pywr_' + base_model
    pywr_model_color = 'pywr_nhmv10_withObsScaled'
    base_model_color = 'nhmv10_withObsScaled'

    ########################################################
    ### subplot a: Reservoir modeled storages
    ########################################################

    ax = axs[0]

    ### get reservoir storage capacities
    istarf = pd.read_csv(f'{model_data_dir}drb_model_istarf_conus.csv')
    def get_reservoir_capacity(reservoir):
        return float(istarf['Adjusted_CAP_MG'].loc[istarf['reservoir'] == reservoir].iloc[0])
    capacities = {r: get_reservoir_capacity(r) for r in reservoir_list_nyc}
    capacities['combined'] = sum([capacities[r] for r in reservoir_list_nyc])

    ffmp_level_boundaries = subset_timeseries(ffmp_level_boundaries, start_date, end_date) * 100
    ffmp_level_boundaries['level1a'] = 100.

    ### First plot FFMP levels as background color
    levels = [f'level{l}' for l in ['1a','1b','1c','2','3','4','5']]
    level_labels = ['Flood A', 'Flood B', 'Flood C', 'Normal', 'Drought Watch', 'Drought Warning', 'Drought Emergency']
    level_colors = [cm.get_cmap('Blues')(v) for v in [0.3, 0.2, 0.1]] +\
                    ['papayawhip'] +\
                    [cm.get_cmap('Reds')(v) for v in [0.1, 0.2, 0.3]]
    level_alpha = [0.7]*3 + [0.7] + [0.7]*3
    x = ffmp_level_boundaries.index
    for i in range(len(levels)):
        y0 = ffmp_level_boundaries[levels[i]]
        if i == len(levels)-1:
            y1 = 0.
        else:
            y1 = ffmp_level_boundaries[levels[i+1]]
        ax.fill_between(x, y0, y1, color=level_colors[i], lw=0.2, edgecolor='k',
                        alpha=level_alpha[i], zorder=1, label=level_labels[i])

    if use_observed:
        historic_storage = pd.read_csv(f'{input_dir}/historic_NYC/NYC_storage_daily_2000-2021.csv', sep=',', index_col=0)
        historic_storage.index = pd.to_datetime(historic_storage.index)
        historic_storage = subset_timeseries(historic_storage['Total'], start_date, end_date)
        historic_storage *= 100 / capacities['combined']
        if historic_storage.shape[0] > 10:
            ax.plot(historic_storage, color='k', ls=':', zorder=2, lw=2,  label='Observed Storage')

    line_color = colordict[pywr_model_color]
    modeled_storage = subset_timeseries(storages[pywr_model][reservoir_list_nyc], start_date, end_date).sum(axis=1)
    modeled_storage *= 100/capacities['combined']
    # ax.plot(modeled_storage, color='k', ls='-', zorder=2, lw=1,  label=model_label_dict[pywr_model])
    ax.plot(modeled_storage, color='k', ls='-', zorder=2, lw=1.7)
    ax.plot(modeled_storage, color=line_color, ls='-', label=f'{model_label_dict[pywr_model]} Storage', zorder=2, lw=1.4)

    ### clean up figure
    ax.set_xlim([start_date, end_date + datetime.timedelta(days=-1)])
    # ax.set_xticks(ax.get_xticks(), ['']*len(ax.get_xticks()), fontsize=fontsize)   # Not needed with sharex=True
    ax.set_yticks(ax.get_yticks(), ax.get_yticklabels(), fontsize=fontsize)
    ax.set_ylabel('Combined NYC Storage (%)', fontsize=fontsize)
    ax.set_ylim([0,100])
    
    # ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.1,0.5), ncols=1, fontsize=fontsize)
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5,1.03), ncols=4, fontsize=fontsize)
    ax.annotate(labels[0], xy=(0.005, 0.975), xycoords='axes fraction', ha='left', va='top', weight='bold',
                fontsize=fontsize)


    ########################################################
    # ### subfig b: first split up NYC releases into components
    ########################################################

    release_total = subset_timeseries(reservoir_releases[pywr_model][reservoir_list_nyc], start_date, end_date).sum(axis=1)
    x = release_total.index

    downstream_gage_pywr = subset_timeseries(reservoir_downstream_gages[pywr_model]['NYCAgg'], start_date, end_date)
    downstream_gage_base = subset_timeseries(reservoir_downstream_gages[base_model]['NYCAgg'], start_date, end_date)
    if use_observed:
        downstream_gage_obs = subset_timeseries(reservoir_downstream_gages['obs']['NYCAgg'], start_date, end_date)
    downstream_uncontrolled_pywr = downstream_gage_pywr - release_total

    ax2 = axs[1]
    ax2.plot(downstream_gage_pywr, color='k', lw=1.7, zorder=2)
    ax2.plot(downstream_gage_pywr, color=colordict[pywr_model_color], lw=1.4, label=f'{model_label_dict[pywr_model]} Flow', zorder=2)
    ax2.plot(downstream_gage_base, color='k', lw=1.7, zorder=1.9)
    ax2.plot(downstream_gage_base, color=colordict[base_model_color], lw=1.4, label=f'{model_label_dict[base_model]} Flow', zorder=1.9)
    if use_observed:
        if len(downstream_gage_obs) > 0:
            ax2.plot(downstream_gage_obs, color='k', ls=':', lw=1.7, label='Observed Flow', zorder=2.1)
    # ax2.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.1, 0.32), ncols=1, fontsize=fontsize)
    ax2.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.94, -1.25), ncols=1, fontsize=fontsize)

    ax2.set_xlim([x[0], x[-1]])
    ax = ax2.twinx()
    ax.set_ylim([0,100])

    if use_log:
        ax2.semilogy()
        ymax = max(downstream_gage_pywr.max(), downstream_gage_base.max())
        ymin = min(downstream_gage_pywr.min(), downstream_gage_base.min())
        if use_observed:
            ymax = max(ymax, downstream_gage_obs.max())
            ymin = max(ymin, downstream_gage_obs.min())
        for i in range(10):
            if ymin < 10 **i:
                ymin = 10 **(i-1)
                break
        for i in range(10):
            if ymax < 10 **i:
                ymax = 10 **(i)
                break
        # ax2.set_ylim([ymin, ymax])
    else:
        ax2.set_ylim([0, ax2.get_ylim()[1]])


    ### colorbrewer brown/teal palette https://colorbrewer2.org/#type=diverging&scheme=BrBG&n=4
    # colors = ['#01665e', '#35978f', '#80cdc1', '#c7eae5', '#f6e8c3', '#dfc27d', '#bf812d', '#8c510a']
    colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f6e8c3', '#dfc27d', '#bf812d', '#8c510a']
    alpha = 1

    release_components_full = subset_timeseries(nyc_release_components[pywr_model], start_date, end_date)

    release_types = ['mrf_target_individual', 'mrf_montagueTrenton', 'flood_release', 'spill']
    release_components = pd.DataFrame({release_type: release_components_full[[c for c in release_components_full.columns
                                                                              if release_type in c]].sum(axis=1)
                                       for release_type in release_types})
    release_components['uncontrolled'] = downstream_uncontrolled_pywr

    release_components = release_components.divide(downstream_gage_pywr, axis=0) * 100

    y1 = 0
    y2 = y1 + release_components[f'uncontrolled'].values
    y3 = y2 + release_components[f'mrf_montagueTrenton'].values
    y4 = y3 + release_components[f'mrf_target_individual'].values
    y5 = y4 + release_components[f'flood_release'].values
    y6 = y5 + release_components[f'spill'].values
    ax.fill_between(x, y5, y6, label='NYC Spill', color=colors[0], alpha=alpha, lw=0)
    ax.fill_between(x, y4, y5, label='NYC FFMP Flood', color=colors[1], alpha=alpha, lw=0)
    ax.fill_between(x, y3, y4, label='NYC FFMP Individual', color=colors[2], alpha=alpha, lw=0)
    ax.fill_between(x, y2, y3, label='NYC FFMP Downstream', color=colors[3], alpha=alpha, lw=0)
    ax.fill_between(x, y1, y2, label='Uncontrolled', color=colors[4], alpha=alpha, lw=0)

    ax2.set_ylabel('Total Flow (MGD)', fontsize=fontsize)
    ax.set_ylabel('Flow Contribution (%)', fontsize=fontsize)

    # ax.set_xticks(ax.get_xticks(), ['']*len(ax.get_xticks()), fontsize=fontsize) # Not needed with sharex=True

    ax.set_zorder(1)
    ax2.set_zorder(2)
    ax2.patch.set_visible(False)


    ax.set_yticks(ax.get_yticks(), ax.get_yticklabels(), fontsize=fontsize)
    ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=fontsize)
    ax.annotate(labels[1], xy=(0.005, 0.975), xycoords='axes fraction', ha='left', va='top', weight='bold',
                fontsize=fontsize)


    ########################################################
    ### subfig c: split up montague/trenton flow into components
    ########################################################
    
    # Get total sim and obs flow
    total_sim_node_flow = subset_timeseries(major_flows[pywr_model][node], start_date, end_date)
    total_base_node_flow = subset_timeseries(major_flows[base_model][node], start_date, end_date)
    if use_observed:
        total_obs_node_flow = subset_timeseries(major_flows['obs'][node], start_date, end_date)

    ### for Trenton, add NJ diversion to simulated flow. also add Blue Marsh MRF contribution for FFMP Trenton equivalent flow
    if node == 'delTrenton':
        nj_diversion = subset_timeseries(diversions[pywr_model]['delivery_nj'], start_date, end_date)
        total_sim_node_flow += nj_diversion
        # total_base_node_flow += nj_diversion
        if use_observed:
            total_obs_node_flow += nj_diversion

        ### get drbc contributions from lower basin reservoirs
        lower_basin_mrf_contributions = subset_timeseries(lower_basin_mrf_contributions[pywr_model], start_date, end_date)
        lower_basin_mrf_contributions.columns = [c.split('_')[-1] for c in lower_basin_mrf_contributions.columns]

        # acct for lag at blue marsh so it can be added to trenton equiv flow. other flows lagged below
        if node == 'delTrenton':
            for c in ['blueMarsh']:
                lag = downstream_node_lags[c]
                downstream_node = immediate_downstream_nodes_dict[c]
                while downstream_node != 'output_del':
                    lag += downstream_node_lags[downstream_node]
                    downstream_node = immediate_downstream_nodes_dict[downstream_node]
                if lag > 0:
                    lower_basin_mrf_contributions[c].iloc[lag:] = lower_basin_mrf_contributions[c].iloc[:-lag]
        total_sim_node_flow += lower_basin_mrf_contributions['blueMarsh']

    ax2 = axs[2]
    ax2.plot(total_base_node_flow, color='k', lw=1.7)
    ax2.plot(total_base_node_flow, color=colordict[base_model_color], lw=1.4)
    ax2.plot(total_sim_node_flow, color='k', lw=1.7)
    ax2.plot(total_sim_node_flow, color=colordict[pywr_model_color], lw=1.4)
    if use_observed:
        if len(total_obs_node_flow)>0:
            ax2.plot(total_obs_node_flow, color='k', ls=':', lw=1.7)
    ax = ax2.twinx()
    ax.set_ylim([0,100])
    ax.set_xlim(start_date, end_date)
    if use_log:
        ax2.semilogy()
        ymax = max(total_base_node_flow.max(), total_sim_node_flow.max())
        ymin = min(total_base_node_flow.min(), total_sim_node_flow.min())
        if use_observed:
            ymax = max(ymax, total_obs_node_flow.max())
            ymin = max(ymin, total_obs_node_flow.min())
        for i in range(10):
            if ymin < 10 ** i:
                ymin = 10 ** (i - 1)
                break
        for i in range(10):
            if ymax < 10 ** i:
                ymax = 10 ** (i)
                break
        # ax2.set_ylim([ymin, ymax])
    else:
        ax2.set_ylim([0, ax2.get_ylim()[1]])

    ax2.set_ylabel('Total Flow (MGD)', fontsize=fontsize)
    ax.set_ylabel('Flow Contribution (%)', fontsize=fontsize)


    # Get contributing flows
    contributing = upstream_nodes_dict[node]
    non_nyc_reservoirs = [i for i in contributing if (i in reservoir_list) and (i not in reservoir_list_nyc)]
    non_nyc_release_contributions = reservoir_releases[pywr_model][non_nyc_reservoirs]

    if node == 'delTrenton':
        ### subtract lower basin ffmp releases from their non-ffmp releases
        for r in drbc_lower_basin_reservoirs:
            if r != 'blueMarsh':
                non_nyc_release_contributions[r] = np.maximum(non_nyc_release_contributions[r] -
                                                                lower_basin_mrf_contributions[r], 0)

    use_inflows = [i for i in contributing if (i in majorflow_list)]
    if node == 'delMontague':
        use_inflows.append('delMontague')
    inflow_contributions = inflows[pywr_model][use_inflows] - consumptions[pywr_model][use_inflows]
    mrf_target_individuals = nyc_release_components[pywr_model][[c for c in nyc_release_components[pywr_model].columns
                                                                 if 'mrf_target_individual' in c]]
    mrf_target_individuals.columns = [c.rsplit('_',1)[1] for c in mrf_target_individuals.columns]
    mrf_montagueTrentons = nyc_release_components[pywr_model][[c for c in nyc_release_components[pywr_model].columns
                                                               if 'mrf_montagueTrenton' in c]]
    mrf_montagueTrentons.columns = [c.rsplit('_',1)[1] for c in mrf_montagueTrentons.columns]
    flood_releases = nyc_release_components[pywr_model][[c for c in nyc_release_components[pywr_model].columns if
                                                         'flood_release' in c]]
    flood_releases.columns = [c.rsplit('_',1)[1] for c in flood_releases.columns]
    spills = nyc_release_components[pywr_model][[c for c in nyc_release_components[pywr_model].columns if 'spill' in c]]
    spills.columns = [c.rsplit('_',1)[1] for c in spills.columns]



    # Impose lag
    for c in upstream_nodes_dict[node][::-1]:
        if c in inflow_contributions.columns:
            lag = downstream_node_lags[c]
            downstream_node = immediate_downstream_nodes_dict[c]
            while downstream_node != node:
                lag += downstream_node_lags[downstream_node]
                downstream_node = immediate_downstream_nodes_dict[downstream_node]
            if lag > 0:
                inflow_contributions[c].iloc[lag:] = inflow_contributions[c].iloc[:-lag]
        elif c in non_nyc_release_contributions.columns:
            lag = downstream_node_lags[c]
            downstream_node = immediate_downstream_nodes_dict[c]
            while downstream_node != node:
                lag += downstream_node_lags[downstream_node]
                downstream_node = immediate_downstream_nodes_dict[downstream_node]
            if lag > 0:
                non_nyc_release_contributions[c].iloc[lag:] = non_nyc_release_contributions[c].iloc[:-lag]
                if node == 'delTrenton' and c in drbc_lower_basin_reservoirs:
                    lower_basin_mrf_contributions[c].iloc[lag:] = lower_basin_mrf_contributions[c].iloc[:-lag]
                ### note: blue marsh lower_basin_mrf_contribution lagged above. It wont show up in upstream_nodes_dict here, so not double lagging.
        elif c in mrf_target_individuals.columns:
            lag = downstream_node_lags[c]
            downstream_node = immediate_downstream_nodes_dict[c]
            while downstream_node != node:
                lag += downstream_node_lags[downstream_node]
                downstream_node = immediate_downstream_nodes_dict[downstream_node]
            if lag > 0:
                mrf_target_individuals[c].iloc[lag:] = mrf_target_individuals[c].iloc[:-lag]
                mrf_montagueTrentons[c].iloc[lag:] = mrf_montagueTrentons[c].iloc[:-lag]
                flood_releases[c].iloc[lag:] = flood_releases[c].iloc[:-lag]
                spills[c].iloc[lag:] = spills[c].iloc[:-lag]


    inflow_contributions = subset_timeseries(inflow_contributions, start_date, end_date).sum(axis=1)

    non_nyc_release_contributions = subset_timeseries(non_nyc_release_contributions, start_date, end_date).sum(axis=1)
    if node == 'delTrenton':
        lower_basin_mrf_contributions = lower_basin_mrf_contributions.sum(axis=1)
    mrf_target_individuals = subset_timeseries(mrf_target_individuals, start_date, end_date).sum(axis=1)
    mrf_montagueTrentons = subset_timeseries(mrf_montagueTrentons, start_date, end_date).sum(axis=1)
    flood_releases = subset_timeseries(flood_releases, start_date, end_date).sum(axis=1)
    spills = subset_timeseries(spills, start_date, end_date).sum(axis=1)

    inflow_contributions = inflow_contributions.divide(total_sim_node_flow) * 100
    non_nyc_release_contributions = non_nyc_release_contributions.divide(total_sim_node_flow) * 100
    if node == 'delTrenton':
        lower_basin_mrf_contributions = lower_basin_mrf_contributions.divide(total_sim_node_flow) * 100
    mrf_target_individuals = mrf_target_individuals.divide(total_sim_node_flow) * 100
    mrf_montagueTrentons = mrf_montagueTrentons.divide(total_sim_node_flow) * 100
    flood_releases = flood_releases.divide(total_sim_node_flow) * 100
    spills = spills.divide(total_sim_node_flow) * 100



    y1 = 0
    y2 = y1 + inflow_contributions
    y3 = y2 + non_nyc_release_contributions
    if node == 'delTrenton':
        y4 = y3 + lower_basin_mrf_contributions
        y5 = y4 + mrf_montagueTrentons
    else:
        y5 = y3 + mrf_montagueTrentons
    y6 = y5 + mrf_target_individuals
    y7 = y6 + flood_releases
    y8 = y7 + spills
    assert(np.allclose(y8, 100, atol=1e-3), 'Flow components do not sum to 100%')
    ax.fill_between(x, y7, y8, label='NYC Spill', color=colors[0], alpha=alpha, lw=0)
    ax.fill_between(x, y6, y7, label='NYC FFMP Flood', color=colors[1], alpha=alpha, lw=0)
    ax.fill_between(x, y5, y6, label='NYC FFMP Individual', color=colors[2], alpha=alpha, lw=0)
    if node == 'delTrenton':
        ax.fill_between(x, y4, y5, label='NYC FFMP Downstream', color=colors[3], alpha=alpha, lw=0)
        ax.fill_between(x, y3, y4, label='Non-NYC FFMP', color=colors[6], alpha=alpha, lw=0)
    else:
        ax.fill_between(x, y3, y5, label='NYC FFMP Downstream', color=colors[3], alpha=alpha, lw=0)
    ax.fill_between(x, y2, y3, label='Non-NYC Other', color=colors[5], alpha=alpha, lw=0)
    ax.fill_between(x, y1, y2, label='Uncontrolled Flow', color=colors[4], alpha=alpha, lw=0)

    ax.legend(frameon=False, fontsize=fontsize, loc='upper center', bbox_to_anchor=(0.37, -0.15), ncols=3)
    # ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.1, 2.), ncols=1, fontsize=fontsize)

    ax.annotate(labels[2], xy=(0.005, 0.975), xycoords='axes fraction', ha='left', va='top', weight='bold',
                fontsize=fontsize)

    ax.set_zorder(1)
    ax2.set_zorder(2)
    ax2.patch.set_visible(False)
    
    ax.set_yticks(ax.get_yticks(), ax.get_yticklabels(), fontsize=fontsize)
    ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=fontsize)
    ax2.set_xticks(ax2.get_xticks(), ax2.get_xticklabels(), fontsize=fontsize)

    plt.xlim(start_date, end_date)

    filename = f'{fig_dir}NYC_release_components_combined_{pywr_model}_{node}_' + \
                f'{release_total.index.year[0]}_{release_total.index.year[-1]}' + \
                f'{"logscale" if use_log else ""}' + '.png'
                
    plt.savefig(filename,
                bbox_inches='tight', dpi=dpi)






def get_shortfall_metrics(major_flows, lower_basin_mrf_contributions, mrf_targets, ibt_demands, ibt_diversions, models_mrf, models_ibt, nodes,
                          shortfall_type='percent', shortfall_threshold=0.95, shortfall_break_length=7,
                          start_date=None, end_date=None):
    """

    """

    ###
    eps = 1e-9

    resultsdict = {}
    for j, node in enumerate(nodes):
        resultsdict[node] = {}
        models = models_mrf if j<2 else models_ibt
        for i, m in enumerate(models):
            resultsdict[node][m] = {}
            if node in majorflow_list:
                flows = subset_timeseries(major_flows[m][node], start_date, end_date)

                ### for Trenton & Pywr models, include BLue Marsh FFMP releases in Trenton Equiv flow
                if 'pywr' in m and node == 'delTrenton':
                    lower_basin_mrf = subset_timeseries(lower_basin_mrf_contributions[m],
                                                                      start_date, end_date)
                    lower_basin_mrf.columns = [c.split('_')[-1] for c in lower_basin_mrf.columns]
                    # acct for lag at blue marsh so it can be added to trenton equiv flow.
                    for c in ['blueMarsh']:
                        lag = downstream_node_lags[c]
                        downstream_node = immediate_downstream_nodes_dict[c]
                        while downstream_node != 'output_del':
                            lag += downstream_node_lags[downstream_node]
                            downstream_node = immediate_downstream_nodes_dict[downstream_node]
                        if lag > 0:
                            lower_basin_mrf[c].iloc[lag:] = lower_basin_mrf[c].iloc[:-lag]

                    flows += lower_basin_mrf['blueMarsh']

                dates = flows.index
                flows = flows.values
                thresholds = np.ones(len(flows)) * mrf_targets[models_ibt[0]][f'mrf_target_{node}'].max() * \
                             shortfall_threshold - eps
                print(f'{node} normal minimum flow target: {thresholds[0]}')
            else:
                flows = subset_timeseries(ibt_diversions[m][f'delivery_{node}'], start_date, end_date)
                dates = flows.index
                flows = flows.values
                thresholds = subset_timeseries(ibt_demands[m][f'demand_{node}'], start_date, end_date).values * \
                             shortfall_threshold - eps

            ### reliability is the fraction of time steps above threshold
            reliability = (flows > thresholds).mean()
            ### resiliency is the probability of recovering to above threshold if currently under threshold
            if reliability < 1 - eps:
                resiliency = np.logical_and(flows[:-1] < thresholds[:-1], \
                                            (flows[1:] >= thresholds[1:])).mean() / (1 - reliability)
            else:
                resiliency = np.nan
            ### convert to percents
            resultsdict[node][m]['reliability'] = reliability * 100
            resultsdict[node][m]['resiliency'] = resiliency * 100

            ### define individual events & get event-specific metrics
            durations = []  ### length of each event
            intensities = []  ### intensity of each event = avg deficit within event
            severities = []  ### severity = duration * intensity
            vulnerabilities = []  ### vulnerability = max daily deficit within event
            event_starts = []    ### define event to start with nonzero shortfall and end with the next shortfall date that preceeds shortfall_break_length non-shortfall dates.
            event_ends = []
            if reliability > eps and reliability < 1 - eps:
                duration = 0
                severity = 0
                vulnerability = 0
                in_event = False
                for i in range(len(flows)):
                    v = flows[i]
                    t = thresholds[i]
                    d = dates[i]
                    if in_event or v < t:
                        ### is this the start of a new event?
                        if not in_event:
                            event_starts.append(d)
                        ### if this is part of event, we add to metrics whether today is deficit or not
                        duration += 1
                        if shortfall_type == 'absolute':
                            s = max(t - v, 0)
                        elif shortfall_type == 'percent':
                            s = max((t - v) / t * 100, 0)
                        severity += s
                        vulnerability = max(vulnerability, s)
                        ### now check if next shortfall_break_length days include any deficits. if not, end event.
                        in_event = np.any(flows[i+1: i+1+shortfall_break_length] < \
                                          thresholds[i+1: i+1+shortfall_break_length])
                        if not in_event:
                            event_ends.append(dates[min(i+1, len(dates)-1)])
                            durations.append(duration)
                            severities.append(severity)
                            intensities.append(severity / duration)
                            vulnerabilities.append(vulnerability)
                            in_event = False
                            duration = 0
                            severity = 0
                            vulnerability = 0

            resultsdict[node][m]['event_starts'] = event_starts
            resultsdict[node][m]['event_ends'] = event_ends
            resultsdict[node][m]['durations'] = durations
            resultsdict[node][m]['severities'] = severities
            resultsdict[node][m]['intensities'] = intensities
            resultsdict[node][m]['vulnerabilities'] = vulnerabilities


    return resultsdict




###
def plot_shortfall_metrics(shortfall_metrics, models_mrf, models_ibt, nodes, shortfall_type='absolute',
                           colordict = base_model_colors, fig_dir = fig_dir,
                           print_reliabilities=True, print_events=False):
    """

    """
    ### 3x4 figure with RRV metrics (rows) & nodes (cols). Montague/Trenton use 8 models, NYC/NJ only 4.
    fig, axs = plt.subplots(4, 4, figsize=(8, 6), gridspec_kw={'width_ratios': [3,3,3,3],
                                                               'hspace':0.15, 'wspace':0.4})
    fontsize = 8
    metrics = ['reliability','durations','intensities','vulnerabilities']
    labels = [['a)','b)','c)','d)'],
              ['e)', 'f)', 'g)', 'h)'],
              ['i)', 'j)', 'k)', 'l)'],
              ['m)', 'n)', 'o)', 'p)']
              ]

    for col, node in enumerate(nodes):
        for row, metric in enumerate(metrics):
            ax = axs[row, col]
            models = models_mrf if node in majorflow_list else models_ibt
            colors = [colordict[model] for model in models]

            ### for reliability, do a bar chart
            if metric == 'reliability':
                heights = [shortfall_metrics[node][m][metric] for m in models]
                if print_reliabilities:
                    for m,v in zip(models, heights):
                        print(f'{node}, {m}, {metric}: {v}')
                bottom = 0
                positions = range(8) if len(heights) == 8 else range(4, 8)

                ### Add bars
                ax.bar(positions, heights, bottom = bottom, width=0.8, linewidth=0.5, color=colors, edgecolor='w')

            ### for other metrics, do cdfs
            else:
                for m in models:
                    values_ordered = np.sort(np.array(shortfall_metrics[node][m][metric]))[::-1]
                    x = np.arange(len(values_ordered))
                    ax.plot(x, values_ordered, color='k', lw=1.7)
                    ax.plot(x, values_ordered, color=colordict[m], lw=1.4, label=model_label_dict[m])


            ### fix ticks/labels/etc
            if row == 0:
                ax.set_ylim([80, 100])
                ax.set_yticks([80, 90, 100], [80, 90, 100], fontsize=fontsize)
                ax.set_xlim([-0.7, 7.7])
                ax.set_xticks([])
            else:
                if shortfall_type == 'absolute':
                    ax.set_ylim([0, ax.get_ylim()[1]])
                    yticks = np.zeros(len(ax.get_yticks()) + 1)
                    yticks[1:] = ax.get_yticks()
                    ax.set_yticks(yticks, [str(round(y)) for y in yticks], fontsize=fontsize)
                elif shortfall_type == 'percent':
                    ax.set_ylim([0,30])
                    ax.set_yticks([0,10,20,30], [0,10,20,30], fontsize=fontsize)
            if col == 0:
                xticks = [0, 20, 40]
            elif col == 1:
                xticks = [0, 15, 30]
            elif col == 2:
                xticks = [0, 10, 20]
            else:
                xticks = [0, 15, 30]

            if row == 3:
                ax.set_xlabel('Exceedance\n(Num. shortfall events)', fontsize=fontsize)
                ax.set_xticks(xticks, xticks, fontsize=fontsize)
                ax.set_xlim([0, ax.get_xlim()[1]])
            elif row > 0:
                ax.set_xticks(xticks, ['']*len(xticks), fontsize=fontsize)
                ax.set_xlim([0, ax.get_xlim()[1]])


            ### subfig label
            x = 0.05 if row == 0 else 0.95
            ha = 'left' if row == 0 else 'right'
            ax.annotate(labels[row][col], xy=(x,0.96), xycoords='axes fraction', fontsize=fontsize, weight='bold',
                        va='top', ha=ha)

            ### legend
            if row == 3 and col == 1:
                legend_elements = []
                for m in models_mrf:
                    legend_elements.append(Patch(facecolor=colordict[m], edgecolor='w', label=model_label_dict[m]))
                leg = ax.legend(handles=legend_elements, loc='upper center', ncols=4, bbox_to_anchor=(1.1, -0.5),
                                frameon=False, fontsize=fontsize)

    ### ylabels & legend
    axs[0,0].set_ylabel('Reliability (%)', fontsize=fontsize)
    axs[1,0].set_ylabel('Duration (days)', fontsize=fontsize)
    if shortfall_type == 'percent':
        axs[2,0].set_ylabel('Intensity\n(%)', fontsize=fontsize)
        axs[3,0].set_ylabel('Vulnerability\n(%)', fontsize=fontsize)
    elif shortfall_type == 'absolute':
        axs[2,0].set_ylabel('Intensity\n(MGD)', fontsize=fontsize)
        axs[3,0].set_ylabel('Vulnerability\n(MGD)', fontsize=fontsize)


    fig.savefig(f'{fig_dir}/shortfall_comparison.png', bbox_inches='tight', dpi=dpi)
    plt.close()

    if print_events:
        for node in nodes:
            models = models_mrf if node in majorflow_list else models_ibt
            for m in models:
                print(f"{node}, {m}, event durations, intensities, vulnerabilities: {shortfall_metrics[node][m]['durations']}, " + \
                      f"{shortfall_metrics[node][m]['intensities']}, {shortfall_metrics[node][m]['vulnerabilities']}")

    return



### Map of major DRB nodes
def make_DRB_map(fig_dir=fig_dir):
    import os
    # os.environ['USE_PYGEOS'] = '0'
    import geopandas as gpd
    from shapely import ops
    from shapely.geometry import Point, LineString, MultiLineString
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import json
    import folium
    from folium.plugins import MarkerCluster
    import contextily as cx
    import sys
    from pywrdrb.pywr_drb_node_data import obs_pub_site_matches, obs_site_matches, nhm_site_matches

    ### set crs consistent with contextily basemap
    crs = 'EPSG:3857'


    # # Reservoir data
    reservoir_data = pd.read_csv('pywrdrb/model_data/drb_model_istarf_conus.csv', sep=',')

    ### Load general shapefiles
    drb_boundary = gpd.read_file(f'{spatial_data_dir}/DRB_shapefiles/drb_bnd_polygon.shp').to_crs(crs)
    states = gpd.read_file(f'{spatial_data_dir}/states/tl_2010_us_state10.shp').to_crs(crs)
    nhd = gpd.read_file(f'{spatial_data_dir}/NHD_0204/Shape/NHDFlowline.shp').to_crs(crs)
    drc = gpd.read_file(f'{spatial_data_dir}/NHD_NJ/DRCanal.shp').to_crs(crs)

    ### load drb node info into geodataframes
    crs_nodedata = 4386
    major_nodes = gpd.read_file(f'{spatial_data_dir}/model_components/drb_model_major_nodes.csv', sep=',')
    reservoirs = major_nodes.loc[major_nodes['type'] == 'reservoir']
    reservoirs = gpd.GeoDataFrame(reservoirs.drop('geometry', axis=1),
                                  geometry=gpd.points_from_xy(reservoirs.long, reservoirs.lat,
                                                              crs=crs_nodedata)).to_crs(crs)
    flow_reqs = major_nodes.loc[major_nodes['type'] == 'regulatory']
    flow_reqs = gpd.GeoDataFrame(flow_reqs.drop('geometry', axis=1),
                                 geometry=gpd.points_from_xy(flow_reqs.long, flow_reqs.lat,
                                                             crs=crs_nodedata)).to_crs(crs)

    ### get river network from NHD
    mainstem = nhd.loc[nhd['gnis_name'] == 'Delaware River']
    ## for river/stream objects, merge rows into single geometry to avoid white space on plot
    multi_linestring = MultiLineString([ls for ls in mainstem['geometry'].values])
    merged_linestring = ops.linemerge(multi_linestring)
    mainstem = gpd.GeoDataFrame({'geometry': [merged_linestring]})

    ### get all other tributary streams containing a Pywr-DRB reservoir, or downstream of one. Note that 2 different regions have Tulpehocken Creek - use only the correct one.
    trib_names = ['West Branch Delaware River', 'East Branch Delaware River', 'Neversink River',
                  'Mongaup River', 'Lackawaxen River', 'West Branch Lackawaxen River', 'Wallenpaupack Creek',
                  'Lehigh River', 'Shohola Creek', 'Pohopoco Creek', 'Merrill Creek', 'Musconetcong River',
                  'Pohatcong Creek', 'Tohickon Creek', 'Assunpink Creek', 'Schuylkill River', 'Maiden Creek',
                  'Tulpehocken Creek', 'Still Creek', 'Little Schuylkill River',
                  'Perkiomen Creek']
    tribs = []
    for trib_name in trib_names:
        trib = nhd.loc[[(n == trib_name) and ((n != 'Tulpehocken Creek') or ('02040203' in c)) for n, c in
                        zip(nhd['gnis_name'], nhd['reachcode'])]]
        multi_linestring = MultiLineString([ls for ls in trib['geometry'].values])
        merged_linestring = ops.linemerge(multi_linestring)
        trib = gpd.GeoDataFrame({'geometry': [merged_linestring], 'name': trib_name})
        tribs.append(trib)

    ### rough lines for NYC Delaware Aqueduct system
    ### cannonsville to rondout
    lines = [LineString([Point(-75.37462, 42.065872), Point(-74.4296, 41.79926)])]
    ### pepacton to rondout
    lines.append(LineString([Point(-74.965531, 42.073603), Point(-74.4296, 41.79926)]))
    ### neversink to rondout
    lines.append(LineString([Point(-74.643266, 41.821286), Point(-74.4296, 41.79926)]))
    ### rondout to west branch reservoir
    lines.append(LineString([Point(-74.4296, 41.79926), Point(-73.69541, 41.41176)]))
    ### west branch reservoir to kensico reservoir
    lines.append(LineString([Point(-73.69541, 41.41176), Point(-73.7659656, 41.0737078)]))
    ### kensico reservoir to hillside reservoir
    lines.append(LineString([Point(-73.7659656, 41.0737078), Point(-73.8693806, 40.90715556)]))

    ### convert projection
    crs_longlat = 'EPSG:4326'
    aqueducts = gpd.GeoDataFrame({'geometry': lines}, crs=crs_longlat)
    aqueducts = aqueducts.to_crs(crs)

    ### create map figures
    fig, ax = plt.subplots(1, 1, figsize=(6, 10))
    use_basemap = True

    ### plot drb boundary
    drb_boundary.plot(ax=ax, color='none', edgecolor='k', lw=1, zorder=0.9)

    ### plot river network
    mainstem.plot(ax=ax, color='navy', lw=3, zorder=1.1)
    for trib in tribs:
        trib.plot(ax=ax, color='cornflowerblue', lw=2, zorder=1)

    ### plot reservoirs & min flow locations
    list_nyc_reservoirs = ('reservoir_cannonsville', 'reservoir_pepacton', 'reservoir_neversink')
    for r in reservoirs['name']:
        color = 'firebrick' if r in list_nyc_reservoirs else 'sandybrown'
        r_abbrev = r.split('_')[1]
        try:
            s = 50 + reservoir_data['Adjusted_CAP_MG'].loc[reservoir_data['reservoir'] == r_abbrev].iloc[0] / 1000 * 2
        except:
            s = 50
        reservoirs.loc[reservoirs['name'] == r].plot(ax=ax, color=color, edgecolor='k', markersize=s, zorder=2)
    flow_reqs.plot(ax=ax, color='mediumseagreen', edgecolor='k', markersize=250, zorder=2.1, marker='*')

    ### NYC tunnel systsem
    aqueducts.plot(ax=ax, color='darkmagenta', lw=2, zorder=1.2, ls=':')

    ### plot NJ diversion - Delaware & Raritan Canal
    drc.plot(ax=ax, color='darkmagenta', lw=2, zorder=1.2, ls=':')

    ### add state boundaries
    if use_basemap:
        states.plot(ax=ax, color='none', edgecolor='0.5', lw=0.7, zorder=0)
    else:
        states.plot(ax=ax, color='0.95', edgecolor='0.5', lw=0.7, zorder=0)

    ### map limits
    ax.set_xlim([-8.517e6, -8.197e6])  # -8.215e6])
    ax.set_ylim([4.75e6, 5.235e6])

    ### annotations
    fontsize = 10
    fontcolor = '0.5'
    plt.annotate('New York', xy=(-8.46e6, 5.168e6), ha='center', va='center', fontsize=fontsize, color=fontcolor)
    plt.annotate('Pennsylvania', xy=(-8.46e6, 5.152e6), ha='center', va='center', fontsize=fontsize, color=fontcolor)
    plt.annotate('New York', xy=(-8.245e6, 5.032e6), rotation=-31, ha='center', va='center', fontsize=fontsize,
                 color=fontcolor)
    plt.annotate('New Jersey', xy=(-8.257e6, 5.019e6), rotation=-31, ha='center', va='center', fontsize=fontsize,
                 color=fontcolor)
    plt.annotate('Pennsylvania', xy=(-8.48e6, 4.833e6), ha='center', va='center', fontsize=fontsize, color=fontcolor)
    plt.annotate('Maryland', xy=(-8.48e6, 4.817e6), ha='center', va='center', fontsize=fontsize, color=fontcolor)
    plt.annotate('Delaware', xy=(-8.43e6, 4.8e6), rotation=-85, ha='center', va='center', fontsize=fontsize,
                 color=fontcolor)
    plt.annotate('Pennsylvania', xy=(-8.359e6, 5.03e6), rotation=50, ha='center', va='center', fontsize=fontsize,
                 color=fontcolor)
    plt.annotate('New Jersey', xy=(-8.337e6, 5.024e6), rotation=50, ha='center', va='center', fontsize=fontsize,
                 color=fontcolor)

    fontcolor = 'firebrick'
    plt.annotate('Cannonsville', xy=(-8.404e6, 5.1835e6), ha='center', va='center', fontsize=fontsize, color=fontcolor,
                 fontweight='bold')
    plt.annotate('Pepacton', xy=(-8.306e6, 5.168e6), ha='center', va='center', fontsize=fontsize, color=fontcolor,
                 fontweight='bold')
    plt.annotate('Neversink', xy=(-8.268e6, 5.143e6), ha='center', va='center', fontsize=fontsize, color=fontcolor,
                 fontweight='bold')
    fontcolor = 'mediumseagreen'
    plt.annotate('Montague', xy=(-8.284e6, 5.055e6), ha='center', va='center', fontsize=fontsize, color=fontcolor,
                 fontweight='bold')
    plt.annotate('Trenton', xy=(-8.353e6, 4.894e6), ha='center', va='center', fontsize=fontsize, color=fontcolor,
                 fontweight='bold')
    fontcolor = 'darkmagenta'
    plt.annotate('NYC\nDiversion', xy=(-8.25e6, 5.085e6), ha='center', va='center', fontsize=fontsize, color=fontcolor,
                 fontweight='bold')
    plt.annotate('NJ\nDiversion', xy=(-8.315e6, 4.959e6), ha='center', va='center', fontsize=fontsize, color=fontcolor,
                 fontweight='bold')

    ### legend
    axin = ax.inset_axes([0.58, 0.006, 0.41, 0.25])
    axin.set_xlim([0, 1])
    axin.set_ylim([0, 1])
    ### mainstem
    axin.plot([0.05, 0.15], [0.93, 0.93], color='navy', lw=3)
    axin.annotate('Delaware River', xy=(0.18, 0.93), ha='left', va='center', color='k', fontsize=fontsize)
    ### tributaries
    axin.plot([0.05, 0.15], [0.83, 0.83], color='cornflowerblue', lw=2)
    axin.annotate('Tributary', xy=(0.18, 0.83), ha='left', va='center', color='k', fontsize=fontsize)
    ### DRB boundary
    axin.plot([0.05, 0.15], [0.73, 0.73], color='k', lw=1)
    axin.annotate('Basin Boundary', xy=(0.18, 0.73), ha='left', va='center', color='k', fontsize=fontsize)
    ### Diversions
    axin.plot([0.05, 0.15], [0.63, 0.63], color='darkmagenta', lw=2, ls=':')
    axin.annotate('Interbasin Transfer', xy=(0.18, 0.63), ha='left', va='center', color='darkmagenta',
                  fontweight='bold', fontsize=fontsize)
    ### Minimum flow targets
    axin.scatter([0.1], [0.53], color='mediumseagreen', edgecolor='k', s=200, marker='*')
    axin.annotate('Flow Target', xy=(0.18, 0.53), ha='left', va='center', color='mediumseagreen', fontweight='bold',
                  fontsize=fontsize)
    ### NYC Reservoirs
    axin.scatter([0.1], [0.43], color='firebrick', edgecolor='k', s=100)
    axin.annotate('NYC Reservoir', xy=(0.18, 0.43), ha='left', va='center', color='firebrick', fontweight='bold',
                  fontsize=fontsize)
    ### Non-NYC Reservoirs
    axin.scatter([0.1], [0.33], color='sandybrown', edgecolor='k', s=100)
    axin.annotate('Non-NYC Reservoir', xy=(0.18, 0.33), ha='left', va='center', color='k', fontsize=fontsize)
    ### marker size for reservoirs
    # axin.annotate('Reservoir Capacity', xy=(0.05, 0.3), ha='left', va='center', color='k', fontsize=fontsize)
    axin.scatter([0.15], [0.18], color='0.5', edgecolor='k', s=50 + 3000 / 1000 * 2)
    axin.scatter([0.45], [0.18], color='0.5', edgecolor='k', s=50 + 50000 / 1000 * 2)
    axin.scatter([0.8], [0.18], color='0.5', edgecolor='k', s=50 + 140000 / 1000 * 2)
    axin.annotate('3 BG', xy=(0.15, 0.05), ha='center', va='center', color='k', fontsize=fontsize)
    axin.annotate('50 BG', xy=(0.45, 0.05), ha='center', va='center', color='k', fontsize=fontsize)
    axin.annotate('140 BG', xy=(0.8, 0.05), ha='center', va='center', color='k', fontsize=fontsize)
    ### clean up
    axin.set_xticks([])
    axin.set_yticks([])
    axin.patch.set_alpha(0.9)

    # ### basemap - this is slow and breaks sometimes, if so just try later
    if use_basemap:
        cx.add_basemap(ax=ax, alpha=0.5, attribution_size=6)
        figname = f'{fig_dir}/static_map_withbasemap.png'
    else:
        figname = f'{fig_dir}static_map.png'

    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(figname, bbox_inches='tight', dpi=dpi)





################################################################################################
### Old figures
###############################################################################################



#
# ### 3-part figure to visualize flow: timeseries, scatter plot, & flow duration curve. Can plot observed plus 1 or 2 modeled series.
# def plot_3part_flows(results, models, node,
#                      colordict = paired_model_colors, markerdict = scatter_model_markers, start_date=None, end_date=None,
#                      uselog=False, save_fig=True, fig_dir = fig_dir):
#     """
#     Plots a 3-part figure to visualize flow data, including a timeseries plot, a scatter plot, and a flow duration curve.
#
#     Args:
#         results (dict): A dictionary containing the flow data, including observed and modeled flows.
#         models (list): A list of model names to plot. It can contain one or two model names.
#         node (str): The name of the node or location for which the flows are plotted.
#         colordict (dict, optional): A dictionary mapping model names to color codes for line and scatter plots.
#             Defaults to paired_model_colors.
#         markerdict (dict, optional): A dictionary mapping model names to marker codes for scatter plots.
#             Defaults to scatter_model_markers.
#         uselog (bool, optional): Determines whether logarithmic scale is used for plotting. Defaults to False.
#         save_fig (bool, optional): Determines whether to save the figure as a PNG file. Defaults to True.
#         fig_dir (str, optional): The directory to save the figure. Defaults to fig_dir.
#
#     Returns:
#         None
#     """
#
#     use2nd = True if len(models) > 1 else False
#     fig = plt.figure(figsize=(16, 4))
#     gs = fig.add_gridspec(1, 3, width_ratios=(2, 1, 1), wspace=0.25, hspace=0.3)
#
#     obs = subset_timeseries(results['obs'][node], start_date, end_date)
#
#     ### first fig: time series of observed & modeled flows
#     ax = fig.add_subplot(gs[0, 0])
#     for i, m in enumerate(models):
#         if use2nd or i == 0:
#             ### first plot time series of observed vs modeled
#             modeled = subset_timeseries(results[m][node], start_date, end_date)
#
#             if i == 0:
#                 ax.plot(obs, label='observed', color=colordict['obs'])
#             ax.plot(modeled, label=m, color=colordict[m])
#
#             ### get error metrics
#             if uselog:
#                 kge, r, alpha, beta = he.evaluator(he.kge, modeled, obs, transform='log')
#                 nse = he.evaluator(he.nse, modeled, obs, transform='log')
#             else:
#                 kge, r, alpha, beta = he.evaluator(he.kge, modeled, obs)
#                 nse = he.evaluator(he.nse, modeled, obs)
#             nse, kge, r, alpha, beta = round(nse[0], 2), round(kge[0], 2), round(r[0], 2), round(alpha[0], 2), round(beta[0], 2)
#
#             ### clean up fig
#             if i == 0:
#                 coords = (0.04, 0.94)
#             else:
#                 coords = (0.04, 0.88)
#             ax.annotate(f'NSE={nse}; KGE={kge}: r={r}, relvar={alpha}, bias={beta}', xy=coords, xycoords=ax.transAxes,
#                         color=colordict[m])
#             # ax.legend(loc='right')
#             ax.set_ylabel('Daily flow (MGD)')
#             ax.set_xlabel('Date')
#             if uselog:
#                 ax.semilogy()
#
#     ### second fig: scatterplot of observed vs modeled flow
#     ax = fig.add_subplot(gs[0, 1])
#     for i, m in enumerate(models):
#         ### now add scatter of observed vs modeled
#         if use2nd or i == 0:
#             ### first plot time series of observed vs modeled
#             modeled = subset_timeseries(results[m][node], start_date, end_date)
#
#             ax.scatter(obs, modeled, alpha=0.25, zorder=2, color=colordict[m], marker='x' if 'pywr' in m else 'o')
#             diagmax = min(ax.get_xlim()[1], ax.get_ylim()[1])
#             ax.plot([0, diagmax], [0, diagmax], color='k', ls='--')
#             if uselog:
#                 ax.loglog()
#             ax.set_xlabel('Observed flow (MGD)')
#             ax.set_ylabel('Modeled flow (MGD)')
#
#     ### third fig: flow duration curves
#     ax = fig.add_subplot(gs[0, 2])
#     for i, m in enumerate(models):
#         if use2nd or i == 0:
#             ### now add exceedance plot
#             def plot_exceedance(data, ax, color, label, **kwargs):
#                 df = data.sort_values()
#                 exceedance = np.arange(1., len(df) + 1.) / len(df)
#                 ax.plot(exceedance, df, color=color, label=label, **kwargs)
#
#             modeled = subset_timeseries(results[m][node], start_date, end_date)
#             if i == 0:
#                 plot_exceedance(obs, ax, color = colordict['obs'], label=model_label_dict['obs'])
#                 ax.semilogy()
#                 ax.set_xlabel('Non-exceedence')
#                 ax.set_ylabel('Daily flow (log scale, MGD)')
#
#             plot_exceedance(modeled, ax, color = colordict[m], label=model_label_dict[m])
#             ax.legend(frameon=False)
#
#
#     # plt.show()
#     if save_fig:
#         if use2nd:
#             fig.savefig(f'{fig_dir}streamflow_3plots_{models[0]}_{models[1]}_{node}.png', bbox_inches='tight', dpi = 250)
#         else:
#             fig.savefig(f'{fig_dir}streamflow_3plots_{models[0]}_{node}.png', bbox_inches='tight', dpi = 250)
#         plt.close()
#     return



#
# ###
# def plot_weekly_flow_distributions(results, models, node, colordict= paired_model_colors, fig_dir = fig_dir,
#                                    start_date=None, end_date=None):
#     """
#     Plot distributions (range and median) of weekly flows for 1 or 2 model simulation results.
#
#     Args:
#         results (dict): A dictionary containing the flow data, including observed and modeled flows.
#         models (list): A list of model names to plot. It can contain one or two model names.
#         node (str): The name of the node or location for which the flows are plotted.
#         colordict (dict, optional): A dictionary mapping model names to color codes for line and scatter plots.
#             Defaults to paired_model_colors.
#         markerdict (dict, optional): A dictionary mapping model names to marker codes for scatter plots.
#             Defaults to scatter_model_markers.
#         fig_dir (str, optional): The directory to save the figure. Defaults to fig_dir.
#
#     Returns:
#         None
#     """
#     use2nd = True if len(models) > 1 else False
#
#     fig = plt.figure(figsize=(16, 4))
#     gs = fig.add_gridspec(1, 2, wspace=0.15, hspace=0.3)
#
#     obs = subset_timeseries(results['obs'][node], start_date, end_date)
#
#     obs_resample = obs.resample('W').sum()
#     nx = len(obs_resample.groupby(obs_resample.index.week).max())
#     ymax = obs_resample.groupby(obs_resample.index.week).max().max()
#     ymin = obs_resample.groupby(obs_resample.index.week).min().min()
#     for i, m in enumerate(models):
#         modeled = subset_timeseries(results[m][node], start_date, end_date)
#         modeled_resample = modeled.resample('W').sum()
#         ymax = max(ymax, modeled_resample.groupby(modeled_resample.index.week).max().max())
#         ymin = min(ymin, modeled_resample.groupby(modeled_resample.index.week).min().min())
#
#     ### first plot time series of observed vs modeled, real scale
#     for i, m in enumerate(models):
#
#         if i == 0:
#             ax = fig.add_subplot(gs[0, 0])
#             ax.fill_between(np.arange(1, (nx+1)), obs_resample.groupby(obs_resample.index.week).max(),
#                             obs_resample.groupby(obs_resample.index.week).min(), color=colordict['obs'], alpha=0.4)
#             ax.plot(obs_resample.groupby(obs_resample.index.week).mean(), label=model_label_dict['obs'], color=colordict['obs'])
#
#         modeled = subset_timeseries(results[m][node], start_date, end_date)
#         modeled_resample = modeled.resample('W').sum()
#         ax.fill_between(np.arange(1, (nx+1)), modeled_resample.groupby(modeled_resample.index.week).max(),
#                         modeled_resample.groupby(modeled_resample.index.week).min(), color=colordict[m], alpha=0.4)
#         ax.plot(modeled_resample.groupby(modeled_resample.index.week).mean(), label=model_label_dict[m], color=colordict[m])
#
#         ax.legend(loc='upper right', frameon=False)
#         ax.set_ylabel('Weekly flow (MGW)')
#         ax.set_xlabel('Week')
#         ax.set_ylim([-0.1 * ymax, ymax * 1.1])
#
#     ### now repeat, log scale
#     for i, m in enumerate(models):
#         if i == 0:
#             ax = fig.add_subplot(gs[0, 1])
#             ax.fill_between(np.arange(1, (nx+1)), obs_resample.groupby(obs_resample.index.week).max(),
#                             obs_resample.groupby(obs_resample.index.week).min(), color=colordict['obs'], alpha=0.4)
#             ax.plot(obs_resample.groupby(obs_resample.index.week).mean(), label='observed', color=colordict['obs'])
#
#         modeled = subset_timeseries(results[m][node], start_date, end_date)
#         modeled_resample = modeled.resample('W').sum()
#         ax.fill_between(np.arange(1, (nx+1)), modeled_resample.groupby(modeled_resample.index.week).max(),
#                         modeled_resample.groupby(modeled_resample.index.week).min(), color=colordict[m], alpha=0.4)
#         ax.plot(modeled_resample.groupby(modeled_resample.index.week).mean(), label=m, color=colordict[m])
#
#         ax.set_ylim([max(ymin * 0.5, 0.01), ymax * 1.5])
#         ax.set_xlabel('Week')
#
#         ax.semilogy()
#
#     # plt.show()
#     if use2nd:
#         fig.savefig(f'{fig_dir}streamflow_weekly_{models[0]}_{models[1]}_{node}.png', bbox_inches='tight', dpi = 250)
#     else:
#         fig.savefig(f'{fig_dir}streamflow_weekly_{models[0]}_{node}.png', bbox_inches='tight', dpi = 250)
#     plt.close()
#     return
#






# ### radial plots across diff metrics/reservoirs/models.
# ### following galleries here https://www.python-graph-gallery.com/circular-barplot-with-groups
# def plot_radial_error_metrics(results_metrics, radial_models, nodes, useNonPep = True, useweap = True, usepywr = True,
#                               usemajorflows=False, fig_dir = fig_dir,
#                               colordict = base_model_colors, hatchdict = model_hatch_styles):
#     """
#     Plot radial error metrics for different models, nodes, and metrics.
#
#     Args:
#         results_metrics (pd.DataFrame): Dataframe containing error metrics.
#         radial_models (list): List of model names (str) to include in the plot.
#         nodes (list): List of node names (str) to include in the plot.
#         useNonPep (bool): Whether to include non-pepacton nodes in the plot (default: True).
#         useweap (bool): Whether to include WEAP models in the plot (default: True).
#         usepywr (bool): Whether to include PyWR models in the plot (default: True).
#         usemajorflows (bool): Whether to use major flows in the plot (default: False).
#         fig_dir (str): Directory to save the generated figure (default: fig_dir).
#         colordict (dict): Dictionary mapping model names to colors (default: base_model_colors).
#         hatchdict (dict): Dictionary mapping model names to hatch styles (default: model_hatch_styles).
#
#     Returns:
#         None
#     """
#
#     fig, axs = plt.subplots(2, 4, figsize=(16, 8), subplot_kw={"projection": "polar"})
#
#     metrics = ['nse', 'kge', 'r', 'alpha', 'beta', 'kss', 'lognse', 'logkge']
#
#     titledict = {'nse': 'NSE', 'kge': 'KGE', 'r': 'Correlation', 'alpha': 'Relative STD', 'beta': 'Relative Bias',
#                  'kss': 'K-S Statistic', 'lognse': 'LogNSE', 'logkge': 'LogKGE'}
#
#     for k, metric in enumerate(metrics):
#         row = k % 2
#         col = int(k / 2)
#         ax = axs[row, col]
#
#         pad = 1
#         groups = len(nodes)
#         angles = np.linspace(0, 2 * np.pi, len(nodes) * len(radial_models) + pad * groups, endpoint=False)
#         values = np.maximum(np.minimum(results_metrics[metric], 3), -1) - 1
#
#         colors = [colordict[model] for model in results_metrics['model']]
#         if not usepywr:
#             if not useweap:
#                 mask = [m in radial_models[-2:] for m in results_metrics['model']]
#             else:
#                 mask = [m in radial_models[-3:] for m in results_metrics['model']]
#             colors = [v if m else 'none' for m, v in zip(mask, colors)]
#         if not useNonPep:
#             mask = [r == 'pepacton' for r in results_metrics['node']]
#             colors = [v if m else 'none' for m, v in zip(mask, colors)]
#
#         edges = ['w' for model in results_metrics['model']]
#         hatches = [hatchdict[model] for model in results_metrics['model']]
#
#         width = 2 * np.pi / len(angles)
#
#         ### Obtaining the right indexes is now a little more complicated
#         offset = 0
#         idxs = []
#         groups_size = [len(radial_models)] * len(nodes)
#         for size in groups_size:
#             idxs += list(range(offset + pad, offset + size + pad))
#             offset += size + pad
#
#         ### Remove all spines
#         ax.set_frame_on(False)
#
#         ax.xaxis.grid(False)
#         ax.yaxis.grid(False)
#         ax.set_xticks([])
#         ax.set_yticks([])
#
#         ### Set limits for radial (y) axis. The negative lower bound creates the whole in the middle.
#         if metric in ['fdcm', 'r', 'kss']:
#             ax.set_ylim(-1, 0.2)
#             yrings = [-1, -0.25, -0.5, -0.75, 0]
#         elif metric in ['alpha', 'beta']:
#             yrings = [-1, -0.5, 0, 0.5, 1]
#         elif metric in ['kge', 'nse', 'logkge', 'lognse']:
#             ax.set_ylim(-2, 0.2)
#             yrings = [-2, -1.5, -1, -0.5, 0]
#
#         # Add reference lines
#         x2 = np.linspace(0, 2 * np.pi, num=50)
#         for j, y in enumerate(yrings):
#             if y == 0:
#                 ax.plot(x2, [y] * 50, color="#333333", lw=1.5, zorder=3)
#
#             ax.plot(x2, [y] * 50, color="0.8", lw=0.8, zorder=1)
#             if (np.abs(y - int(y)) < 0.001):
#                 ax.text(0, y, round(y + 1, 2), color="#333333", fontsize=12, fontweight="bold", ha="left", va="center")
#
#         for j in range(groups):
#             angle = 2 * np.pi / groups * j
#             ax.plot([angle, angle], [yrings[0], yrings[-1] + 0.1], color='0.8', zorder=1)
#
#         ### Add bars
#         ax.bar(angles[idxs], values, width=width, linewidth=0.5, color=colors, hatch=hatches, edgecolor=edges, zorder=2)
#
#         ### customization to add group annotations
#         offset = 0
#         for j, node in enumerate(nodes):
#             # Add line below bars
#             x1 = np.linspace(angles[offset + pad], angles[offset + size + pad - 1], num=50)
#
#             # Add text to indicate group
#             wedge = 360 / len(nodes)
#             rotation = -90 + wedge / 2 + wedge * j
#             if j >= 3:
#                 rotation += 180
#             if useNonPep or node == 'pepacton':
#                 fontcolor = "#333333"
#             else:
#                 fontcolor = "w"
#
#             ax.text(
#                 np.mean(x1), ax.get_ylim()[1] + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]), node_label_dict[node],
#                 color=fontcolor, fontsize=14,
#                 ha="center", va="center", rotation=rotation
#             )
#
#             offset += size + pad
#
#         ax.text(np.pi / 2, ax.get_ylim()[1] + 0.18 * (ax.get_ylim()[1] - ax.get_ylim()[0]), titledict[metric],
#                 color="#333333", fontsize=16,
#                 fontweight="bold", ha="center", va="center")
#
#     legend_elements = []
#     legend_elements.append(Line2D([0], [0], color='none', label='models'))
#     for m in radial_models[::-1]:
#         if usepywr or m in radial_models[-2:] or (useweap and m == radial_models[-3]):
#             legend_elements.append(Patch(facecolor=colordict[m], edgecolor='w', label=model_label_dict[m], hatch=hatchdict[m]))
#         else:
#             legend_elements.append(Patch(facecolor='w', edgecolor='w', label=model_label_dict[m], hatch=hatchdict[m]))
#
#     leg = plt.legend(handles=legend_elements, loc='center', bbox_to_anchor=(1.5, 1.1), frameon=False)
#     for i, text in enumerate(leg.get_texts()):
#         if not usepywr and i > 3:
#             text.set_color("w")
#         elif not useweap and i > 2:
#             text.set_color('w')
#
#     if usemajorflows:
#         filename_mod = 'combinedNodeTypes'
#     else:
#         filename_res = 'allRes' if useNonPep else 'pep'
#         if usepywr:
#             filename_mod = 'allMod_withPywr'
#         elif useweap:
#             filename_mod = 'allMod_withoutPywr'
#         else:
#             filename_mod = 'NhmNwm_withoutPywr'
#         filename_mod = filename_res + '_' + filename_mod
#
#     # plt.show()
#     fig.savefig(f'{fig_dir}/radialMetrics_{filename_mod}.png', bbox_inches='tight', dpi=300)
#     plt.close()
#     return


# def get_xQn_flow(data, x, n):
#     ### find the worst x-day rolling average each year, then get the value of this with n-year return interval
#     data_rolling = data.rolling(x).mean()[x:]
#     data_rolling_annualWorst = data_rolling.resample('A').min()
#     xQn = np.percentile(data_rolling_annualWorst, 100 / n)
#     return xQn
#
#
#
# def get_lowflow_metrics(reservoir_downstream_gages, major_flows, models, nodes, start_date=None, end_date=None):
#     """
#
#     """
#     ### compile low flow metrics across models/nodes/metrics
#     for j, node in enumerate(nodes):
#         if node in reservoir_list + ['NYCAgg']:
#             results = reservoir_downstream_gages
#         else:
#             results = major_flows
#
#         for i, m in enumerate(models):
#             resultsdict = {}
#             for timescale in ['D','M']:
#                 modeled = subset_timeseries(results[m][node], start_date, end_date)
#
#                 ### get xQn metrics
#                 for x in [1, 7, 30, 365]:
#                     for n in [10, 33]:
#                         resultsdict[f'{x}Q{n}'] = get_xQn_flow(modeled, x, n)
#
#             resultsdict['node'] = node
#             resultsdict['model'] = m
#             if i == 0 and j == 0:
#                 results_metrics = pd.DataFrame(resultsdict, index=[0])
#             else:
#                 results_metrics = results_metrics.append(pd.DataFrame(resultsdict, index=[0]))
#
#     results_metrics.reset_index(inplace=True, drop=True)
#     return results_metrics
#
#
#
#
#
#
#
# ### gridded version of multimetric/multisite/multimodel comparison of low flow metrics over full sim period
# def plot_gridded_lowflow_metrics(results_metrics, models, nodes, start_date, end_date, figstage=0, fig_dir = fig_dir):
#     """
#
#     """
#
#     fontsize = 7
#
#
#     fig, ax = plt.subplots(1,1, figsize=(6,9))
#
#     metrics = ['1Q34', '1Q10', '7Q34', '7Q10', '30Q34', '30Q10', '365Q34', '365Q10']
#
#
#
#     ### individual colormap for each location
#     sm_dict = {}
#     vrange_dict = {}
#     scale_dict = {}
#     # print(results_metrics)
#     for node in nodes:
#         vmin = results_metrics.loc[results_metrics['node'] == node, :].iloc[:,:-2].min().min()
#         vmax = results_metrics.loc[results_metrics['node'] == node, :].iloc[:,:-2].max().max()
#
#         ### separate color scale for each location, used across metrics & models
#         vrange_dict[node] = (vmin, vmax)
#         # ### now expand slightly for colormap at purple end so that colors aren't too dark to read text
#         vmax += (vmax - vmin) * 0.15
#         ### norm/cmap
#         cmaps = [cm.get_cmap('viridis_r')]
#         norms = [mpl.colors.Normalize(vmin=vmin, vmax=vmax)]
#         scale_dict[node] = 'linear'
#
#         sm_dict[node] = [cm.ScalarMappable(cmap=cmap, norm=norm) for cmap, norm in zip(cmaps, norms)]
#
#     ### Add node-specific colorbars annotate metrics
#     num_gradations = 200
#     has_upper_extension = False
#     for x, node in enumerate(nodes):
#         print(node, vrange_dict[node])
#         vs = np.arange(vrange_dict[node][0], vrange_dict[node][1]+0.001,
#                            (vrange_dict[node][1]-vrange_dict[node][0]) / num_gradations)
#         dy = 3 / num_gradations
#         ys = np.arange(101, 104+0.001, dy)
#         for v,y in zip(vs, ys):
#             v_color = min(max(v, vrange_dict[node][0]), vrange_dict[node][1])
#             cmap_idx = 0
#
#             c = sm_dict[node][cmap_idx].to_rgba(v_color)
#             box = [Rectangle((x, y), 1, dy)]
#             pc = PatchCollection(box, facecolor=c, lw=0.0, zorder=1)
#             ax.add_collection(pc)
#             ### annotation in square based on metric range
#             dy_dict = {ys[0]: -0.5, ys[-1]: 0.4}
#
#             if y in dy_dict:
#                 ax.annotate(text=f'{v_color:.2f}', xy=(x + 0.5, y + dy_dict[y]), ha='center', va='center',
#                             fontsize=fontsize, color='k', annotation_clip=False, zorder=3)
#
#         ### add outer boundary
#         for lines in [[(x,x), (101,104)],[(x+1,x+1), (101,104)]]:
#             ax.plot(lines[0], lines[1], lw=0.5, color='0.8', zorder=2)
#
#         for lines in [[(x, x+1), (101, 101)]]:
#             ax.plot(lines[0], lines[1], lw=0.5, color='0.8', zorder=2)
#
#         for lines in [[(x, x+1), (104, 104)]]:
#             ax.plot(lines[0], lines[1], lw=0.5, color='0.8', zorder=2)
#
#
#     ### annotate color scale
#     ax.annotate(text='Color\nScale', xy=(-0.2, 102.5), rotation=90, ha='right', va='center', ma='center',
#                 fontsize=fontsize+1, color='k', annotation_clip=False)
#
#
#     ### loop over models, nodes, & metrics. Color squares based on metrics, plus annotations.
#     y = 100
#     for i,model in enumerate(models):
#         y -= 1
#         for j,metric in enumerate(metrics):
#             y -= 1
#             for x,node in enumerate(nodes):
#                 ### color square based on metric value
#                 v = results_metrics[metric].loc[np.logical_and(results_metrics['node']==node,
#                                                                results_metrics['model']==model)].values[0]
#                 v_color = min(max(v, vrange_dict[node][0]), vrange_dict[node][1])
#
#                 cmap_idx = 0
#
#                 c = sm_dict[node][cmap_idx].to_rgba(v_color)
#                 box = [Rectangle((x,y), 1, 1)]
#                 pc = PatchCollection(box, facecolor=c, edgecolor='0.8', lw=0.5)
#                 ax.add_collection(pc)
#                 ### annotation in square based on metric value
#                 ax.annotate(text=f'{v:.2f}', xy=(x+0.5, y+0.45), ha='center', va='center',
#                             fontsize=fontsize, color='k', annotation_clip=False)
#             ### annotate metric
#             ax.annotate(text=metric, xy=(x+1.1, y+0.5), ha='left', va='center',
#                         fontsize=fontsize, color='k', annotation_clip=False)
#         ### annotate model
#         ax.annotate(text=model_label_dict[model].replace('-','-\n'),
#                     xy=(-0.2, y+len(nodes)/2), rotation=90, ha='right', va='center', ma='center',
#                     fontsize=fontsize+1, color='k', annotation_clip=False)
#
#     ### annotate locations
#     for x, node in enumerate(nodes):
#         ax.annotate(text=node, xy=(x+0.5, y-0.5), rotation=90, ha='center', va='top',
#                     fontsize=fontsize+1, color='k', annotation_clip=False)
#     # ax.annotate(text='Node', xy=(10, y-1.8), ha='center', va='top',
#     #             fontsize=fontsize+1, color='k', annotation_clip=False)
#
#     ### clean up
#     ax.set_xlim([-1,x+2])
#     ax.set_ylim([y-1,105])
#     ax.spines[['right', 'left', 'bottom', 'top']].set_visible(False)
#     ax.set_xticks([])
#     ax.set_yticks([])
#
#
#     fig.savefig(f'{fig_dir}/griddedLowFlowMetrics_{start_date.year}_{end_date.year}_{figstage}.png',
#                 bbox_inches='tight', dpi=300)
#     plt.close()
#     return



### Note: There were some issues with unaccounted flows in plot_flow_contributions() that have been fixed in
###       plot_NYC_release_components_combined() below. If you want to use this figure make sure to read through those
###       updates and add here. -ALH, 8/25/2023
# def plot_flow_contributions(reservoir_releases, major_flows, inflows, model, node, start_date=None, end_date=None,
#                             upstream_nodes_dict = upstream_nodes_dict,
#                             downstream_node_lags= downstream_node_lags,
#                             reservoir_list = reservoir_list,
#                             log_flows=True,
#                             smoothing=True, smoothing_window=7,
#                             fig_dir = fig_dir,
#                             ):
#     """
#     Plot flow contributions at a specific node for a given model.
#
#     Args:
#         reservoir_releases (dict): Dictionary of reservoir releases data for different models.
#         major_flows (dict): Dictionary of major flows data.
#         model (str): Name of the model.
#         node (str): Name of the node.
#         start_date (str): Start date of the plot in 'YYYY-MM-DD' format.
#         end_date (str): End date of the plot in 'YYYY-MM-DD' format.
#         upstream_nodes_dict (dict): Dictionary mapping nodes to their upstream contributing nodes (optional).
#         reservoir_list (list): List of reservoir names (optional).
#         majorflow_list (list): List of major flow names (optional).
#         percentage_flow (bool): Whether to plot flow contributions as percentages (optional).
#         plot_target (bool): Whether to plot the flow target line (optional).
#         fig_dir (str): Directory to save the figure (optional).
#         input_dir (str): Directory to load input data (optional).
#
#     Returns:
#         None
#     """
#
#     # Get contributions
#     contributing = upstream_nodes_dict[node]
#     non_nyc_reservoirs = [i for i in contributing if (i in reservoir_list) and (i not in reservoir_list_nyc)]
#
#     use_releases = [i for i in contributing if i in reservoir_list]
#     use_inflows = [i for i in contributing if (i in majorflow_list)]
#     if node == 'delMontague':
#         use_inflows.append('delMontague')
#
#     title_text = 'Contributing flows at Trenton' if (node == 'delTrenton') else 'Contributing flows at Montague'
#     if node == 'delMontague':
#         target = 1750*cfs_to_mgd
#     elif node == 'delTrenton':
#         target = 3000*cfs_to_mgd
#     else:
#         print('Invalid node specification. Options are "delMontague" and "delTrenton"')
#
#     ## Pull just contributing data
#     release_contributions = reservoir_releases[model][use_releases]
#     inflow_contributions = inflows[model][use_inflows]
#     contributions = pd.concat([release_contributions, inflow_contributions], axis=1)
#
#     # Impose lag
#     for c in upstream_nodes_dict[node][::-1]:
#         if c in contributions.columns:
#             lag= downstream_node_lags[c]
#             downstream_node = immediate_downstream_nodes_dict[c]
#
#             while downstream_node not in ['delDRCanal', 'delTrenton', 'output_del']:
#                 if node == 'delDRCanal':
#                     break
#                 lag += downstream_node_lags[downstream_node]
#                 downstream_node = immediate_downstream_nodes_dict[downstream_node]
#
#             if lag > 0:
#                 contributions[c].iloc[lag:] = contributions[c].iloc[:-lag]
#
#     contributions = subset_timeseries(contributions, start_date, end_date)
#
#     # Get total sim and obs flow
#     total_obs_node_flow = subset_timeseries(major_flows['obs'][node], start_date, end_date)
#     total_sim_node_flow = subset_timeseries(major_flows[model][node], start_date, end_date)
#
#     #there shouldn't be any unaccounted flows, but leave here just in case
#     unaccounted_flow = (total_sim_node_flow - contributions.sum(axis=1)).divide(total_sim_node_flow, axis=0)*100
#     assert unaccounted_flow.max() < 0.01
#
#     contributions = contributions.divide(total_sim_node_flow, axis =0) * 100
#     contributions[contributions<0] = 0
#
#     ## Plotting
#     nyc_color = 'midnightblue'
#     other_reservoir_color = 'darkcyan'
#     upstream_inflow_color = 'lightsteelblue'
#     obs_flow_color = 'red'
#
#     fig, axes = plt.subplots(nrows=2, ncols=1,
#                            figsize=(8, 5), dpi =200,
#                            sharex=True,
#                            gridspec_kw={'height_ratios': [1, 1.5], 'wspace': 0.05})
#     ax1= axes[0]
#     ax2= axes[1]
#
#     ts = contributions.index
#
#     B = contributions[use_inflows].sum(axis=1) + unaccounted_flow
#     C = contributions[non_nyc_reservoirs].sum(axis=1) + B
#     D = contributions[reservoir_list_nyc].sum(axis=1) + C
#     if smoothing:
#         B = B.rolling(window=smoothing_window).mean()
#         C = C.rolling(window=smoothing_window).mean()
#         D = D.rolling(window=smoothing_window).mean()
#
#         total_sim_node_flow = total_sim_node_flow.rolling(window=7).mean()
#
#     # Total flows and target flow
#     ax1.hlines(target, ts[0], ts[-1], linestyle = 'dotted', color = 'maroon', alpha = 0.85, label = f'Flow target {target:.0f} (MGD)')
#     ax1.plot(ts, total_sim_node_flow.loc[ts], color = 'dodgerblue', label = 'Sim. Flow')
#     ax1.plot(ts, total_obs_node_flow.loc[ts], color = 'black', ls='dashed', label = 'Obs. Flow')
#     # ax1.fill_between(ts, total_sim_node_flow.loc[ts], target, where=(total_sim_node_flow.loc[ts] < target), color='red', alpha=0.5)
#
#     ax1.set_ylabel('Flow (MGD)', fontsize=14)
#     if log_flows:
#         ax1.set_yscale('log')
#     ax1.set_ylim([1000,100000])
#
#     # plot percent contribution
#     # ax.fill_between(ts, A, color = node_inflow_color, label = 'Direct node inflow')
#
#     ax2.fill_between(ts, B, color = upstream_inflow_color, label = 'Unmanaged Flows')
#     ax2.fill_between(ts, C, B, color = other_reservoir_color, label = 'Non-NYC Reservoir Releases')
#     ax2.fill_between(ts, D, C, color = nyc_color, label = 'NYC Reservoir Releases')
#
#     ax2.set_ylabel('Contributions (%)', fontsize=14)
#     ax2.set_ylim([0,100])
#
#     # Create legend
#     handles1, labels1 = ax1.get_legend_handles_labels()
#     handles2, labels2 = ax2.get_legend_handles_labels()
#     handles = handles1 + handles2
#     labels = labels1 + labels2
#     plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.0, 0.9))
#
#     title = f'{fig_dir}/flow_contributions_{node}_{model}_{contributions.index.year[0]}_{contributions.index.year[-1]}'
#
#     plt.xlim([contributions.index[0], contributions.index[-1]])
#     plt.xlabel('Date')
#     fig.align_labels()
#     plt.savefig(f'{title}.png', bbox_inches='tight', dpi=300)
#     plt.close()
#     return




#
# ###
# def get_RRV_metrics(results, models, nodes, start_date=None, end_date=None):
#     """
#     Calculate measures of reliability, resilience, and vulnerability based on Hashimoto et al. (1982) WRR.
#
#     Args:
#         results (dict): Dictionary containing model results for different nodes.
#         models (list): List of model names (str) to include in the analysis.
#         nodes (list): List of node names (str) to include in the analysis.
#
#     Returns:
#         pd.DataFrame: DataFrame containing reliability, resiliency, and vulnerability metrics for each model and node.
#     """
#     thresholds = {'delMontague': 1131.05, 'delTrenton': 1938.950669}  ### FFMP flow targets (MGD)
#     eps = 1e-9
#     thresholds = {k: v - eps for k, v in thresholds.items()}
#     for j, node in enumerate(nodes):
#         for i, m in enumerate(models):
#             modeled = subset_timeseries(results[m][node], start_date, end_date)
#
#             ### only do models with nonzero entries (eg remove some weap)
#             if np.sum(modeled) > 0:
#
#                 ### reliability is the fraction of time steps above threshold
#                 reliability = (modeled > thresholds[node]).mean()
#                 ### resiliency is the probability of recovering to above threshold if currently under threshold
#                 if reliability < 1 - eps:
#                     resiliency = np.logical_and((modeled.iloc[:-1] < thresholds[node]).reset_index(drop=True), \
#                                                 (modeled.iloc[1:] >= thresholds[node]).reset_index(drop=True)).mean() / \
#                                  (1 - reliability)
#                 else:
#                     resiliency = np.nan
#                 ### vulnerability is the expected maximum severity of a failure event
#                 if reliability > eps:
#                     max_shortfalls = []
#                     max_shortfall = 0
#                     in_event = False
#                     for i in range(len(modeled)):
#                         v = modeled.iloc[i]
#                         if v < thresholds[node]:
#                             in_event = True
#                             s = thresholds[node] - v
#                             max_shortfall = max(max_shortfall, s)
#                         else:
#                             if in_event:
#                                 max_shortfalls.append(max_shortfall)
#                                 in_event = False
#                     vulnerability = np.mean(max_shortfalls)
#                 else:
#                     vulnerability = np.nan
#
#                 resultsdict = {'reliability': reliability, 'resiliency': resiliency, 'vulnerability': vulnerability}
#
#                 resultsdict['node'] = node
#                 resultsdict['model'] = m
#                 try:
#                     rrv_metrics = rrv_metrics.append(pd.DataFrame(resultsdict, index=[0]))
#                 except:
#                     rrv_metrics = pd.DataFrame(resultsdict, index=[0])
#
#     rrv_metrics.reset_index(inplace=True, drop=True)
#     return rrv_metrics
#
#
#
#
# ###
# def plot_rrv_metrics(rrv_metrics, rrv_models, nodes, fig_dir = fig_dir,
#                      colordict = base_model_colors, hatchdict = model_hatch_styles):
#     """
#     Plot histograms of reliability, resiliency, and vulnerability for different models and nodes.
#
#     Args:
#         rrv_metrics (pd.DataFrame): DataFrame containing reliability, resiliency, and vulnerability metrics.
#         rrv_models (list): List of model names (str) to include in the plot.
#         nodes (list): List of node names (str) to include in the plot.
#         fig_dir (str): Directory to save the figure (optional).
#         colordict (dict): Dictionary mapping model names to color codes (optional).
#         hatchdict (dict): Dictionary mapping model names to hatch styles (optional).
#
#     Returns:
#         None
#     """
#
#     fig, axs = plt.subplots(2, 3, figsize=(16, 8))
#
#     metrics = ['reliability','resiliency','vulnerability']
#
#     for n, node in enumerate(nodes):
#         for k, metric in enumerate(metrics):
#             ax = axs[n, k]
#
#             colors = [colordict[model] for model in rrv_models]
#             hatches = [hatchdict[model] for model in rrv_models]
#             heights = [rrv_metrics[metric].loc[np.logical_and(rrv_metrics['node']==node, rrv_metrics['model']==model)].iloc[0] for model in rrv_models]
#             positions = range(len(heights))
#
#             ### Add bars
#             ax.bar(positions, heights, width=0.8, linewidth=0.5, color=colors, hatch=hatches, edgecolor='w', zorder=2)
#
#             ax.set_xlim([-0.5, positions[-1]+0.5])
#             if k == 0:
#                 ax.set_ylim([0.8, 1.])
#             if n>0:
#                 ax.set_xticks(positions, rrv_models, rotation=90)
#             else:
#                 ax.set_xticks(positions, ['']*len(positions))
#                 ax.set_title(metric)
#
#     legend_elements = []
#     legend_elements.append(Line2D([0], [0], color='none', label='models'))
#     for m in rrv_models:
#         legend_elements.append(Patch(facecolor=colordict[m], edgecolor='w', label=model_label_dict[m], hatch=hatchdict[m]))
#     leg = plt.legend(handles=legend_elements, loc='center', bbox_to_anchor=(1.5, 1.1), frameon=False)
#
#     fig.savefig(f'{fig_dir}/rrv_comparison.png', bbox_inches='tight', dpi=300)
#     plt.close()
#
#     return






# def compare_inflow_data(inflow_data, nodes, models, start_date = None, end_date = None, fig_dir = fig_dir):
#     """Generates a boxplot comparison of inflows are specific nodes for different datasets.
#
#     Args:
#         inflow_data (dict): Dictionary containing pd.DataFrames with inflow data.
#         nodes (list): List of nodes with inflows.
#         fig_dir (str, optional): Folder to save figures. Defaults to 'figs/'.
#     """
#
#     results = {}
#     for m in models:
#         results[m] = subset_timeseries(inflow_data[m].loc[:,nodes], start_date, end_date)
#         results[m] = results[m].assign(Dataset=model_label_dict[m])
#
#     cdf = pd.concat([results[m] for m in models])
#     mdf = pd.melt(cdf, id_vars=['Dataset'], var_name=['Node'])
#     mdf.value.name = "Inflow (MGD)"
#
#     plt.figure(figsize=(15,7))
#     ax = sns.boxplot(x="Node", y="value", hue="Dataset", data=mdf,
#                     showfliers=False, linewidth=1.2, saturation=0.8)
#     ax.set(ylim=(1, 100000))
#     ax.tick_params(axis='x', rotation=90)
#     for patch in ax.artists:
#         r,g,b,a = patch.get_facecolor()
#         patch.set_edgecolor((0,0,0,.0))
#         patch.set_facecolor((r,g,b,.0))
#     plt.yscale('log')
#     plt.savefig(f'{fig_dir}inflow_comparison_boxplot.png', bbox_inches='tight', dpi=250)
#     plt.close()
#     return



#
# def plot_combined_nyc_storage_old(storages, releases, all_drought_levels, models,
#                       start_date = '1999-10-01',
#                       end_date = '2010-05-31',
#                       reservoir = 'agg',
#                       colordict = base_model_colors,
#                       use_percent = True,
#                       plot_observed=True, plot_sim=True,
#                       add_ffmp_levels=True, ffmp_levels_to_plot=[2,5],
#                       plot_drought_levels = True,
#                       smooth_releases=False, smooth_window=7,
#                       plot_releases = True,
#                       fig_dir=fig_dir):
#     """
#     Plot simulated and observed combined NYC reservoir storage.
#
#     Args:
#         storages (dict): Dictionary of storage results from `get_pywr_results`.
#         releases (dict): Dictionary of release data.
#         models (list): List of models to plot.
#         start_date (str): Start date of the plot in 'YYYY-MM-DD' format.
#         end_date (str): End date of the plot in 'YYYY-MM-DD' format.
#         colordict (dict): Dictionary mapping model names to colors (optional).
#         use_percent (bool): Whether to plot storage as percentages of capacity (optional).
#         plot_drought_levels (bool): Whether to plot drought levels (optional).
#         plot_releases (bool): Whether to plot releases (optional).
#
#     Returns:
#         None
#     """
#
#
#
#     ffmp_level_colors = ['blue', 'blue', 'blue', 'cornflowerblue', 'green', 'darkorange', 'maroon']
#     drought_cmap = ListedColormap(ffmp_level_colors, N=7)
#     norm = plt.Normalize(0, 6)
#
#     ### get reservoir storage capacities
#     istarf = pd.read_csv(f'{model_data_dir}drb_model_istarf_conus.csv')
#     def get_reservoir_capacity(reservoir):
#         return float(istarf['Adjusted_CAP_MG'].loc[istarf['reservoir'] == reservoir].iloc[0])
#     capacities = {r: get_reservoir_capacity(r) for r in reservoir_list_nyc}
#     capacities['combined'] = sum([capacities[r] for r in reservoir_list_nyc])
#
#     historic_storage = pd.read_csv(f'{input_dir}/historic_NYC/NYC_storage_daily_2000-2021.csv', sep=',', index_col=0)
#     historic_storage.index = pd.to_datetime(historic_storage.index)
#     historic_storage = subset_timeseries(historic_storage, start_date, end_date)
#
#     historic_release = pd.read_excel(f'{input_dir}/historic_NYC/Pep_Can_Nev_releases_daily_2000-2021.xlsx', index_col=0)
#     historic_release.index = pd.to_datetime(historic_release.index)
#     historic_release = historic_release.iloc[:,:3]
#     historic_release = subset_timeseries(historic_release, start_date, end_date) * cfs_to_mgd
#     historic_release.columns = ['pepacton','cannonsville','neversink']
#     historic_release['Total'] = historic_release.sum(axis=1)
#
#     ### add seasonal min FFMP releases (table 3 https://webapps.usgs.gov/odrm/documents/ffmp/Appendix_A_FFMP-20180716-Final.pdf)
#     min_releases = {'agg': [95, 190], 'cannonsville': [40, 90], 'pepacton': [35, 60], 'neversink': [20, 40]}
#     historic_release['FFMP_min_release'] = min_releases[reservoir][0] * cfs_to_mgd
#     historic_release['FFMP_min_release'].loc[[m in (6,7,8) for m in historic_release.index.month]] = min_releases[reservoir][1] * cfs_to_mgd
#
#     # model_names = [m[5:] for m in models]
#     drought_levels = pd.DataFrame()
#     for model in models:
#         if reservoir == 'agg':
#             drought_levels[model] = subset_timeseries(all_drought_levels[model]['nyc'], start_date, end_date)
#         else:
#             drought_levels[model] = subset_timeseries(all_drought_levels[model][f'{reservoir}'], start_date, end_date)
#
#
#     # Create figure with m subplots
#     n_subplots = 3 if plot_releases else 2
#
#     fig = plt.figure(figsize=(8, 5), dpi=200)
#     gs = gridspec.GridSpec(nrows=n_subplots, ncols=2, width_ratios=[15, 1], height_ratios=[1, 3, 2], wspace=0.05)
#
#     ## Plot drought levels
#     if plot_drought_levels:
#         ax1 = fig.add_subplot(gs[0, 0])
#         ax_cbar = fig.add_subplot(gs[0, 1])
#         sns.heatmap(drought_levels.transpose(), cmap = drought_cmap,
#                     ax = ax1, norm=norm,
#                     cbar_ax = ax_cbar, cbar_kws = dict(use_gridspec=False))
#         ax1.set_xticklabels([])
#         ax1.set_xticks([])
#         ax1.set_yticklabels([])
#         ax1.set_ylabel('FFMP\nLevel', fontsize=12)
#
#     # Invert the colorbar
#     if ax_cbar is not None:
#         ax_cbar.invert_yaxis()
#
#     ## Plot combined storage
#     ax2 = fig.add_subplot(gs[1, 0])
#     ax2.grid(True, which='major', axis='y')
#     for m in models:
#         if reservoir == 'agg':
#             sim_data = subset_timeseries(storages[m][reservoir_list_nyc].sum(axis=1), start_date, end_date)
#             hist_data = subset_timeseries(historic_storage['Total'], start_date, end_date)
#             total_capacity = capacities['combined']
#         else:
#             sim_data = subset_timeseries(storages[m][reservoir], start_date, end_date)
#             hist_data = subset_timeseries(historic_storage[reservoir], start_date, end_date)
#             total_capacity = capacities[reservoir]
#
#         if use_percent:
#             sim_data = sim_data / total_capacity *100
#             hist_data = hist_data / total_capacity *100
#             ylab = f'Storage\n(% Useable)'
#         else:
#             ylab = f'Storage\n(MG)'
#         if plot_sim:
#             ax2.plot(sim_data, color=colordict[m], label=f'{m}')
#     if plot_observed:
#         ax2.plot(hist_data, color=colordict['obs'], label=f'Observed')
#     datetime = sim_data.index
#
#     if add_ffmp_levels:
#         # Load profiles
#         level_profiles = pd.read_csv(f'{model_data_dir}drb_model_dailyProfiles.csv', sep=',')
#         level_profiles = level_profiles.transpose()
#         level_profiles.columns= level_profiles.iloc[0]
#         level_profiles=level_profiles[1:]
#         # Format to make datetime
#         level_profiles.index=pd.to_datetime(level_profiles.index+f'-1944',
#                                             format='%d-%b-%Y')
#         for l in ffmp_levels_to_plot:
#             d_emergency=pd.DataFrame(data= level_profiles[f'level{l}']*100,
#                                      index=pd.date_range('1944-01-01', end_date))
#             first_year_data = d_emergency[d_emergency.index.year == 1944]
#             day_of_year_to_value = {day.day_of_year: value for day, value in zip(first_year_data.index, first_year_data[f'level{l}'])}
#             d_emergency.columns=[f'level{l}']
#
#             d_emergency[f'level{l}'] = d_emergency.apply(lambda row: day_of_year_to_value[row.name.day_of_year] if np.isnan(row[f'level{l}']) else row[f'level{l}'], axis=1)
#
#             # Plot
#             ax2.plot(subset_timeseries(d_emergency, start_date, end_date),
#                      color=drought_cmap(l),ls='dashed', zorder=1, alpha = 0.3,
#                      label= f'FFMP L{l}')
#
#     ax2.set_ylabel(ylab, fontsize = 12)
#     ax2.yaxis.set_label_coords(-0.1, 0.5) # Set y-axis label position
#     ax2.set_ylim([0, 110])
#     ax2.set_xticklabels([])
#     ax2.set_xlim([start_date, end_date])
#
#     # Plot releases
#     ax3 = fig.add_subplot(gs[2,0])
#     ymax = 0
#     if plot_sim:
#         for m in models:
#             # print(m)
#             # print(releases[m][reservoir_list_nyc].max(axis=0))
#             if reservoir == 'agg':
#                 sim_data = subset_timeseries(releases[m][reservoir_list_nyc].sum(axis=1), start_date, end_date)
#                 hist_data = subset_timeseries(historic_release['Total'], start_date, end_date)
#             else:
#                 sim_data = subset_timeseries(releases[m][reservoir], start_date, end_date)
#                 hist_data = subset_timeseries(historic_release[reservoir], start_date, end_date)
#
#             sim_data.index = datetime
#             ymax = max(ymax, sim_data.max())
#             # print(sim_data.max())
#
#             if smooth_releases:
#                 rd_rolling= sim_data.rolling(window=smooth_window).mean().values
#                 rd_rolling[0:smooth_window]= sim_data.values[0:smooth_window]
#                 rd_rolling[-smooth_window:]= sim_data.values[-smooth_window:]
#
#                 ax3.plot(sim_data.index, rd_rolling, color = colordict[m], label = m, lw = 1)
#             else:
#                 ax3.plot(sim_data.index, sim_data, color = colordict[m], label = m, lw = 1)
#
#     if plot_observed:
#         ax3.plot(hist_data, color = colordict['obs'], label=f'Observed', lw = 1, zorder=3)
#     ax3.plot(historic_release['FFMP_min_release'], color ='black', ls =':', zorder=3,
#             label = f'FFMP Min. Allowable Combined Release\nAt Drought Level 5')
#
#     ax3.set_yscale('log')
#     ax3.set_ylim([10, ymax*1.3])
#     ### max total controlled+spill release suggested in FFMP
#     if reservoir == 'agg':
#         ax3.axhline(sum([get_reservoir_max_release(r, 'controlled') for r in reservoir_list_nyc]), color='k', ls=':')
#         ax3.axhline(sum([get_reservoir_max_release(r, 'flood') for r in reservoir_list_nyc]), color='k', ls=':')
#     else:
#         ax3.axhline(get_reservoir_max_release(reservoir, 'controlled'), color='k', ls=':')
#         ax3.axhline(get_reservoir_max_release(reservoir, 'flood'), color='k', ls=':')
#
#     ax3.yaxis.set_label_coords(-0.1, 0.5)
#     ax3.set_ylabel('Releases\n(MGD)', fontsize = 12)
#     ax3.set_xlabel('Date', fontsize = 12)
#     ax3.set_xlim([start_date, end_date])
#
#     # Create legend
#     handles1, labels1 = ax1.get_legend_handles_labels()
#     handles2, labels2 = ax2.get_legend_handles_labels()
#     handles3, labels3 = ax3.get_legend_handles_labels()
#     handles = handles1 + handles2 + handles3
#     labels = labels1 + labels2 + labels3
#     plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.5))
#
#     plt.xlabel('Date')
#
#     # plt.legend(loc = 'upper left', bbox_to_anchor=(0., -0.5), ncols=2)
#     plt.tight_layout()
#     fig.align_labels()
#     plt.suptitle(f'{reservoir} Reservoir Operations\nSimulated & Observed')
#     plt.savefig(f'{fig_dir}NYC_reservoir_ops_{reservoir}_{sim_data.index.year[0]}_{sim_data.index.year[-1]}.png', dpi=250)
#     # plt.show()
#     return




# def plot_xQn_grid(reservoir_downstream_gages, major_flows, models, nodes, nlist, xlist,
#                   start_date = None, end_date = None, fig_dir=fig_dir):
#     fontsize=10
#     fig = plt.figure(figsize=(14, 7))
#     gs = GridSpec(len(nodes), 5, width_ratios=[1,2.5,30,1,1], hspace=0.15)
#
#     # for ax, node in zip(axs, nodes):
#     for j,node in enumerate(nodes):
#         ax = fig.add_subplot(gs[j,2])
#         a = np.zeros((len(nlist), len(xlist), len(models)))
#         for i, model in enumerate(models):
#             count = 0
#             if node in reservoir_list:
#                 data = subset_timeseries(reservoir_downstream_gages[model][node], start_date, end_date)
#             elif node in majorflow_list:
#                 data = subset_timeseries(major_flows[model][node], start_date, end_date)
#             for nn, n in enumerate(nlist):
#                 for xx, x in enumerate(xlist):
#                     a[nn, xx, i] = get_xQn_flow(data, x, n)
#                     count += 1
#         for c in range(1, len(models)):
#             a[:, :, c] = (a[:, :, c] - a[:, :, 0]) / a[:, :, 0] * 100
#         # print(a[:,:,0].max().max(), a[:,:,1:].max().max().max(), a[:,:,0].min().min(), a[:,:,1:].min().min().min())
#
#         ### create custom cmap following https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar
#         cmap_obs = cm.get_cmap('viridis')
#         cmaplist_obs = [cmap_obs(i) for i in range(cmap_obs.N)]
#         cmap_obs = LinearSegmentedColormap.from_list('Custom cmap', cmaplist_obs, cmap_obs.N)
#         bounds_obs = np.array([0, 25, 50, 100, 250, 500, 1000, 2000, 3000, 6000])
#         norm_obs = BoundaryNorm(bounds_obs, cmap_obs.N)
#         sm_obs = cm.ScalarMappable(cmap=cmap_obs, norm=norm_obs)
#
#         cmap_mod = cm.get_cmap('RdBu')
#         cmaplist_mod = [cmap_mod(i) for i in range(cmap_mod.N)]
#         cmap_mod = LinearSegmentedColormap.from_list('Custom cmap', cmaplist_mod, cmap_mod.N)
#         bounds_mod = np.array([-500, -200, -100, -60, -30, -15, -5, 5, 15, 30, 60, 100, 200, 500])
#         norm_mod = BoundaryNorm(bounds_mod, cmap_mod.N)
#         sm_mod = cm.ScalarMappable(cmap=cmap_mod, norm=norm_mod)
#
#         xmax = 0
#         for i, model in enumerate(models):
#             for nn, n in enumerate(nlist):
#                 for xx, x in enumerate(xlist):
#                     box = [Rectangle((nn + xmax, xx), 1, 1)]
#                     if i == 0:
#                         color = sm_obs.to_rgba(a[nn, xx, i])
#                     else:
#                         color = sm_mod.to_rgba(a[nn, xx, i])
#                     pc = PatchCollection(box, facecolor=color, edgecolor='0.8', lw=0.5)
#                     ax.add_collection(pc)
#             ### add model labels
#             if j == 0:
#                 ax.annotate(model_label_dict[model], xy=(nn+xmax - 1, xx+1.6), annotation_clip=False, va='center', ha='center',
#                             fontsize=fontsize-1)
#                 if i == 4:
#                     ax.annotate('Model', xy=(nn + xmax - 1, xx + 2.8), annotation_clip=False, va='center',
#                                 ha='center', fontsize=fontsize+1)
#             xmax += xx + 1
#
#         ### add node labels
#         ax.annotate(node_label_full_dict[node], xy=(xmax-0.5, xx-1.5), rotation=270, annotation_clip=False, va='center',
#                     ha='center', fontsize=fontsize-1)
#         if j == 3:
#             ax.annotate('Node', xy=(xmax+0.6, xx+1), rotation=270, annotation_clip=False, va='center',
#                         ha='center', fontsize=fontsize+1)
#
#         ### add x & y labels
#         if j == len(nodes)-1:
#             ax.annotate('Low flow return period (years)', xy=(xmax/2, -1.8), annotation_clip=False,
#                         va='center', ha='center', fontsize=fontsize+1)
#
#         if j == 3:
#             ax.annotate('Rolling average (days)', xy=(-2.5, xx+1), rotation=90, annotation_clip=False,
#                         va='center', ha='center', fontsize=fontsize+1)
#
#         ### add ticks etc
#         ax.spines[['right', 'left', 'bottom', 'top']].set_visible(False)
#         ax.set_xlim([0, xmax - 1])
#         ax.set_ylim([0, xx + 1])
#         if node == nodes[-1]:
#             ax.set_xticks([v - 0.5 for v in range(xmax) if v % (len(nlist) + 1) > 0], nlist * len(models),
#                           fontsize = fontsize-1)
#         else:
#             ax.set_xticks([])
#         ax.set_yticks(np.arange(0.5, xx + 1), xlist, fontsize = fontsize-1)
#         ax.tick_params(axis='both', which='both', length=0)
#
#
#
#     ### now add colorbars
#     ax = fig.add_subplot(gs[1:5,0])
#     cb = fig.colorbar(sm_obs, cax=ax)
#     cb.ax.set_title('Observed flow\n(MGD)', fontsize = fontsize)
#     cb.ax.set_yticks(bounds_obs, fontsize = fontsize)
#
#     ax = fig.add_subplot(gs[1:5, 4])
#     cb = fig.colorbar(sm_mod, cax=ax)
#     cb.ax.set_title('Modeled flow\n(% deviation\nfrom obs)', fontsize = fontsize)
#     cb.ax.set_yticks(bounds_mod, fontsize = fontsize)
#     cb.ax.set_ylim([-60, cb.ax.get_ylim()[1]])
#
#     plt.savefig(f'{fig_dir}xQn_grid.png', bbox_inches='tight', dpi=250)
#     # plt.close()
#
#
# def get_fdc(data):
#     df = data.sort_values()
#     fdc_x = np.arange(1., len(df) + 1.) / len(df)
#     fdc_y = df.values
#     return fdc_x, fdc_y
#
#
# def plot_monthly_boxplot_fdc_combined(reservoir_downstream_gages, major_flows, base_models, pywr_models, node,
#                                       colordict = base_model_colors, start_date = None, end_date = None, fig_dir=fig_dir):
#     ### plot monthly FDCs & boxplots for raw inputs (top) vs pywr (bottom) - show 4 months only
#
#     alpha = 0.9
#
#     fig, axs = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'hspace': 0.05})
#     for ax, model_sets in zip(axs, [base_models, ['obs']+pywr_models]):
#         xpartitions = []
#         for i, model in enumerate(model_sets):
#             if node in reservoir_list:
#                 flow = subset_timeseries(reservoir_downstream_gages[model][node], start_date, end_date)
#             elif node in majorflow_list:
#                 flow = subset_timeseries(major_flows[model][node], start_date, end_date)
#             flow_monthly_q01 = flow.groupby(flow.index.month).apply(np.quantile, 0.01)
#             flow_monthly_q25 = flow.groupby(flow.index.month).apply(np.quantile, 0.25)
#             flow_monthly_q50 = flow.groupby(flow.index.month).apply(np.quantile, 0.50)
#             flow_monthly_q75 = flow.groupby(flow.index.month).apply(np.quantile, 0.75)
#             flow_monthly_q99 = flow.groupby(flow.index.month).apply(np.quantile, 0.99)
#
#             for midx in range(1, 5):
#                 m = 3 * midx - 2
#                 dx = 0.05
#                 mx = midx + (dx + 0.04) * i
#                 ax.plot((mx + 0.01, mx + dx), (flow_monthly_q50[m], flow_monthly_q50[m]), color='k', zorder=3)
#                 ax.plot((mx + 0.01, mx + dx), (flow_monthly_q99[m], flow_monthly_q99[m]), color='k', zorder=1)
#                 ax.plot((mx + 0.01, mx + dx), (flow_monthly_q01[m], flow_monthly_q01[m]), color='k', zorder=1)
#                 ax.plot((mx + dx / 2, mx + dx / 2), (flow_monthly_q01[m], flow_monthly_q25[m]), color='k', zorder=1)
#                 ax.plot((mx + dx / 2, mx + dx / 2), (flow_monthly_q99[m], flow_monthly_q75[m]), color='k', zorder=1)
#
#                 box = [Rectangle((mx, flow_monthly_q25[m]), dx, flow_monthly_q75[m] - flow_monthly_q25[m])]
#                 pc = PatchCollection(box, facecolor=colordict[model], edgecolor='k', lw=0.5, zorder=2, alpha=alpha)
#                 ax.add_collection(pc)
#                 if i == len(base_models) - 1:
#                     if m == 1:
#                         xpartitions.append(mx + dx + (midx + 1 - mx - dx) / 2 - 1)
#                     xpartitions.append(mx + dx + (midx + 1 - mx - dx) / 2)
#         for midx in range(1, 5):
#             ax.axvline(xpartitions[midx], color='k', lw=0.5)
#
#         for i, model in enumerate(model_sets):
#             if node in reservoir_list:
#                 flow = subset_timeseries(reservoir_downstream_gages[model][node], start_date, end_date)
#             elif node in majorflow_list:
#                 flow = subset_timeseries(major_flows[model][node], start_date, end_date)
#             for midx in range(1, 5):
#                 m = 3 * midx - 2
#                 fdc_x, fdc_y = get_fdc(flow.loc[flow.index.month == m])
#                 zorder = 0 if model == 'obs' else -10 + i
#                 ax.plot(fdc_x + xpartitions[midx - 1], fdc_y, color=colordict[model], zorder=zorder, alpha=alpha, lw=2)
#
#         ax.set_xlim(xpartitions[0], xpartitions[-1])
#         ax.set_xticks([(xpartitions[midx] - 0.5) for midx in range(1, 5)],
#                       [month_dict[midx * 3 - 2] for midx in range(1, 5)])
#
#         ax.semilogy()
#
#         if model_sets == base_models:
#             ax.set_title(node)
#             ax.set_xticks([])
#             ax.set_ylabel('Input Streamflow (MGD)')
#
#         else:
#             ax.set_xticks([(xpartitions[midx] - 0.5) for midx in range(1, 5)],
#                           [month_dict[midx * 3 - 2] for midx in range(1, 5)])
#             ax.set_ylabel('Output Streamflow (MGD)')
#
#     ### reset ylim to be equal across subplots
#     ylim = [min(axs[0].get_ylim()[0], axs[1].get_ylim()[0]), max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])]
#     axs[0].set_ylim(ylim)
#     axs[1].set_ylim(ylim)
#
#     plt.savefig(f'{fig_dir}monthly_boxplot_fdc_combined_{node}.png', bbox_inches='tight', dpi=250)
#
#
#
#
#
#
# def plot_NYC_release_components_indiv(nyc_release_components, reservoir_releases, model, start_date = None,
#                                       end_date = None, use_proportional=False, use_log=False, fig_dir=fig_dir):
#     fig, axs = plt.subplots(3,1,figsize=(12,9))
#
#     ### colorbrewer brown/teal palette https://colorbrewer2.org/#type=diverging&scheme=BrBG&n=4
#     colors = ['#a6611a', '#dfc27d', '#80cdc1', '#018571']
#     alpha = 1
#
#     for ax, reservoir in zip(axs, reservoir_list_nyc):
#         release_components = subset_timeseries(nyc_release_components[model], start_date, end_date)
#         release_components = release_components[[c for c in release_components.columns if reservoir in c]]
#         release_total = subset_timeseries(reservoir_releases[model][reservoir], start_date, end_date)
#
#         if use_proportional:
#             release_components = release_components.divide(release_total, axis=0)
#
#         x = release_components[f'mrf_target_individual_{reservoir}'].index
#         y1 = 0
#
#         y2 = y1 + release_components[f'mrf_montagueTrenton_{reservoir}'].values
#         ax.fill_between(x, y1, y2, label='FFMP Tre/Mon', color=colors[0], alpha=alpha)
#         y3 = y2 + release_components[f'mrf_target_individual_{reservoir}'].values
#         ax.fill_between(x, y2, y3, label='FFMP Individual', color=colors[1], alpha=alpha)
#         y4 = y3 + release_components[f'flood_release_{reservoir}'].values
#         ax.fill_between(x, y3, y4, label='Flood', color=colors[2], alpha=alpha)
#         y5 = y4 + release_components[f'spill_{reservoir}'].values
#         ax.fill_between(x, y4, y5, label='Spill', color=colors[3], alpha=alpha)
#
#         ax.set_xlim([x[0], x[-1]])
#         if use_proportional:
#             ax.set_ylim([0,1])
#             ax2 = ax.twinx()
#             ax2.plot(release_total, color='k', lw=0.5)
#             if use_log:
#                 ax2.semilogy()
#
#         else:
#             ax.plot(release_total, color='k', lw=0.5)
#
#             if use_log:
#                 ax.semilogy()
#
#         ax.set_title(reservoir)
#         if reservoir == reservoir_list_nyc[1]:
#             ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.1, 0.5))
#             if use_proportional:
#                 ax2.set_ylabel('Total release (MGD)')
#                 ax.set_ylabel('Release component fraction')
#             else:
#                 ax.set_ylabel('Total release (MGD)')
#
#         plt.savefig(f'{fig_dir}NYC_release_components_{model}_' + \
#                     f'{release_total.index.year[0]}_{release_total.index.year[-1]}.png',
#                     bbox_inches='tight', dpi=500)
#
#
#


# def plot_combined_nyc_storage_vs_minflows_allPywrMod(storages, ffmp_level_boundaries, major_flows, mrf_targets, models,
#                                            mrf, shortfall_metrics, colordict = paired_model_colors,
#                                            start_date = '1999-10-01', end_date = '2010-05-31', fig_dir=fig_dir):
#     """
#
#     """
#
#     fig, axs= plt.subplots(5,1,figsize=(8, 10), gridspec_kw={'height_ratios':[2,1,1,1,1], 'hspace':0.1})
#
#     fontsize = 10
#     labels = ['a)','b)','c)','d)','e)']
#
#     ### subplot a: Reservoir modeled storages, no observed
#     ax = axs[0]
#     ### get reservoir storage capacities
#     istarf = pd.read_csv(f'{model_data_dir}drb_model_istarf_conus.csv')
#     def get_reservoir_capacity(reservoir):
#         return float(istarf['Adjusted_CAP_MG'].loc[istarf['reservoir'] == reservoir].iloc[0])
#     capacities = {r: get_reservoir_capacity(r) for r in reservoir_list_nyc}
#     capacities['combined'] = sum([capacities[r] for r in reservoir_list_nyc])
#
#     ffmp_level_boundaries = subset_timeseries(ffmp_level_boundaries, start_date, end_date) * 100
#     ffmp_level_boundaries['level1a'] = 100.
#
#     ### First plot FFMP levels as background color
#     levels = [f'level{l}' for l in ['1a','1b','1c','2','3','4','5']]
#     level_colors = [cm.get_cmap('Blues')(v) for v in [0.3, 0.2, 0.1]] +\
#                     ['papayawhip'] +\
#                     [cm.get_cmap('Reds')(v) for v in [0.1, 0.2, 0.3]]
#     level_alpha = [1]*3 + [1] + [1]*3
#     x = ffmp_level_boundaries.index
#     for i in range(len(levels)):
#         y0 = ffmp_level_boundaries[levels[i]]
#         if i == len(levels)-1:
#             y1 = 0.
#         else:
#             y1 = ffmp_level_boundaries[levels[i+1]]
#         ax.fill_between(x, y0, y1, color=level_colors[i], lw=0.2, edgecolor='k',
#                         alpha=level_alpha[i], zorder=1, label=levels[i])
#
#     line_colors = [colordict[m] for m in models]
#     for m,c in zip(models,line_colors):
#         modeled_storage = subset_timeseries(storages[m][reservoir_list_nyc], start_date, end_date).sum(axis=1)
#         modeled_storage *= 100/capacities['combined']
#         ax.plot(modeled_storage, color='k', ls='-', zorder=2, lw=2)
#         ax.plot(modeled_storage, color=c, ls='-', label=model_label_dict[m], zorder=2, lw=1.6)
#
#     ### clean up figure
#     ax.set_xlim([start_date, end_date + datetime.timedelta(days=-1)])
#     ax.set_xticks(ax.get_xticks(), ['']*len(ax.get_xticks()), fontsize=fontsize)
#     ax.set_yticks(ax.get_yticks(), ax.get_yticklabels(), fontsize=fontsize)
#     ax.set_ylabel('Combined NYC Storage (%)', fontsize=fontsize)
#     ax.set_ylim([0,100])
#     ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02,0.5), fontsize=fontsize)
#     ax.annotate(labels[0], xy=(0.005, 0.025), xycoords='axes fraction', ha='left', va='bottom', weight='bold',
#                 fontsize=fontsize)
#
#
#
#     ### subfigure b-e min flow and satisfaction for each of 4 pywr models
#     for i, m in enumerate(models):
#         ax = axs[i+1]
#
#         ### first plot fraction of static/max min flow satisfied on right y axis
#         target = subset_timeseries(mrf_targets[m][f'mrf_target_{mrf}'], start_date, end_date)
#         target_max = np.ones(len(target)) * target.max()
#         flow = subset_timeseries(major_flows[m][mrf], start_date, end_date)
#         satisfaction = np.minimum(flow.divide(target_max) * 100, np.ones(len(flow)) * 100)
#         # twincolor = 'sienna'
#         leftlinecolor = 'k'
#         ax.plot(satisfaction, color=leftlinecolor)
#         ylims = [45, 110] if mrf == 'delMontague' else [50,105]
#         ax.set_ylim(ylims)
#         ax.set_ylabel('Normal\nMin. Flow\nSatisfied (%)', fontsize=fontsize, color=leftlinecolor)
#         ax.set_yticks(ax.get_yticks(), ax.get_yticklabels(), fontsize=fontsize, color=leftlinecolor)
#         ax.tick_params(axis='y', colors=leftlinecolor)
#         ax.set_ylim(ylims)
#
#         ### add shortfall events
#         event_starts = shortfall_metrics[mrf][m]['event_starts']
#         event_ends = shortfall_metrics[mrf][m]['event_ends']
#         for event_start, event_end in zip(event_starts, event_ends):
#             ax.fill_between([event_start, event_end], [ylims[0]]*2, [ylims[1]]*2, color='lightcoral', alpha=1)
#
#
#         ### now plot dynamic min flow target on right y axis
#         ax2 = ax.twinx()
#         twincolor = 'sienna'
#         ax2.plot(target, color=twincolor, label='Daily')
#
#         ax2.set_ylabel('Dynamic\nMin. Flow\nTarget (MGD)', fontsize=fontsize, rotation=270,
#                        labelpad=35, color=twincolor)
#         ax2.annotate(labels[i+1], xy=(0.005, 0.05), xycoords='axes fraction', ha='left', va='bottom', weight='bold',
#                      fontsize=fontsize, color=twincolor)
#         ax2.tick_params(axis='y', colors=twincolor)
#
#         ax2.set_xlim(axs[0].get_xlim())
#         if i < 3:
#             ax2.set_xticks(ax2.get_xticks(), [''] * len(ax2.get_xticks()), fontsize=fontsize)
#         else:
#             ax2.set_xticks(ax2.get_xticks(), ax2.get_xticklabels(), fontsize=fontsize)
#         ### set right axis to correspond with left
#         ax2.set_ylim([target_max[0] * ylims[0]/100, target_max[0] * ylims[1]/100])
#
#
#
#
#         # ### set ax to be in front of ax2
#         # ax.set_zorder(ax2.get_zorder() + 1)
#         # ax.patch.set_visible(False)
#
#
#     ### save fig
#     plt.savefig(f'{fig_dir}storages_minflows_{mrf}_' + \
#                 f'{ffmp_level_boundaries.index.year[0]}_{ffmp_level_boundaries.index.year[-1]}.png',
#                 bbox_inches='tight', dpi=300)
#
#     return