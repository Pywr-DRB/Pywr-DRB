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
import hydroeval as he



# Constants
cms_to_mgd = 22.82
cm_to_mg = 264.17/1e6
cfs_to_mgd = 0.645932368556

# The USGS gage data available downstream of reservoirs
reservoir_link_pairs = {'cannonsville': '01425000',
                           'pepacton': '01417000',
                           'neversink': '01436000',
                           'mongaupeCombined': '01433500',
                           'beltzvilleCombined': '01449800',
                           'fewalter': '01447800',
                           'assunpink': '01463620',
                           'blueMarsh': '01470960'}


### 3-part figure to visualize flow: timeseries, scatter plot, & flow duration curve. Can plot observed plus 1 or 2 modeled series.
def plot_3part_flows(results, models, node, colors=['0.5', '#67a9cf', '#ef8a62'], uselog=False, fig_dir = 'figs/'):
    
    use2nd = True if len(models) > 1 else False
    fig = plt.figure(figsize=(16, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=(2, 1, 1), wspace=0.25, hspace=0.3)

    obs = results['obs'][node]

    ### first fig: time series of observed & modeled flows
    ax = fig.add_subplot(gs[0, 0])
    for i, m in enumerate(models):
        if use2nd or i == 0:
            ### first plot time series of observed vs modeled
            modeled = results[m][node]

            if i == 0:
                ax.plot(obs, label='observed', color=colors[0])
            color = colors[i + 1]
            ax.plot(modeled, label=m, color=color)

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
                        color=color)
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
            modeled = results[m][node]

            color = colors[i + 1]
            ax.scatter(obs, modeled, alpha=0.2, zorder=2, color=color)
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

            modeled = results[m][node]

            color = colors[i + 1]
            plot_exceedance(obs, ax, colors[0])
            ax.semilogy()
            ax.set_xlabel('Non-exceedence')
            ax.set_ylabel('Daily flow (log scale, MGD)')

            plot_exceedance(modeled, ax, color)

    # plt.show()
    if use2nd:
        fig.savefig(f'{fig_dir}streamflow_3plots_{models[0]}_{models[1]}_{node}.png', bbox_inches='tight')
    else:
        fig.savefig(f'{fig_dir}streamflow_3plots_{models[0]}_{node}.png', bbox_inches='tight')
    plt.close()
    return


### plot distributions of weekly flows, with & without log scale
def plot_weekly_flow_distributions(results, models, node, colors=['0.5', '#67a9cf', '#ef8a62'], fig_dir = 'figs/'):
    use2nd = True if len(models) > 1 else False

    fig = plt.figure(figsize=(16, 4))
    gs = fig.add_gridspec(1, 2, wspace=0.15, hspace=0.3)

    obs = results['obs'][node]

    obs_resample = obs.resample('W').sum()
    nx = len(obs_resample.groupby(obs_resample.index.week).max())
    ymax = obs_resample.groupby(obs_resample.index.week).max().max()
    ymin = obs_resample.groupby(obs_resample.index.week).min().min()
    for i, m in enumerate(models):
        modeled = results[m][node]
        modeled_resample = modeled.resample('W').sum()
        ymax = max(ymax, modeled_resample.groupby(modeled_resample.index.week).max().max())
        ymin = min(ymin, modeled_resample.groupby(modeled_resample.index.week).min().min())
    
    ### first plot time series of observed vs modeled, real scale
    for i, m in enumerate(models):
        color = colors[i + 1]
        if i == 0:
            ax = fig.add_subplot(gs[0, 0])
            ax.fill_between(np.arange(1, (nx+1)), obs_resample.groupby(obs_resample.index.week).max(),
                            obs_resample.groupby(obs_resample.index.week).min(), color=colors[0], alpha=0.4)
            ax.plot(obs_resample.groupby(obs_resample.index.week).mean(), label='observed', color=colors[0])

        modeled = results[m][node]
        modeled_resample = modeled.resample('W').sum()
        ax.fill_between(np.arange(1, (nx+1)), modeled_resample.groupby(modeled_resample.index.week).max(),
                        modeled_resample.groupby(modeled_resample.index.week).min(), color=color, alpha=0.4)
        ax.plot(modeled_resample.groupby(modeled_resample.index.week).mean(), label=m, color=color)

        ax.legend(loc='upper right')
        ax.set_ylabel('Weekly flow (MGW)')
        ax.set_xlabel('Week')
        ax.set_ylim([-0.1 * ymax, ymax * 1.1])

    ### now repeat, log scale
    for i, m in enumerate(models):
        color = colors[i + 1]
        if i == 0:
            ax = fig.add_subplot(gs[0, 1])
            ax.fill_between(np.arange(1, (nx+1)), obs_resample.groupby(obs_resample.index.week).max(),
                            obs_resample.groupby(obs_resample.index.week).min(), color=colors[0], alpha=0.4)
            ax.plot(obs_resample.groupby(obs_resample.index.week).mean(), label='observed', color=colors[0])

        modeled = results[m][node]
        modeled_resample = modeled.resample('W').sum()
        ax.fill_between(np.arange(1, (nx+1)), modeled_resample.groupby(modeled_resample.index.week).max(),
                        modeled_resample.groupby(modeled_resample.index.week).min(), color=color, alpha=0.4)
        ax.plot(modeled_resample.groupby(modeled_resample.index.week).mean(), label=m, color=color)

        ax.set_ylim([max(ymin * 0.5, 0.01), ymax * 1.5])
        ax.set_xlabel('Week')

        ax.semilogy()

    # plt.show()
    if use2nd:
        fig.savefig(f'{fig_dir}streamflow_weekly_{models[0]}_{models[1]}_{node}.png', bbox_inches='tight')
    else:
        fig.savefig(f'{fig_dir}streamflow_weekly_{models[0]}_{node}.png', bbox_inches='tight')
    plt.close()
    return

### radial plots across diff metrics/reservoirs/models.
### following galleries here https://www.python-graph-gallery.com/circular-barplot-with-groups
def plot_radial_error_metrics(results_metrics, radial_models, nodes, useNonPep = True, useweap = True, usepywr = True, usemajorflows=False, fig_dir = 'figs/'):

    fig, axs = plt.subplots(2, 4, figsize=(16, 8), subplot_kw={"projection": "polar"})

    metrics = ['nse', 'kge', 'r', 'alpha', 'beta', 'kss', 'lognse', 'logkge']

    colordict = {'obs_pub':'#097320', 'nhmv10': '#66c2a5', 'nwmv21': '#8da0cb', 'nwmv21_withLakes': '#8da0cb', 'WEAP_23Aug2022_gridmet': '#fc8d62',
                 'pywr_obs_pub':'#097320', 'pywr_nhmv10': '#66c2a5', 'pywr_nwmv21': '#8da0cb', 'pywr_nwmv21_withLakes': '#8da0cb',
                 'pywr_WEAP_23Aug2022_gridmet_nhmv10': '#fc8d62'}
    hatchdict = {'obs_pub': '', 'nhmv10': '', 'nwmv21': '', 'nwmv21_withLakes': '', 'WEAP_23Aug2022_gridmet': '', 'pywr_nhmv10': '///',
                 'pywr_obs_pub': '///', 'pywr_nwmv21': '///', 'pywr_nwmv21_withLakes': '///', 'pywr_WEAP_23Aug2022_gridmet_nhmv10': '///'}
    edgedict = {'obs_pub':'w', 'nhmv10': 'w', 'nwmv21': 'w', 'nwmv21_withLakes': 'w', 'WEAP_23Aug2022_gridmet': 'w',
                'pywr_obs_pub':'w', 'pywr_nhmv10': 'w', 'pywr_nwmv21': 'w', 'pywr_nwmv21_withLakes': 'w', 'pywr_WEAP_23Aug2022_gridmet_nhmv10': 'w'}
    nodelabeldict = {'pepacton': 'Pep', 'cannonsville': 'Can', 'neversink': 'Nev', 'prompton': 'Pro', 'assunpink': 'AspRes',\
                    'beltzvilleCombined': 'Bel', 'blueMarsh': 'Blu', 'mongaupeCombined': 'MonGop',\
                    'delLordville':'Lor', 'delMontague':'Mon', 'delTrenton':'Tre', 'outletAssunpink':'Asp', 'outletSchuylkill':'Sch','outletChristina':'Chr'}
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

        edges = [edgedict[model] for model in results_metrics['model']]
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
        ax.bar(
            angles[idxs], values, width=width, linewidth=0.5,
            color=colors, hatch=hatches, edgecolor=edges, zorder=2,
        )

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
            legend_elements.append(Patch(facecolor=colordict[m], edgecolor=edgedict[m], label=m, hatch=hatchdict[m]))
        else:
            legend_elements.append(Patch(facecolor='w', edgecolor='w', label=m, hatch=hatchdict[m]))

    leg = plt.legend(handles=legend_elements, loc='center', bbox_to_anchor=(1.45, 1.1), frameon=False)
    for i, text in enumerate(leg.get_texts()):
        if not usepywr and i > 3:
            text.set_color("w")
        elif not useweap and i > 2:
            text.set_color('w')

    if usemajorflows:
        filename_mod = 'majorFlows'
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


### histogram of reliability, resiliency, & vulnerability for different models & nodes
def plot_rrv_metrics(rrv_metrics, rrv_models, nodes, fig_dir = 'figs/'):

    fig, axs = plt.subplots(2, 3, figsize=(16, 8))

    metrics = ['reliability','resiliency','vulnerability']
    
    colordict = {'obs':'grey', 'obs_pub':'#097320', 'nhmv10': '#66c2a5', 'nwmv21': '#8da0cb', 'nwmv21_withLakes': '#8da0cb', 'WEAP_23Aug2022_gridmet': '#fc8d62',
                 'pywr_obs_pub':'#097320', 'pywr_nhmv10': '#66c2a5', 'pywr_nwmv21': '#8da0cb', 'pywr_nwmv21_withLakes': '#8da0cb',
                 'pywr_WEAP_23Aug2022_gridmet_nhmv10': '#fc8d62'}
    hatchdict = {'obs':'', 'obs_pub': '', 'nhmv10': '', 'nwmv21': '', 'nwmv21_withLakes': '', 'WEAP_23Aug2022_gridmet': '', 'pywr_nhmv10': '///',
                 'pywr_obs_pub': '///', 'pywr_nwmv21': '///', 'pywr_nwmv21_withLakes': '///', 'pywr_WEAP_23Aug2022_gridmet_nhmv10': '///'}

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



def plot_flow_contributions(res_releases, major_flows,
                            model, node,
                            separate_pub_contributions = False,
                            percentage_flow = True,
                            plot_target = False, 
                            fig_dir = 'figs/',
                            input_dir = 'input_data/'):

    title = f'{fig_dir}/flow_contributions_{node}_{model}'

    pub_reservoirs = ['wallenpaupack', 'shoholaMarsh', 'mongaupeCombined', 'merrillCreek', 'hopatcong', 'nockamixon',
                    'assunpink', 'ontelaunee', 'stillCreek', 'greenLane']

    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']

    site_matches_link = [['delLordville', ['01427207'], ['cannonsville', 'pepacton']],
                     ['delMontague', ['01438500'], ['cannonsville', 'pepacton', 'delLordville',
                                                    'prompton', 'wallenpaupack', 'shoholaMarsh', 'mongaupeCombined', 'neversink']],
                     ['delTrenton', ['01463500'], ['cannonsville', 'pepacton', 'delLordville',
                                                    'prompton', 'wallenpaupack', 'shoholaMarsh', 'mongaupeCombined', 'neversink', 'delMontague',
                                                   'beltzvilleCombined', 'fewalter', 'merrillCreek', 'hopatcong', 'nockamixon']],
                     ['outletAssunpink', ['01463620'], ['assunpink']], ## note, should get downstream junction, just using reservoir-adjacent gage for now
                     ['outletSchuylkill', ['01474500'], ['ontelaunee', 'stillCreek', 'blueMarsh', 'greenLane']],
                     ['outletChristina', ['01480685'], ['marshCreek']] ## note, should use ['01481500, 01480015, 01479000, 01478000'], but dont have yet. so use marsh creek gage for now.
                     ]

    # Get contributions
    contributing = []
    if node == 'delLordville':
        contributing = site_matches_link[0][2].copy()
        title_text = 'Contributing flows at Lordville'
    elif node == 'delMontague':
        contributing = site_matches_link[1][2].copy()
        title_text = 'Contributing flows at Montague'
        target = 1750*cfs_to_mgd
    elif node == 'delTrenton':
        contributing = site_matches_link[2][2].copy()
        title_text = 'Contributing flows at Trenton'
        target = 3000*cfs_to_mgd
    else:
        print('Invalid node specification.')

    # Pull just contributing data
    use_releases = [i for i in contributing if i in res_releases[model].columns]
    use_flows = [i for i in contributing if (i not in use_releases) and (i in major_flows[model].columns)]
    release_contributions = res_releases[model][use_releases]
    flow_contributions = major_flows[model][use_flows]
    contributions = pd.concat([release_contributions, flow_contributions], axis=1)

    if separate_pub_contributions:
        pub_contributions = [res for res in pub_reservoirs if res in contributing]
        gauged_other = [c for c in contributing if (c not in nyc_reservoirs) and (c not in pub_reservoirs)and (c not in major_flows[model].columns)]
        upper_basin_main_inflows = [c for c in contributing if (c in major_flows[model].columns) and (c not in [node])]
        title = f'{title}_with_pub'
    else:
        pub_contributions = []
        gauged_other = [c for c in contributing if (c not in nyc_reservoirs) and (c not in major_flows[model].columns)]
        upper_basin_main_inflows = [c for c in contributing if (c in major_flows[model].columns) and (c not in [node])]

    # Account for mainstem inflows
    if model.split('_')[0] == 'pywr':
        if len(model.split('_'))==3:
            m = f'{model.split("_")[1]}_{model.split("_")[2]}'
        else:
            m = f'{model.split("_")[1]}'
    else:
        m = model
    inflows = pd.read_csv(f'{input_dir}catchment_inflow_{m}.csv', sep = ',', index_col = 0)
    inflows.index = pd.to_datetime(inflows.index)
    inflows = inflows.loc[inflows.index >= res_releases[model].index[0]]
    inflows = inflows.loc[inflows.index <= res_releases[model].index[-1]]

    contributions[upper_basin_main_inflows] = inflows[upper_basin_main_inflows]
    contributions[node] = inflows[node]

    if percentage_flow:
        total_node_flow = major_flows['obs'][node]
        contributions = contributions.divide(total_node_flow, axis =0)
        contributions[contributions<0] = 0
        ymax = 1.25
        title = f'{title}_percentage'
    else:
        title = f'{title}_absolute'
        ymax = np.quantile(major_flows[model][node], 0.75)

    nyc_color = 'steelblue'
    pub_color = 'maroon'
    other_reservoir_color = 'lightsteelblue'
    upstream_inflow_color = 'darkcyan'
    node_inflow_color = 'midnightblue'

    # Plotting
    fig = plt.figure(figsize=(16, 4), dpi =250)
    gs = fig.add_gridspec(2, 1, wspace=0.15, hspace=0.3)
    ax = fig.add_subplot(gs[0,0])

    ts = contributions .index
    fig,ax = plt.subplots(figsize = (16,4), dpi = 250)

    if separate_pub_contributions:
        # Partition contributions
        A = contributions[node]
        B = contributions[upper_basin_main_inflows].sum(axis=1) + A
        C = contributions[gauged_other].sum(axis=1) + B
        D = contributions[pub_contributions].sum(axis=1) + C
        E = contributions[nyc_reservoirs].sum(axis=1) + D

        ax.fill_between(ts, E, D, color = nyc_color, label = 'NYC reservoir contributions')
        ax.fill_between(ts, D, C, color = pub_color, label = 'PUB reservoir contributions')
        ax.fill_between(ts, C, B, color = other_reservoir_color, label = 'Other reservoir contributions')
        ax.fill_between(ts, B, A, color = upstream_inflow_color, label = 'Direct inflow at upstream mainstem nodes')
        ax.fill_between(ts, A, color = node_inflow_color, label = 'Direct node inflow')
    else:
        A = contributions[node]
        B = contributions[upper_basin_main_inflows].sum(axis=1) + A
        C = contributions[gauged_other].sum(axis=1) + contributions[pub_contributions].sum(axis=1) + B
        E = contributions[nyc_reservoirs].sum(axis=1) + C

        ax.fill_between(ts, E, C, color = nyc_color, label = 'NYC reservoir contributions')
        ax.fill_between(ts, C, B, color = other_reservoir_color, label = 'Other reservoir contributions')
        ax.fill_between(ts, B, A, color = upstream_inflow_color, label = 'Direct inflow at upstream mainstem nodes')
        ax.fill_between(ts, A, color = node_inflow_color, label = 'Direct node inflow')


    if plot_target and not percentage_flow:
        ax.hlines(target, ts[0], ts[-1], linestyle = 'dashed', color = 'black', alpha = 0.5, label = 'Flow target')
    if percentage_flow:
        plt.ylabel('Percentage flow contributions (%)')
    else:
        plt.ylabel('Flow contributions (MGD)')
    plt.title(title_text)
    plt.ylim([0,ymax])
    plt.xlim([contributions.index[0], contributions.index[-1]])
    plt.xlabel('Date')
    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.8))
    plt.savefig(f'{title}.png', bbox_inches='tight', dpi=300)
    plt.close()
    return



def compare_inflow_data(inflow_data, nodes,
                        fig_dir = 'figs/'):
    """Generates a boxplot comparison of inflows are specific nodes for different datasets.

    Args:
        inflow_data (dict): Dictionary containing pd.DataFrames with inflow data. 
        nodes (list): List of nodes with inflows.
        fig_dir (str, optional): Folder to save figures. Defaults to 'figs/'.
    """
    
    pub_df = inflow_data['obs_pub'].loc[:,nodes]
    nhm_df = inflow_data['nhmv10'].loc[:,nodes]
    nwm_df = inflow_data['nwmv21'].loc[:,nodes]
    #weap_df = inflow_data['WEAP_23Aug2022_gridmet_nhmv10']  
    
    pub_df= pub_df.assign(Dataset='PUB')
    nhm_df=nhm_df.assign(Dataset='NHMv10')
    nwm_df=nwm_df.assign(Dataset='NWMv21')

    cdf = pd.concat([pub_df, nhm_df, nwm_df])    
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
    plt.savefig(f'{fig_dir}inflow_comparison_boxplot.png')
    plt.close()
    return


## TODO
def plot_nyc_storage():
    return