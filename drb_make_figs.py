import numpy as np
import pandas as pd
import h5py
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import hydroeval as he
from scipy import stats

### I was having trouble with interactive console plotting in Pycharm for some reason - comment this out if you want to use that and not having issues
mpl.use('TkAgg')

### directories
output_dir = 'output_data/'
input_dir = 'input_data/'
fig_dir = 'figs/'

### list of reservoirs and major flow points to compare across models
reservoir_list = ['cannonsville', 'pepacton', 'neversink', 'wallenpaupack', 'prompton', 'shoholaMarsh', \
                   'mongaupeCombined', 'beltzvilleCombined', 'fewalter', 'merrillCreek', 'hopatcong', 'nockamixon', \
                   'assunpink', 'ontelaunee', 'stillCreek', 'blueMarsh', 'greenLane', 'marshCreek']
majorflow_list = ['delLordville', 'delMontague', 'delTrenton', 'outletAssunpink', 'outletSchuylkill', 'outletChristina']

### load pywr model results. "catchment" means inflow directly into node, "outflow" means flow out of node, "reservoir" means storage
scenario = 0
def get_pywr_results(output_dir, model, results_set='all'):
    '''
    :param output_dir:
    :param model:
    :param results_set: can be "all" to return all results,
                            "res_release" to return reservoir releases (downstream gage comparison),
                            "major_flow" to return flow at major flow points of interest
    :return:
    '''
    with h5py.File(f'{output_dir}drb_output_{model}.hdf5', 'r') as f:
        keys = list(f.keys())
        first = 0
        results = pd.DataFrame()
        for k in keys:
            if results_set == 'all':
                results[k] = f[k][:,scenario]
            elif results_set == 'res_release':
                if k.split('_')[0] == 'outflow' and k.split('_')[1] in reservoir_list:
                    results[k.split('_')[1]] = f[k][:, scenario]
            elif results_set == 'major_flow':
                if k.split('_')[0] == 'link' and k.split('_')[1] in majorflow_list:
                    results[k.split('_')[1]] = f[k][:, scenario]
            # elif 'catchment' in k or 'outflow' in k or 'reservoir' in k or 'delLordville' in k or 'delMontague' in k or 'delTrenton' in k:
            #     results[k] = f[k][:,scenario]
        day = [f['time'][i][0] for i in range(len(f['time']))]
        month = [f['time'][i][2] for i in range(len(f['time']))]
        year = [f['time'][i][3] for i in range(len(f['time']))]
        date = [f'{y}-{m}-{d}' for y,m,d in zip(year, month, day)]
        date = pd.to_datetime(date)
        results.index = date
        return results

pywr_models = ['nhmv10', 'nwmv21', 'nwmv21_withLakes', 'WEAP_23Aug2022_gridmet_nhmv10']
res_releases = {}
major_flows = {}
for model in pywr_models:
    res_releases[f'pywr_{model}'] = get_pywr_results(output_dir, model, 'res_release')
    major_flows[f'pywr_{model}'] = get_pywr_results(output_dir, model, 'major_flow')
pywr_models = [f'pywr_{m}' for m in pywr_models]


### load other flow estimates. each column represents modeled flow at USGS gage downstream of reservoir or gage on mainstem
def get_base_results(input_dir, model, datetime_index, results_set='all'):
    '''
    function for retreiving & organizing results from non-pywr results (NHM, NWM, WEAP)
    :param input_dir:
    :param model:
    :param datetime_index:
    :param results_set: can be "all" to return all results,
                            "res_release" to return reservoir releases (downstream gage comparison),
                            "major_flow" to return flow at major flow points of interest
    :return:
    '''
    gage_flow = pd.read_csv(f'{input_dir}gage_flow_{model}.csv')
    gage_flow.index = pd.DatetimeIndex(gage_flow['datetime'])
    gage_flow = gage_flow.drop('datetime', axis=1)
    if results_set == 'res_release':
        for c in gage_flow.columns:
            if c not in reservoir_list:
                gage_flow = gage_flow.drop(c, axis=1)
    elif results_set == 'major_flow':
        for c in gage_flow.columns:
            if c not in majorflow_list:
                gage_flow = gage_flow.drop(c, axis=1)
    gage_flow = gage_flow.loc[datetime_index,:]
    return gage_flow

base_models = ['obs', 'nhmv10', 'nwmv21', 'WEAP_23Aug2022_gridmet']
datetime_index = list(res_releases.values())[0].index
for model in base_models:
    res_releases[model] = get_base_results(input_dir, model, datetime_index, 'res_release')
    major_flows[model] = get_base_results(input_dir, model, datetime_index, 'major_flow')

### verify that all datasets have same datetime index
for r in res_releases.values():
    assert ((r.index == datetime_index).mean() == 1)
for r in major_flows.values():
    assert ((r.index == datetime_index).mean() == 1)
print(f'successfully loaded {len(base_models)} base model results & {len(pywr_models)} pywr model results')


### 3-part figure to visualize flow: timeseries, scatter plot, & flow duration curve. Can plot observed plus 1 or 2 modeled series.
def plot_3part_flows(results, models, node, colors=['0.5', '#67a9cf', '#ef8a62'], uselog=False):
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
        fig.savefig(f'{fig_dir}streamflow_3plots_{models[0]}_{models[1]}.png', bbox_inches='tight')
    else:
        fig.savefig(f'{fig_dir}streamflow_3plots_{models[0]}.png', bbox_inches='tight')



### nhm only - slides 36-39 in 10/24/2022 presentation
plot_3part_flows(res_releases, ['nhmv10'], 'pepacton')

### nhm vs nwm - slides 40-42 in 10/24/2022 presentation
plot_3part_flows(res_releases, ['nhmv10', 'nwmv21'], 'pepacton')

### nhm vs weap (with nhm backup) - slides 60-62 in 10/24/2022 presentation
plot_3part_flows(res_releases, ['nhmv10', 'WEAP_23Aug2022_gridmet'], 'pepacton')

### nhm vs pywr-nhm - slides 60-62 in 10/24/2022 presentation
plot_3part_flows(res_releases, ['nhmv10', 'pywr_nhmv10'], 'pepacton')

## nwm vs pywr-nwm
plot_3part_flows(res_releases, ['nwmv21', 'pywr_nwmv21'], 'pepacton')

## pywr-nwm vs pywr-nwm_withLakes
plot_3part_flows(res_releases, ['pywr_nwmv21', 'pywr_nwmv21_withLakes'], 'pepacton')





### plot distributions of weekly flows, with & without log scale
def plot_weekly_flow_distributions(results, models, node, colors=['0.5', '#67a9cf', '#ef8a62']):
    use2nd = True if len(models) > 1 else False

    fig = plt.figure(figsize=(16, 4))
    gs = fig.add_gridspec(1, 2, wspace=0.15, hspace=0.3)

    obs = results['obs'][node]

    obs_resample = obs.resample('W').sum()
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
            ax.fill_between(np.arange(1, 54), obs_resample.groupby(obs_resample.index.week).max(),
                            obs_resample.groupby(obs_resample.index.week).min(), color=colors[0], alpha=0.4)
            ax.plot(obs_resample.groupby(obs_resample.index.week).mean(), label='observed', color=colors[0])

        modeled = results[m][node]
        modeled_resample = modeled.resample('W').sum()
        ax.fill_between(np.arange(1, 54), modeled_resample.groupby(modeled_resample.index.week).max(),
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
            ax.fill_between(np.arange(1, 54), obs_resample.groupby(obs_resample.index.week).max(),
                            obs_resample.groupby(obs_resample.index.week).min(), color=colors[0], alpha=0.4)
            ax.plot(obs_resample.groupby(obs_resample.index.week).mean(), label='observed', color=colors[0])

        modeled = results[m][node]
        modeled_resample = modeled.resample('W').sum()
        ax.fill_between(np.arange(1, 54), modeled_resample.groupby(modeled_resample.index.week).max(),
                        modeled_resample.groupby(modeled_resample.index.week).min(), color=color, alpha=0.4)
        ax.plot(modeled_resample.groupby(modeled_resample.index.week).mean(), label=m, color=color)

        ax.set_ylim([max(ymin * 0.5, 0.01), ymax * 1.5])
        ax.set_xlabel('Week')

        ax.semilogy()

    # plt.show()
    if use2nd:
        fig.savefig(f'{fig_dir}streamflow_weekly_{models[0]}_{models[1]}.png', bbox_inches='tight')
    else:
        fig.savefig(f'{fig_dir}streamflow_weekly_{models[0]}.png', bbox_inches='tight')



### nhm only - slides 36-39 in 10/24/2022 presentation
plot_weekly_flow_distributions(res_releases, ['nhmv10'], 'pepacton')

### nhm vs nwm - slides 35-37 in 10/24/2022 presentation
plot_weekly_flow_distributions(res_releases, ['nhmv10', 'nwmv21'], 'pepacton')

### nhm vs weap (with nhm backup) - slides 68 in 10/24/2022 presentation
plot_weekly_flow_distributions(res_releases, ['nhmv10', 'WEAP_23Aug2022_gridmet'], 'pepacton')

### nhm vs pywr-nhm - slides 68 in 10/24/2022 presentation
plot_weekly_flow_distributions(res_releases, ['nhmv10', 'pywr_nhmv10'], 'pepacton')

## nwm vs pywr-nwm
plot_weekly_flow_distributions(res_releases, ['nwmv21', 'pywr_nwmv21'], 'pepacton')

## pywr-nwm vs pywr-nwm_withLakes
plot_weekly_flow_distributions(res_releases, ['pywr_nwmv21', 'pywr_nwmv21_withLakes'], 'pepacton')




### compile error metrics across models/nodes/metrics
nodes = ['cannonsville', 'pepacton', 'neversink', 'prompton', 'beltzvilleCombined', 'blueMarsh']
radial_models = ['nhmv10', 'nwmv21', 'WEAP_23Aug2022_gridmet', 'pywr_nhmv10', 'pywr_nwmv21', 'pywr_WEAP_23Aug2022_gridmet_nhmv10']
radial_models = radial_models[::-1]

def get_error_metrics(results, models, nodes):
    ### compile error across models/nodes/metrics
    for j, node in enumerate(nodes):
        obs = results['obs'][node]
        for i, m in enumerate(models):
            # print(node, m)
            # print(results[m])
            modeled = results[m][node]

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
                               'logbeta': logbeta[0],
                               'kss': kss} #'fdcArea': fdc_area,

                resultsdict['node'] = node
                resultsdict['model'] = m
                if i == 0 and j == 0:
                    results_metrics = pd.DataFrame(resultsdict, index=[0])
                else:
                    results_metrics = results_metrics.append(pd.DataFrame(resultsdict, index=[0]))

    results_metrics.reset_index(inplace=True, drop=True)
    return results_metrics

res_release_metrics = get_error_metrics(res_releases, radial_models, nodes)


### radial plots across diff metrics/reservoirs/models.
### following galleries here https://www.python-graph-gallery.com/circular-barplot-with-groups
def plot_radial_error_metrics(results_metrics, radial_models, nodes, useNonPep = True, useweap = True, usepywr = True, usemajorflows=False):

    fig, axs = plt.subplots(2, 4, figsize=(16, 8), subplot_kw={"projection": "polar"})

    metrics = ['nse', 'kge', 'r', 'alpha', 'beta', 'kss', 'lognse', 'logkge']

    colordict = {'nhmv10': '#66c2a5', 'nwmv21': '#8da0cb', 'nwmv21_withLakes': '#8da0cb', 'WEAP_23Aug2022_gridmet': '#fc8d62',
                 'pywr_nhmv10': '#66c2a5', 'pywr_nwmv21': '#8da0cb', 'pywr_nwmv21_withLakes': '#8da0cb',
                 'pywr_WEAP_23Aug2022_gridmet_nhmv10': '#fc8d62'}
    hatchdict = {'nhmv10': '', 'nwmv21': '', 'nwmv21_withLakes': '', 'WEAP_23Aug2022_gridmet': '', 'pywr_nhmv10': '///',
                 'pywr_nwmv21': '///', 'pywr_nwmv21_withLakes': '///', 'pywr_WEAP_23Aug2022_gridmet_nhmv10': '///'}
    edgedict = {'nhmv10': 'w', 'nwmv21': 'w', 'nwmv21_withLakes': 'w', 'WEAP_23Aug2022_gridmet': 'w',
                'pywr_nhmv10': 'w', 'pywr_nwmv21': 'w', 'pywr_nwmv21_withLakes': 'w', 'pywr_WEAP_23Aug2022_gridmet_nhmv10': 'w'}
    nodelabeldict = {'pepacton': 'Pep', 'cannonsville': 'Can', 'neversink': 'Nev', 'prompton': 'Pro',\
                    'beltzvilleCombined': 'Bel', 'blueMarsh': 'Blu',\
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


### nhm vs nwm only, pepacton only - slides 48-54 in 10/24/2022 presentation
plot_radial_error_metrics(res_release_metrics, radial_models, nodes, useNonPep = False, useweap = False, usepywr = False)

### nhm vs nwm only, all reservoirs - slides 55-58 in 10/24/2022 presentation
plot_radial_error_metrics(res_release_metrics, radial_models, nodes, useNonPep = True, useweap = False, usepywr = False)

### nhm vs nwm vs weap only, pepaction only - slides 69 in 10/24/2022 presentation
plot_radial_error_metrics(res_release_metrics, radial_models, nodes, useNonPep = False, useweap = True, usepywr = False)

### nhm vs nwm vs weap only, all reservoirs - slides 70 in 10/24/2022 presentation
plot_radial_error_metrics(res_release_metrics, radial_models, nodes, useNonPep = True, useweap = True, usepywr = False)

### all models, pepaction only - slides 72-73 in 10/24/2022 presentation
plot_radial_error_metrics(res_release_metrics, radial_models, nodes, useNonPep = False, useweap = True, usepywr = True)

### all models, all reservoirs - slides 74-75 in 10/24/2022 presentation
plot_radial_error_metrics(res_release_metrics, radial_models, nodes, useNonPep = True, useweap = True, usepywr = True)

### all models, but using nwm_withLakes for pywr comparison. all reservoirs - slides 74-75 in 10/24/2022 presentation
radial_models = ['nhmv10', 'nwmv21', 'WEAP_23Aug2022_gridmet', 'pywr_nhmv10', 'pywr_nwmv21_withLakes', 'pywr_WEAP_23Aug2022_gridmet_nhmv10']
radial_models = radial_models[::-1]
res_release_metrics = get_error_metrics(res_releases, radial_models, nodes)
plot_radial_error_metrics(res_release_metrics, radial_models, nodes, useNonPep = True, useweap = True, usepywr = True)



### now do error metric fig for major flow locations
nodes = ['delMontague', 'delTrenton', 'outletAssunpink', 'outletSchuylkill']#, 'outletChristina', 'delLordville']
major_flow_metrics = get_error_metrics(major_flows, radial_models, nodes)
plot_radial_error_metrics(major_flow_metrics, radial_models, nodes, useNonPep = True, useweap = True, usepywr = True, usemajorflows=True)



### plot montague & trenton flows
# fig = plt.figure(figsize=(8,8))
# plt.plot(major_flows['obs']['delMontague'], color='k')
# plt.plot(major_flows['nhmv10']['delMontague'], color='b')
# plt.plot(major_flows['pywr_nhmv10']['delMontague'], color='r')
# plt.title('montague')
# plt.show()
#
# fig = plt.figure(figsize=(8,8))
# plt.plot(major_flows['obs']['delTrenton'], color='k')
# plt.plot(major_flows['nhmv10']['delTrenton'], color='b')
# plt.plot(major_flows['pywr_nhmv10']['delTrenton'], color='r')
# plt.title('trenton')
# plt.show()
#
# print('nhm montague', major_flows['nhmv10']['delMontague'].mean(), major_flows['nhmv10']['delMontague'].std(), major_flows['nhmv10']['delMontague'].max())
# print('pywr montague', major_flows['pywr_nhmv10']['delMontague'].mean(), major_flows['pywr_nhmv10']['delMontague'].std(), major_flows['pywr_nhmv10']['delMontague'].max())
# print('nhm trenton', major_flows['nhmv10']['delTrenton'].mean(), major_flows['nhmv10']['delTrenton'].std(), major_flows['nhmv10']['delTrenton'].max())
# print('pywr trenton', major_flows['pywr_nhmv10']['delTrenton'].mean(), major_flows['pywr_nhmv10']['delTrenton'].std(), major_flows['pywr_nhmv10']['delTrenton'].max())

