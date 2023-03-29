"""
Trevor Amestoy

Exploring ways to interact with simulated streamflows. 

Stacked streamgraph here. 
"""

import pandas as pd
import numpy as np
import altair as alt
from drb_make_figs import get_pywr_results, get_base_results, reservoir_list, majorflow_list

def plot_interactive_streamflow_stack(node, model, 
                                    percentage_flow = True,  
                                    output_dir = 'output_data/', 
                                    fig_dir = 'figs/' , 
                                    plot_target = False):
    """
    Generates an HTML based interactive stacked timeseries plot of the contributing streamflows at a node of interest.
    """
    
    cms_to_mgd = 22.82
    cm_to_mg = 264.17/1e6
    cfs_to_mgd = 0.0283 * 22824465.32 / 1e6

    flow_data = get_pywr_results(output_dir, model, results_set='major_flow')
    release_data = get_pywr_results(output_dir, model, results_set='res_release')
    inflow_data = get_pywr_results(output_dir, model, results_set='inflow')
    base_flow = get_base_results('input_data/','obs', release_data.index, results_set='major_flow')

    unmanaged = ['delLordville', 'delMontague']


    site_matches_link = [['delLordville', ['01427207'], ['cannonsville', 'pepacton', '01425000', '01417000']],
                        # '01432055': '01432055',  ### lackawaxen river gage downstream of wallenpaupack & prompton
                        ['delMontague', ['01438500'], ['cannonsville', 'pepacton', '01425000', '01417000', 'delLordville',
                                                        'prompton', 'wallenpaupack', 'shoholaMarsh', 'mongaupeCombined', 'neversink', '01436000', '01433500']],
                        ['delTrenton', ['01463500'], ['cannonsville', 'pepacton', '01425000', '01417000', 'delLordville',
                                                        'prompton', 'wallenpaupack', 'shoholaMarsh', 'mongaupeCombined', 'neversink', '01436000', '01433500', 'delMontague',
                                                    'beltzvilleCombined', 'fewalter', 'merrillCreek', 'hopatcong', 'nockamixon', '01449800', '01447800']],
                        ['outletAssunpink', ['01463620'], ['assunpink', '01463620']], ## note, should get downstream junction, just using reservoir-adjacent gage for now
                        ['outletSchuylkill', ['01474500'], ['ontelaunee', 'stillCreek', 'blueMarsh', 'greenLane', '01470960']],
                        ['outletChristina', ['01480685'], ['marshCreek']]
                        ]


    ### Find contributions
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
    use_releases = [i for i in contributing if i in release_data.columns]
    use_inflows = [i for i in contributing if (i not in use_releases) and (i in unmanaged)]

    release_contributions = release_data[use_releases]
    inflow_contributions = inflow_data[use_inflows]
    node_inflow = inflow_data[node]
    total = base_flow[node]
    contributions = pd.concat([release_contributions, inflow_contributions, node_inflow], axis=1)

    print(f'Res Releases: {release_contributions.columns}\n')
    print(f'Inflow contributing: {inflow_contributions.columns}\n')
    # count lag

    lag_1 = ['wallenpaupack', 'prompton', 'shoholaMarsh', 'mongaupeCombined','neversink', 'delLordville']
    lag_2 = ['cannonsville', 'pepacton']
    direct = [i for i in contributions.columns if (i not in lag_1) and (i not in lag_2)]
    labels = ['Wallenpaupack', 'Prompton', 'Shohola Marsh', 'Mongaupe','Neversink', 'Lordville', 'Cannonsville', 'Pepacton','Beltzville',
    'FE Walter', 'Merrill Creek', 'Hopatcong','Nockamixon','Montague','Trenton']

    shifted = pd.concat([contributions[direct].iloc[2:,:], contributions[lag_1].iloc[1:-1,:], contributions[lag_2].iloc[0:-2,:]], ignore_index=True, axis=1)

    contributions = shifted.dropna(axis=0)
    contributions.columns = labels
    percent_contributions = contributions.divide(total, axis=0)

    if percentage_flow:
        ylabel = 'Fraction of Total Observed Flow'
        stream_data = percent_contributions.copy()
        filenameadd = 'percent'
        ymax = 1.25
    else:
        ylabel = 'Flow (MGD)'
        filenameadd = 'absolute'
        stream_data = contributions.copy()
        ymax = 0.8 * np.quantile(total, 0.7)


    s_data = stream_data.stack().reset_index()
    s_data.columns = ['Date', 'Source', 'flow']
    
    ## PLOTTING via altair

    # Setup interactive selection via legend
    selection = alt.selection_multi(fields=['Source'], bind='legend')

    # Generate plot
    viz = alt.Chart(s_data).mark_area().encode(
        alt.X('Date',
            axis=alt.Axis(format='%Y', domain=False, tickSize=1)
        ),
        alt.Y('sum(flow):Q', stack='zero', title = ylabel, scale = alt.Scale(domain=[0,ymax])),
        alt.Color('Source:N', scale=alt.Scale(scheme='tableau20')),
        opacity = alt.condition(selection, alt.value(1), alt.value(0.2))
        ).add_selection(selection).properties(width=600, height = 300).interactive()

    viz.save(f'{fig_dir}{model}_streamflow_stack_{filenameadd}.html')
    return_contributions = False
    if return_contributions:
        return release_contributions, inflow_contributions
    else:
        return