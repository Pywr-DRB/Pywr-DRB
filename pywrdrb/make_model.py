"""
Contains functions used to construct a pywrdrb model in JSON format.
"""

import json
import pandas as pd

from utils.directories import input_dir, model_data_dir
from utils.lists import majorflow_list, reservoir_list, reservoir_list_nyc
from pywr_drb_node_data import upstream_nodes_dict, immediate_downstream_nodes_dict, downstream_node_lags

EPS = 1e-8
nhm_inflow_scaling = True

##########################################################################################
### add_major_node()
##########################################################################################

def add_major_node(model, name, node_type, inflow_type, outflow_type=None, downstream_node=None, downstream_lag=0,
                   capacity=None, initial_volume_frac=None, variable_cost=None, has_catchment=True, inflow_ensemble_indices = None):
    """
    Add a major node to the model.

    Major node types include reservoirs and rivers. This function adds the major node
    and all standard minor nodes that belong to each major node (i.e., catchment,
    withdrawal, consumption, outflow), along with their standard parameters and edges.
    All nodes, edges, and parameters are added to the model dictionary, which is then returned.

    Args:
        model (dict): The dictionary holding all model elements, which will be written to a JSON file upon completion.
        name (str): The name of the major node.
        node_type (str): The type of major node, either 'reservoir' or 'river'.
        inflow_type (str): The inflow type, e.g., 'nhmv10'.
        outflow_type (str): Define what type of outflow node to use (if any), either 'starfit' or 'regulatory'.
        downstream_node (str): The name of the node directly downstream for writing the edge network.
        downstream_lag (int): The travel time (in days) between flow leaving a node and reaching its downstream node.
        capacity (float): The capacity of the reservoir in million gallons (MG) (reservoirs only).
        initial_volume_frac (float): The initial fraction full for the reservoir (reservoirs only).
                                     Note: This is a fraction, not a percentage, despite being named "initial_volume_pc" in the Pywr JSON by convention.
        variable_cost (bool): If False, the cost is fixed throughout the simulation.
                              If True, it varies according to a state-dependent parameter (reservoirs only).
        has_catchment (bool): True if the node has a catchment with inflows and withdrawal/consumption.
                              False for artificial nodes that coincide with another (e.g., delTrenton, which shares a catchment with delDRCanal).
        inflow_ensemble_indices (None or list): List of ensemble indices for inflows (optional).

    Returns:
        dict: The updated model dictionary.
    """

    ### NYC reservoirs are a bit more complex, leave some of model creation in csv files for now
    is_NYC_reservoir = name in ['cannonsville', 'pepacton', 'neversink']
    ### does it have explicit outflow node for starfit or regulatory behavior?
    has_outflow_node = outflow_type in ['starfit', 'regulatory']

    ### first add major node to dict
    if node_type == 'reservoir':
        node_name = f'reservoir_{name}'
        initial_volume = capacity * initial_volume_frac
        reservoir = {
            'name': node_name,
            'type': 'storage',
            'max_volume': f'max_volume_{name}',
            'initial_volume': initial_volume,
            'initial_volume_pc': initial_volume_frac,
            'cost': -10.0 if not variable_cost else f'storage_cost_{name}'
        }
        model['nodes'].append(reservoir)
    elif node_type == 'river':
        node_name = f'link_{name}'
        river = {
            'name': node_name,
            'type': 'link'
        }
        model['nodes'].append(river)

    if has_catchment:
        ### add catchment node that sends inflows to reservoir
        catchment = {
            'name': f'catchment_{name}',
            'type': 'catchment',
            'flow': f'flow_{name}'
        }
        model['nodes'].append(catchment)

        ### add withdrawal node that withdraws flow from catchment for human use
        withdrawal = {
            'name': f'catchmentWithdrawal_{name}',
            'type': 'link',
            'cost': -15.0,
            'max_flow': f'max_flow_catchmentWithdrawal_{name}'
        }
        model['nodes'].append(withdrawal)

        ### add consumption node that removes flow from model - the rest of withdrawals return back to reservoir
        consumption = {
            'name': f'catchmentConsumption_{name}',
            'type': 'output',
            'cost': -200.0,
            'max_flow': f'max_flow_catchmentConsumption_{name}'
        }
        model['nodes'].append(consumption)

    ### add outflow node (if any), either using STARFIT rules or regulatory targets.
    ### Note reservoirs must have either starfit or regulatory, river nodes have either regulatory or None
    if outflow_type == 'starfit':
        outflow = {
            'name': f'outflow_{name}',
            'type': 'link',
            'cost': -500.0,
            'max_flow': f'starfit_release_{name}'
        }
        model['nodes'].append(outflow)
        
    elif outflow_type == 'regulatory':
        outflow = {
            'name': f'outflow_{name}',
            'type': 'rivergauge',
            'mrf': f'mrf_target_{name}',
            'mrf_cost': -1000.0
        }
        model['nodes'].append(outflow)

    ### add Delay node to account for flow travel time between nodes. Lag unit is days.
    if downstream_lag>0:
        delay = {
            'name': f'delay_{name}',
            'type': 'DelayNode',
            'days': downstream_lag
        }
        model['nodes'].append(delay)


    ### now add edges of model flow network
    if has_catchment:
        ### catchment to reservoir
        model['edges'].append([f'catchment_{name}', node_name])
        ### catchment to withdrawal
        model['edges'].append([f'catchment_{name}', f'catchmentWithdrawal_{name}'])
        ### withdrawal to consumption
        model['edges'].append([f'catchmentWithdrawal_{name}', f'catchmentConsumption_{name}'])
        ### withdrawal to reservoir
        model['edges'].append([f'catchmentWithdrawal_{name}', node_name])
    ### reservoir downstream node (via outflow node if one exists)
    if downstream_node in majorflow_list:
        downstream_name = f'link_{downstream_node}'
    elif downstream_node == 'output_del':
        downstream_name = downstream_node
    else:
        downstream_name = f'reservoir_{downstream_node}'
    if has_outflow_node:
        model['edges'].append([node_name, f'outflow_{name}'])
        if downstream_lag > 0:
            model['edges'].append([f'outflow_{name}', f'delay_{name}'])
            model['edges'].append([f'delay_{name}', downstream_name])
        else:
            model['edges'].append([f'outflow_{name}', downstream_name])

    else:
        if downstream_lag > 0:
            model['edges'].append([node_name, f'delay_{name}'])
            model['edges'].append([f'delay_{name}', downstream_name])
        else:
            model['edges'].append([node_name, downstream_name])

    ################################################################
    ### now add standard parameters
    ################################################################
    ### max volume of reservoir, from GRanD database except where adjusted from other sources (eg NYC)
    if node_type == 'reservoir':
        model['parameters'][f'max_volume_{name}'] = {
            'type': 'constant',
            'url': 'drb_model_istarf_conus.csv',
            'column': 'Adjusted_CAP_MG',
            'index_col': 'reservoir',
            'index': name
        }

    ### for starfit reservoirs, need to add a bunch of starfit specific params
    if outflow_type == 'starfit':
        model['parameters'][f'starfit_release_{name}'] = {
            'type': 'STARFITReservoirRelease',
            'node': name
        }



    if has_catchment:
        ### Assign inflows to nodes
        if inflow_ensemble_indices:
            ### Use custom FlowEnsemble parameter to handle scenario indexing
            model['parameters'][f'flow_{name}'] = {
                'type': 'FlowEnsemble',
                'node': name,
                'inflow_type': inflow_type,
                'inflow_ensemble_indices': inflow_ensemble_indices
                }
            
        else:
            inflow_source = f'{input_dir}catchment_inflow_{inflow_type}.csv'

            ### Use single-scenario historic data
            model['parameters'][f'flow_base_{name}'] = {
                'type': 'dataframe',
                'url': inflow_source,
                'column': name,
                'index_col': 'datetime',
                'parse_dates': True
            }

            model['parameters'][f'flow_{name}'] = {
                'type': 'aggregated',
                'agg_func': 'product',
                'parameters': [
                    f'flow_base_{name}',
                    'flow_factor'
                ]
            }

        ### get max flow for catchment withdrawal nodes based on DRBC data
        model['parameters'][f'max_flow_catchmentWithdrawal_{name}'] = {
            'type': 'constant',
            'url': f'{input_dir}sw_avg_wateruse_Pywr-DRB_Catchments.csv',
            'column': 'Total_WD_MGD',
            'index_col': 'node',
            'index': node_name
        }

        ### get max flow for catchment consumption nodes based on DRBC data
        ### assume the consumption_t = R * withdrawal_{t-1}, where R is the ratio of avg consumption to withdrawal from DRBC data
        model['parameters'][f'catchmentConsumptionRatio_{name}'] = {
            'type': 'constant',
            'url': f'{input_dir}sw_avg_wateruse_Pywr-DRB_Catchments.csv',
            'column': 'Total_CU_WD_Ratio',
            'index_col': 'node',
            'index': node_name
        }
        model['parameters'][f'prev_flow_catchmentWithdrawal_{name}'] = {
            'type': 'flow',
            'node': f'catchmentWithdrawal_{name}'
        }
        model['parameters'][f'max_flow_catchmentConsumption_{name}'] = {
            'type': 'aggregated',
            'agg_func': 'product',
            'parameters': [
                f'catchmentConsumptionRatio_{name}',
                f'prev_flow_catchmentWithdrawal_{name}'
            ]
        }

    return model


##########################################################################################
### drb_make_model()
##########################################################################################

def make_model(inflow_type, model_filename, start_date, end_date, use_hist_NycNjDeliveries=True,
                   inflow_ensemble_indices = None):
    """
    Creates the JSON file used by Pywr to define the model, including all nodes, edges, and parameters.

    Args:
        inflow_type (str): Type of inflow.
        start_date (str): Start date of the model.
        end_date (str): End date of the model.
        use_hist_NycNjDeliveries (bool): Flag indicating whether to use historical NYC and NJ deliveries.
        inflow_ensemble_indices (list): List of inflow ensemble indices.
    Returns:
        dict: Model JSON representation.
    """

    #######################################################################
    ### Basic model info
    #######################################################################

    if inflow_ensemble_indices:
        N_SCENARIOS = len(inflow_ensemble_indices)
    else:
        N_SCENARIOS = 1
        
    ### create dict to hold all model nodes, edges, params, etc, following Pywr protocol. This will be saved to JSON at end.
    model = {
        'metadata': {
            'title': 'DRB',
            'description': 'Pywr DRB representation',
            'minimum_version': '0.4'
        },
        'timestepper': {
            'start': start_date,
            'end': end_date,
            'timestep': 1
        },
        'scenarios': [
            {
                'name': 'inflow',
                'size': N_SCENARIOS
            }
        ]
    }

    model['nodes'] = []
    model['edges'] = []
    model['parameters'] = {}


    #######################################################################
    ### add major nodes (e.g., reservoirs) to model, along with corresponding minor nodes (e.g., withdrawals), edges, & parameters
    #######################################################################

    ### get initial reservoir storages as 80% of capacity
    istarf = pd.read_csv(f'{model_data_dir}drb_model_istarf_conus.csv')
    initial_volume_frac = 0.8
    def get_reservoir_capacity(reservoir):
        return float(istarf['Adjusted_CAP_MG'].loc[istarf['reservoir'] == reservoir].iloc[0])

    ### get downstream node to link to for the current node
    for node, downstream_node in immediate_downstream_nodes_dict.items():
    
        node_type = 'reservoir' if node in reservoir_list else 'river'

        ### get outflow regulatory constraint type for reservoirs
        if node in reservoir_list_nyc:
            outflow_type = 'regulatory'
        elif node in reservoir_list:
            outflow_type = 'starfit'
        else:
            outflow_type = None

        ### get flow lag (days) between current node and its downstream connection
        downstream_lag = downstream_node_lags[node]
            
        variable_cost = True if (outflow_type == 'regulatory') else False
        capacity = get_reservoir_capacity(node) if (node_type == 'reservoir') else None
        has_catchment = True if (node != 'delTrenton') else False

        ### set up major node
        model = add_major_node(model, node, node_type, inflow_type, outflow_type, downstream_node,  downstream_lag,
                               capacity, initial_volume_frac, variable_cost, has_catchment, inflow_ensemble_indices)


    #######################################################################
    ### Add additional nodes beyond those associated with major nodes above
    #######################################################################

    ### Node for NYC aggregated storage across 3 reservoirs
    nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']
    model['nodes'].append({
            'name': 'reservoir_agg_nyc',
            'type': 'aggregatedstorage',
            'storage_nodes': [f'reservoir_{r}' for r in nyc_reservoirs]
    })

    ### Nodes linking each NYC reservoir to NYC deliveries
    for r in nyc_reservoirs:
        model['nodes'].append({
            'name': f'link_{r}_nyc',
            'type': 'link',
            'cost': -500.0,
            'max_flow': f'volbalance_max_flow_delivery_nyc_{r}'
        })

    ### Nodes for NYC & NJ deliveries
    for d in ['nyc', 'nj']:
        model['nodes'].append({
            'name': f'delivery_{d}',
            'type': 'output',
            'cost': -500.0,
            'max_flow': f'max_flow_delivery_{d}'
        })

    ### Node for final model sink in Delaware Bay
    model['nodes'].append({
            'name': 'output_del',
            'type': 'output'
        })


    #######################################################################
    ### Add additional edges beyond those associated with major nodes above
    #######################################################################

    ### Edges linking each NYC reservoir to NYC deliveries
    for r in nyc_reservoirs:
        model['edges'].append([f'reservoir_{r}', f'link_{r}_nyc'])
        model['edges'].append([f'link_{r}_nyc', 'delivery_nyc'])

    ### Edge linking Delaware River at DRCanal to NJ deliveries
    model['edges'].append(['link_delDRCanal', 'delivery_nj'])



    #######################################################################
    ### Add additional parameters beyond those associated with major nodes above
    #######################################################################

    ### Define "scenarios" based on flow multiplier -> only run one with 1.0 for now
    if N_SCENARIOS == 1:
        model['parameters']['flow_factor'] = {
                'type': 'constantscenario',
                'scenario': 'inflow',
                'values': [1.0]
            }

    ### demand for NYC
    if use_hist_NycNjDeliveries:
        ### if this flag is True, we assume demand is equal to historical deliveries timeseries
        model['parameters'][f'demand_nyc'] = {
            'type': 'dataframe',
            'url': f'{input_dir}deliveryNYC_ODRM_extrapolated.csv',
            'column': 'aggregate',
            'index_col': 'datetime',
            'parse_dates': True
        }
    else:
        ### otherwise, assume demand is equal to max allotment under FFMP
        model['parameters'][f'demand_nyc'] = {
            'type': 'constant',
            'url': 'drb_model_constants.csv',
            'column': 'value',
            'index_col': 'parameter',
            'index': 'max_flow_baseline_delivery_nyc'
        }

    ### repeat for NJ deliveries
    if use_hist_NycNjDeliveries:
        ### if this flag is True, we assume demand is equal to historical deliveries timeseries
        model['parameters'][f'demand_nj'] = {
            'type': 'dataframe',
            'url': f'{input_dir}deliveryNJ_DRCanal_extrapolated.csv',
            'index_col': 'datetime',
            'parse_dates': True
        }
    else:
        ### otherwise, assume demand is equal to max allotment under FFMP
        model['parameters'][f'demand_nj'] = {
            'type': 'constant',
            'url': 'drb_model_constants.csv',
            'column': 'value',
            'index_col': 'parameter',
            'index': 'max_flow_baseline_monthlyAvg_delivery_nj'
        }

    ### max allowable delivery to NYC (on moving avg)
    model['parameters']['max_flow_baseline_delivery_nyc'] = {
        'type': 'constant',
        'url': 'drb_model_constants.csv',
        'column': 'value',
        'index_col': 'parameter',
        'index': 'max_flow_baseline_delivery_nyc'
    }

    ### NJ has both a daily limit and monthly average limit
    model['parameters']['max_flow_baseline_daily_delivery_nj'] = {
        'type': 'constant',
        'url': 'drb_model_constants.csv',
        'column': 'value',
        'index_col': 'parameter',
        'index': 'max_flow_baseline_daily_delivery_nj'
    }
    model['parameters']['max_flow_baseline_monthlyAvg_delivery_nj'] = {
        'type': 'constant',
        'url': 'drb_model_constants.csv',
        'column': 'value',
        'index_col': 'parameter',
        'index': 'max_flow_baseline_monthlyAvg_delivery_nj'
    }

    ### levels defining operational regimes for NYC reservoirs base on combined storage: 1a, 1b, 1c, 2, 3, 4, 5.
    ### Note 1a assumed to fill remaining space, doesnt need to be defined here.
    levels = ['1a', '1b', '1c', '2', '3', '4', '5']
    for level in levels[1:]:
        model['parameters'][f'level{level}'] = {
            'type': 'dailyprofile',
            'url': 'drb_model_dailyProfiles.csv',
            'index_col': 'profile',
            'index': f'level{level}'
        }

    ### Control curve index that tells us which level the aggregated NYC storage is currently in
    model['parameters']['drought_level_agg_nyc'] = {
        'type': 'controlcurveindex',
        'storage_node': 'reservoir_agg_nyc',
        'control_curves': [f'level{level}' for level in levels[1:]]
    }

    ### factors defining delivery profiles for NYC and NJ, for each storage level: 1a, 1b, 1c, 2, 3, 4, 5.
    demands = ['nyc', 'nj']
    for demand in demands:
        for level in levels:
            model['parameters'][f'level{level}_factor_delivery_{demand}'] = {
                'type': 'constant',
                'url': 'drb_model_constants.csv',
                'column': 'value',
                'index_col': 'parameter',
                'index': f'level{level}_factor_delivery_{demand}'
            }

    ### Indexed arrays that dictate cutbacks to NYC & NJ deliveries, based on current storage level and DOY
    for demand in demands:
        model['parameters'][f'drought_factor_delivery_{demand}'] = {
                'type': 'indexedarray',
                'index_parameter': 'drought_level_agg_nyc',
                'params': [f'level{level}_factor_delivery_{demand}' for level in levels]
            }

    ### Max allowable daily delivery to NYC & NJ- this will be very large for levels 1&2 (so that moving average limit
    ### in the next parameter is active), but apply daily flow limits for more drought stages.
    model['parameters']['max_flow_drought_delivery_nyc'] = {
            'type': 'aggregated',
            'agg_func': 'product',
            'parameters': [
                'max_flow_baseline_delivery_nyc',
                'drought_factor_delivery_nyc'
            ]
        }

    ### Max allowable delivery to NYC in current time step to maintain moving avg limit. Based on custom Pywr parameter.
    model['parameters']['max_flow_ffmp_delivery_nyc'] = {
            'type': 'FfmpNycRunningAvg',
            'node': 'delivery_nyc',
            'max_avg_delivery': 'max_flow_baseline_delivery_nyc'
        }

    ### Max allowable delivery to NJ in current time step to maintain moving avg limit. Based on custom Pywr parameter.
    model['parameters']['max_flow_ffmp_delivery_nj'] = {
            'type': 'FfmpNjRunningAvg',
            'node': 'delivery_nj',
            'max_avg_delivery': 'max_flow_baseline_monthlyAvg_delivery_nj',
            'max_daily_delivery': 'max_flow_baseline_daily_delivery_nj',
            'drought_factor': 'drought_factor_delivery_nj'
        }

    ### Now actual max flow to NYC delivery node is the min of demand, daily FFMP limit, and daily limit to meet FFMP moving avg limit
    model['parameters']['max_flow_delivery_nyc'] = {
            'type': 'aggregated',
            'agg_func': 'min',
            'parameters': [
                'demand_nyc',
                'max_flow_drought_delivery_nyc',
                'max_flow_ffmp_delivery_nyc'
            ]
        }

    ### Now actual max flow to NJ delivery node is the min of demand and daily limit to meet FFMP moving avg limit
    model['parameters']['max_flow_delivery_nj'] = {
            'type': 'aggregated',
            'agg_func': 'min',
            'parameters': [
                'demand_nj',
                'max_flow_ffmp_delivery_nj'
            ]
        }

    ### baseline release flow rate for each NYC reservoir, dictated by FFMP
    for reservoir in nyc_reservoirs:
        model['parameters'][f'mrf_baseline_{reservoir}'] = {
            'type': 'constant',
            'url': 'drb_model_constants.csv',
            'column': 'value',
            'index_col': 'parameter',
            'index': f'mrf_baseline_{reservoir}'
        }

    ### Control curve index that tells us which level each individual NYC reservoir's storage is currently in
    for reservoir in nyc_reservoirs:
        model['parameters'][f'drought_level_{reservoir}'] = {
            'type': 'controlcurveindex',
            'storage_node': f'reservoir_{reservoir}',
            'control_curves': [f'level{level}' for level in levels[1:]]
        }

    ### Factor governing changing release reqs from NYC reservoirs, based on aggregated storage across 3 reservoirs
    for reservoir in nyc_reservoirs:
        model['parameters'][f'mrf_drought_factor_agg_{reservoir}'] = {
            'type': 'indexedarray',
            'index_parameter': 'drought_level_agg_nyc',
            'params': [f'level{level}_factor_mrf_{reservoir}' for level in levels]
        }

    ### Levels defining operational regimes for individual NYC reservoirs, as opposed to aggregated level across 3 reservoirs
    for reservoir in nyc_reservoirs:
        for level in levels:
            model['parameters'][f'level{level}_factor_mrf_{reservoir}'] = {
                'type': 'dailyprofile',
                'url': 'drb_model_dailyProfiles.csv',
                'index_col': 'profile',
                'index': f'level{level}_factor_mrf_{reservoir}'
            }

    ### Factor governing changing release reqs from NYC reservoirs, based on individual storage for particular reservoir
    for reservoir in nyc_reservoirs:
        model['parameters'][f'mrf_drought_factor_individual_{reservoir}'] = {
            'type': 'indexedarray',
            'index_parameter': f'drought_level_{reservoir}',
            'params': [f'level{level}_factor_mrf_{reservoir}' for level in levels]
        }

    ### Factor governing changing release reqs from NYC reservoirs, depending on whether aggregated or individual storage level is activated
    ### Based on custom Pywr parameter.
    for reservoir in nyc_reservoirs:
        model['parameters'][f'mrf_drought_factor_combined_final_{reservoir}'] = {
            'type': 'NYCCombinedReleaseFactor',
            'node': f'reservoir_{reservoir}'
        }

    ### FFMP mandated releases from NYC reservoirs
    for reservoir in nyc_reservoirs:
        model['parameters'][f'mrf_target_individual_{reservoir}'] = {
            'type': 'aggregated',
            'agg_func': 'product',
            'parameters': [
                f'mrf_baseline_{reservoir}',
                f'mrf_drought_factor_combined_final_{reservoir}'
            ]
        }

    ### variable storage cost for each reservoir, based on its fractional storage
    ### Note: may not need this anymore now that we have volume balancing rules. but maybe makes sense to leave in for extra protection.
    volumes = {'cannonsville': get_reservoir_capacity('cannonsville'),
               'pepacton':  get_reservoir_capacity('pepacton'),
               'neversink': get_reservoir_capacity('neversink')}
    for reservoir in nyc_reservoirs:
        model['parameters'][f'storage_cost_{reservoir}'] = {
            'type': 'interpolatedvolume',
            'values': [-100,-1],
            'node': f'reservoir_{reservoir}',
            'volumes': [-EPS, volumes[reservoir] + EPS]
        }

    ### current volume stored in each reservoir, plus the aggregated storage node
    for reservoir in nyc_reservoirs + ['agg_nyc']:
        model['parameters'][f'volume_{reservoir}'] = {
            'type': 'interpolatedvolume',
            'values': [-EPS, 1000000],
            'node': f'reservoir_{reservoir}',
            'volumes': [-EPS, 1000000]
        }

    ### aggregated inflows to NYC reservoirs
    model['parameters']['flow_agg_nyc'] = {
            'type': 'aggregated',
            'agg_func': 'sum',
            'parameters': [f'flow_{reservoir}' for reservoir in nyc_reservoirs]
        }

    ### aggregated max volume across to NYC reservoirs
    model['parameters']['max_volume_agg_nyc'] = {
            'type': 'aggregated',
            'agg_func': 'sum',
            'parameters': [f'max_volume_{reservoir}' for reservoir in nyc_reservoirs]
        }

    ### Target release from each NYC reservoir to satisfy NYC demand - part 1.
    ### Uses custom Pywr parameter.
    for reservoir in nyc_reservoirs:
        model['parameters'][f'volbalance_target_max_flow_delivery_nyc_{reservoir}'] = {
            'type': 'VolBalanceNYCDemandTarget',
            'node': f'reservoir_{reservoir}'
        }

    ### Sum of target releases from step above
    model['parameters'][f'volbalance_target_max_flow_delivery_agg_nyc'] = {
            'type': 'aggregated',
            'agg_func': 'sum',
            'parameters': [f'volbalance_target_max_flow_delivery_nyc_{reservoir}' for reservoir in nyc_reservoirs]
        }

    ### Target release from each NYC reservoir to satisfy NYC demand - part 2,
    ### rescaling to make sure total contribution across 3 reservoirs is equal to total demand.
    ### Uses custom Pywr parameter.
    for reservoir in nyc_reservoirs:
        model['parameters'][f'volbalance_max_flow_delivery_nyc_{reservoir}'] = {
            'type': 'VolBalanceNYCDemandFinal',
            'node': f'reservoir_{reservoir}'
        }



    ### Baseline flow target at Montague & Trenton
    mrfs = ['delMontague', 'delTrenton']
    for mrf in mrfs:
        model['parameters'][f'mrf_baseline_{mrf}'] = {
                'type': 'constant',
                'url': 'drb_model_constants.csv',
                'column': 'value',
                'index_col': 'parameter',
                'index': f'mrf_baseline_{mrf}'
            }

    ### Seasonal multiplier factors for Montague & Trenton flow targets based on drought level of NYC aggregated storage
    for mrf in mrfs:
        for level in levels:
            model['parameters'][f'level{level}_factor_mrf_{mrf}'] = {
                'type': 'monthlyprofile',
                'url': 'drb_model_monthlyProfiles.csv',
                'index_col': 'profile',
                'index': f'level{level}_factor_mrf_{mrf}'
            }

    ### Current value of seasonal multiplier factor for Montague & Trenton flow targets based on drought level of NYC aggregated storage
    for mrf in mrfs:
        model['parameters'][f'mrf_drought_factor_{mrf}'] = {
            'type': 'indexedarray',
            'index_parameter': 'drought_level_agg_nyc',
            'params': [f'level{level}_factor_mrf_{mrf}' for level in levels]
        }

    ### Total Montague & Trenton flow targets based on drought level of NYC aggregated storage
    for mrf in mrfs:
        model['parameters'][f'mrf_target_{mrf}'] = {
            'type': 'aggregated',
            'agg_func': 'product',
            'parameters': [
                f'mrf_baseline_{mrf}',
                f'mrf_drought_factor_{mrf}'
            ]
        }

    ### total non-NYC inflows to Montague & Trenton
    inflow_nodes = upstream_nodes_dict['delMontague'] + ['delMontague']
    inflow_nodes = [n for n in inflow_nodes if n not in reservoir_list_nyc]
    model['parameters'][f'volbalance_flow_agg_nonnyc_delMontague'] = {
            'type': 'aggregated',
            'agg_func': 'sum',
            'parameters': [f'flow_{node}' for node in inflow_nodes]
        }

    inflow_nodes = upstream_nodes_dict['delTrenton'] ### note: dont include delTrenton in this because it doesnt have catchment -> catchment inflows go to delDRCanal
    inflow_nodes = [n for n in inflow_nodes if n not in reservoir_list_nyc]
    model['parameters'][f'volbalance_flow_agg_nonnyc_delTrenton'] = {
            'type': 'aggregated',
            'agg_func': 'sum',
            'parameters': [f'flow_{node}' for node in inflow_nodes]
        }


    ### Get total release needed from NYC reservoirs to satisfy Montague & Trenton flow targets, after accting for non-NYC inflows and NJ diversions
    ### Uses custom Pywr parameter.
    model['parameters']['volbalance_relative_mrf_montagueTrenton'] = {
        'type': 'VolBalanceNYCDownstreamMRFTargetAgg'
    }

    ### Target release from each NYC reservoir to satisfy Montague & Trenton flow targets - part 1.
    ### Uses custom Pywr parameter.
    for reservoir in nyc_reservoirs:
        model['parameters'][f'volbalance_target_max_flow_montagueTrenton_{reservoir}'] = {
            'type': 'VolBalanceNYCDownstreamMRFTarget',
            'node': f'reservoir_{reservoir}'
        }

    ### Sum of target releases from step above
    model['parameters'][f'volbalance_target_max_flow_montagueTrenton_agg_nyc'] = {
        'type': 'aggregated',
        'agg_func': 'sum',
        'parameters': [f'volbalance_target_max_flow_montagueTrenton_{reservoir}' for reservoir in nyc_reservoirs]
    }

    ### Target release from each NYC reservoir to satisfy Montague & Trenton flow targets - part 2,
    ### rescaling to make sure total contribution across 3 reservoirs is equal to total flow requirement.
    ### Uses custom Pywr parameter.
    for reservoir in nyc_reservoirs:
        model['parameters'][f'volbalance_max_flow_montagueTrenton_{reservoir}'] = {
            'type': 'VolBalanceNYCDownstreamMRFFinal',
            'node': f'reservoir_{reservoir}'
        }

    ### finally, get final effective mandated release from each NYC reservoir, which is the max of its
    ###    individually-mandated release from FFMP and its individual contribution to the Montague/Trenton targets
    for reservoir in nyc_reservoirs:
        model['parameters'][f'mrf_target_{reservoir}'] = {
            'type': 'aggregated',
            'agg_func': 'max',
            'parameters': [f'mrf_target_individual_{reservoir}', f'volbalance_max_flow_montagueTrenton_{reservoir}']
        }

    #######################################################################
    ### save full model as json
    #######################################################################

    with open(f'{model_filename}', 'w') as o:
        json.dump(model, o, indent=4)