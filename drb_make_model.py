import json
import ast
import numpy as np
import pandas as pd
import sys

def drb_make_model(inflow_type, backup_inflow_type):
    ### first load baseline model
    input_dir = '../input_data/'
    model_sheets_dir = 'model_data/'
    model_base_file = model_sheets_dir + 'drb_model_base.json'
    model_full_file = model_sheets_dir + 'drb_model_full.json'
    model_sheets_start = model_sheets_dir + 'drb_model_'
    model = json.load(open(model_base_file, 'r'))

    ### parameters associated with STARFIT rule type
    starfit_remove_Rmax = True
    starfit_linear_below_NOR = True

    ### function for writing all relevant parameters to simulate starfit reservoir
    def create_starfit_params(d, r):
        ### d = param dictionary, r = reservoir name

        ### first get starfit const params for this reservoir
        for s in ['NORhi_alpha', 'NORhi_beta', 'NORhi_max', 'NORhi_min', 'NORhi_mu',
                  'NORlo_alpha', 'NORlo_beta', 'NORlo_max', 'NORlo_min', 'NORlo_mu',
                  'Release_alpha1', 'Release_alpha2', 'Release_beta1', 'Release_beta2',
                  'Release_c', 'Release_max', 'Release_min', 'Release_p1', 'Release_p2',
                  'GRanD_CAP_MG', 'GRanD_MEANFLOW_MGD']:
            name = 'starfit_' + s + '_' + r
            d[name] = {}
            d[name]['type'] = 'constant'
            d[name]['url'] = 'drb_model_istarf_conus.csv'
            d[name]['column'] = s
            d[name]['index_col'] = 'reservoir'
            d[name]['index'] = r

        ### aggregated params - each needs agg function and list of params to agg
        agg_param_list = [('NORhi_sin', 'product', ['sin_weekly', 'NORhi_alpha']),
                          ('NORhi_cos', 'product', ['cos_weekly', 'NORhi_beta']),
                          ('NORhi_sum', 'sum', ['NORhi_mu', 'NORhi_sin', 'NORhi_cos']),
                          ('NORhi_minbound', 'max', ['NORhi_sum', 'NORhi_min']),
                          ('NORhi_maxbound', 'min', ['NORhi_minbound', 'NORhi_max']),
                          ('NORhi_final', 'product', ['NORhi_maxbound', 0.01]),
                          ('NORlo_sin', 'product', ['sin_weekly', 'NORlo_alpha']),
                          ('NORlo_cos', 'product', ['cos_weekly', 'NORlo_beta']),
                          ('NORlo_sum', 'sum', ['NORlo_mu', 'NORlo_sin', 'NORlo_cos']),
                          ('NORlo_minbound', 'max', ['NORlo_sum', 'NORlo_min']),
                          ('NORlo_maxbound', 'min', ['NORlo_minbound', 'NORlo_max']),
                          ('NORlo_final', 'product', ['NORlo_maxbound', 0.01]),
                          ('NORlo_final_unnorm', 'product', ['NORlo_final', 'GRanD_CAP_MG']),
                          ('neg_NORhi_final_unnorm', 'product', ['neg_NORhi_final', 'GRanD_CAP_MG']),
                          ('aboveNOR_sum', 'sum', ['volume', 'neg_NORhi_final_unnorm', 'flow_weekly']),
                          ('aboveNOR_final', 'product', ['aboveNOR_sum', 1 / 7]),
                          ('inNOR_sin', 'product', ['sin_weekly', 'Release_alpha1']),
                          ('inNOR_cos', 'product', ['cos_weekly', 'Release_beta1']),
                          ('inNOR_sin2x', 'product', ['sin2x_weekly', 'Release_alpha2']),
                          ('inNOR_cos2x', 'product', ['cos2x_weekly', 'Release_beta2']),
                          ('inNOR_p1a_num', 'sum', ['inNOR_fracvol', 'neg_NORlo_final']),
                          ('inNOR_p1a_denom', 'sum', ['NORhi_final', 'neg_NORlo_final']),
                          ('inNOR_p1a_final', 'product', ['inNOR_p1a_div', 'Release_p1']),
                          ('inNOR_inorm_pt1', 'sum', ['flow', 'neg_GRanD_MEANFLOW_MGD']),
                          ('inNOR_p2i', 'product', ['inNOR_inorm_final', 'Release_p2']),
                          ('inNOR_norm', 'sum',
                           ['inNOR_sin', 'inNOR_cos', 'inNOR_sin2x', 'inNOR_cos2x', 'Release_c', 'inNOR_p1a_final',
                            'inNOR_p2i', 1]),
                          ('inNOR_final', 'product', ['inNOR_norm', 'GRanD_MEANFLOW_MGD']),
                          ('Release_min_norm', 'sum', ['Release_min', 1]),
                          ('Release_min_final', 'product', ['Release_min_norm', 'GRanD_MEANFLOW_MGD'])]

        ### adjust params depending on whether we want to follow starfit strictly (starfit_linear_below_NOR = False),
        ###     or use a smoother linearly declining release policy below NOR
        if starfit_linear_below_NOR == False:
            agg_param_list += [('belowNOR_final', 'product', ['Release_min_final'])]
        else:
            agg_param_list += [('belowNOR_pt1', 'product', ['inNOR_final', 'belowNOR_frac_NORlo']),
                               ('belowNOR_final', 'max', ['belowNOR_pt1', 'Release_min_final'])]

        ### adjust params depending on whether we want to follow starfit directly (starfit_remove_Rmax = False),
        ###     or remove the max release param to allow for more realistic high flows
        if starfit_remove_Rmax == False:
            agg_param_list += [('Release_max_norm', 'sum', ['Release_max', 1])]
        else:
            agg_param_list += [('Release_max_norm', 'sum', ['Release_max', 999999])]

        ### now rest of aggregated params
        agg_param_list += [('Release_max_final', 'product', ['Release_max_norm', 'GRanD_MEANFLOW_MGD']),
                           ('target_pt2', 'max', ['target_pt1', 'Release_min_final']),
                           ('target_final', 'min', ['target_pt2', 'Release_max_final']),
                           ('release_pt1', 'sum', ['flow', 'volume']),
                           ('release_pt2', 'min', ['release_pt1', 'target_final']),
                           ('release_pt3', 'sum', ['release_pt1', 'neg_GRanD_CAP_MG']),
                           ('release_final', 'max', ['release_pt2', 'release_pt3'])]

        ### loop over agg params, add to pywr dictionary/json
        for s, f, lp in agg_param_list:
            name = 'starfit_' + s + '_' + r
            d[name] = {}
            d[name]['type'] = 'aggregated'
            d[name]['agg_func'] = f
            d[name]['parameters'] = []
            for p in lp:
                if type(p) is int:
                    param = p
                elif type(p) is float:
                    param = p
                elif type(p) is str:
                    if p.split('_')[0] in ('sin', 'cos', 'sin2x', 'cos2x'):
                        param = p
                    elif p.split('_')[0] in ('volume', 'flow'):
                        param = p + '_' + r
                    else:
                        param = 'starfit_' + p + '_' + r
                else:
                    print('unsupported type in parameter list, ', p)
                d[name]['parameters'].append(param)

        ### negative params
        for s in ['NORhi_final', 'NORlo_final', 'GRanD_MEANFLOW_MGD', 'GRanD_CAP_MG']:
            name = 'starfit_neg_' + s + '_' + r
            d[name] = {}
            d[name]['type'] = 'negative'
            d[name]['parameter'] = 'starfit_' + s + '_' + r

        ### division params
        for s, num, denom in [('inNOR_fracvol', 'volume', 'GRanD_CAP_MG'),
                              ('inNOR_p1a_div', 'inNOR_p1a_num', 'inNOR_p1a_denom'),
                              ('inNOR_inorm_final', 'inNOR_inorm_pt1', 'GRanD_MEANFLOW_MGD'),
                              ('belowNOR_frac_NORlo', 'volume', 'NORlo_final_unnorm')]:
            name = 'starfit_' + s + '_' + r
            d[name] = {}
            d[name]['type'] = 'division'
            if num.split('_')[0] in ('sin', 'cos', 'sin2x', 'cos2x'):
                d[name]['numerator'] = num
            elif num.split('_')[0] in ('volume', 'flow'):
                d[name]['numerator'] = num + '_' + r
            else:
                d[name]['numerator'] = 'starfit_' + num + '_' + r
            if denom.split('_')[0] in ('sin', 'cos', 'sin2x', 'cos2x'):
                d[name]['denominator'] = denom
            elif denom.split('_')[0] in ('volume', 'flow'):
                d[name]['denominator'] = denom + '_' + r
            else:
                d[name]['denominator'] = 'starfit_' + denom + '_' + r

        ### other params
        other = {'starfit_level_' + r: {'type': 'controlcurveindex',
                                        'storage_node': 'reservoir_' + r,
                                        'control_curves': ['starfit_NORhi_final_' + r,
                                                           'starfit_NORlo_final_' + r]},
                 'flow_weekly_' + r: {'type': 'aggregated', 'agg_func': 'product', 'parameters': ['flow_' + r, 7]},
                 'volume_' + r: {'type': 'interpolatedvolume',
                                 'values': [0, 1000000],
                                 'node': 'reservoir_' + r,
                                 'volumes': [0, 1000000]},
                 'starfit_target_pt1_' + r: {'type': 'indexedarray',
                                             'index_parameter': 'starfit_level_' + r,
                                             'params': ['starfit_aboveNOR_final_' + r,
                                                        'starfit_inNOR_final_' + r,
                                                        'starfit_belowNOR_final_' + r]}}
        for name, params in other.items():
            d[name] = {}
            for k, v in params.items():
                d[name][k] = v

        return d



    ### create standard model node structures
    def add_major_node(model, name, node_type, inflow_type, backup_inflow_type=None, outflow_type=None, downstream_node=None,
                       initial_volume=None, initial_volume_perc=None, variable_cost=None):
        '''
        Add a major node to the model. Major nodes types include reservoir & river.
        This function will add the major node and all standard minor nodes that belong to each major node
        ( i.e., catchment, withdrawal, consumption, outflow), along with their standard parameters and edges.
        All nodes, edges, and parameters are added to the model dict, which is then returned
        :param model: the dict holding all model elements, which will be written to JSON file at completion.
        :param name: name of major node
        :param node_type: type of major node - either 'reservoir' or 'river'
        :param inflow_type: 'nhmv10', etc
        :param backup_inflow_type: 'nhmv10', etc. only active if inflow_type is a WEAP series - backup used to fill inflows for non-WEAP reservoirs.
        :param outflow_type: define what type of outflow node to use (if any) - either 'starfit' or 'regulatory'
        :param downstream_node: name of node directly downstream, for writing edge network.
        :param initial_volume: (reservoirs only) starting volume of reservoir in MG. Must correspond to "initial_volume_perc" times
                               total volume, as pywr doesnt calculate this automatically in time step 0
        :param initial_volume_perc: (reservoirs only) fraction full for reservoir initially
                               (note this is fraction, not percent, a confusing pywr convention)
        :param variable_cost: (reservoirs only) If False, cost is fixed throughout simulation.
                               If True, it varies according to state-dependent parameter.
        :return: model
        '''

        ### NYC reservoirs are a bit more complex, leave some of model creation in csv files for now
        is_NYC_reservoir = name in ['cannonsville', 'pepacton', 'neversink']
        ### does it have explicit outflow node for starfit or regulatory behavior?
        has_outflow_node = outflow_type in ['starfit', 'regulatory']
        ### list of river nodes
        river_nodes = ['delLordville','delMontague','delTrenton','outletAssunpink','outletSchuylkill','outletChristina']

        ### first add major node to dict
        if node_type == 'reservoir':
            node_name = f'reservoir_{name}'
            reservoir = {
                'name': node_name,
                'type': 'storage',
                'max_volume': f'max_volume_{name}',
                'initial_volume': initial_volume,
                'initial_volume_pc': initial_volume_perc,
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
                'max_flow': f'starfit_release_final_{name}'
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


        ### NYC reservoirs have additional 2 link nodes between reservoir and outflow, to account for storage balancing rules
        if is_NYC_reservoir:
            outflow_link1 = {
                'name': f'link_{name}_outflow_1',
                'type': 'link',
                'max_flow': f'volbalance_max_flow_montagueTrenton_{name}'
            }
            model['nodes'].append(outflow_link1)

            outflow_link2 = {
                'name': f'link_{name}_outflow_2',
                'type': 'link',
                'cost': 100.0
            }
            model['nodes'].append(outflow_link2)



        ### now add edges of model flow network
        ### catchment to reservoir
        model['edges'].append([f'catchment_{name}', node_name])
        ### catchment to withdrawal
        model['edges'].append([f'catchment_{name}', f'catchmentWithdrawal_{name}'])
        ### withdrawal to consumption
        model['edges'].append([f'catchmentWithdrawal_{name}', f'catchmentConsumption_{name}'])
        ### withdrawal to reservoir
        model['edges'].append([f'catchmentWithdrawal_{name}', node_name])
        ### reservoir downstream node (via outflow node if one exists)
        if downstream_node in river_nodes:
            downstream_name = f'link_{downstream_node}'
        elif downstream_node == 'output_del':
            downstream_name = downstream_node
        else:
            downstream_name = f'reservoir_{downstream_node}'
        if has_outflow_node:
            if not is_NYC_reservoir:
                model['edges'].append([node_name, f'outflow_{name}'])
            else:
                ### NYC reservoirs have additional two links between reservoir and outflow to acct for storage balancing rules
                model['edges'].append([node_name, f'link_{name}_outflow_1'])
                model['edges'].append([node_name, f'link_{name}_outflow_2'])
                model['edges'].append([f'link_{name}_outflow_1', f'outflow_{name}'])
                model['edges'].append([f'link_{name}_outflow_2', f'outflow_{name}'])
            model['edges'].append([f'outflow_{name}', downstream_name])
        else:
            model['edges'].append([node_name, downstream_name])



        ### now add standard parameters
        ### inflows to catchment
        if 'WEAP' not in inflow_type:
            inflow_source = f'{input_dir}catchment_inflow_{inflow_type}.csv'
        else:
            if name in ['cannonsville', 'pepacton', 'neversink', 'wallenpaupack', 'promption',
                        'mongaupeCombined', 'beltzvilleCombined', 'blueMarsh', 'ontelaunee', 'nockamixon', 'assunpink']:
                inflow_source = f'{input_dir}catchment_inflow_{inflow_type}.csv'
            else:
                inflow_source = f'{input_dir}catchment_inflow_{backup_inflow_type}.csv'

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

        ### max volume of reservoir, from GRanD database
        if node_type == 'reservoir':
            model['parameters'][f'max_volume_{name}'] = {
                'type': 'constant',
                'url': 'drb_model_istarf_conus.csv',
                'column': 'GRanD_CAP_MG',
                'index_col': 'reservoir',
                'index': name
            }

        ### for starfit reservoirs, need to add a bunch of starfit specific params
        if outflow_type == 'starfit':
            model['parameters'] = create_starfit_params(model['parameters'], name)

        ### get max flow for catchment withdrawal nodes based on DRBC data
        model['parameters'][f'max_flow_catchmentWithdrawal_{name}'] = {
            'type': 'constant',
            'url': '../input_data/sw_avg_wateruse_Pywr-DRB_Catchments.csv',
            'column': 'Total_WD_MGD',
            'index_col': 'node',
            'index': node_name
        }

        ### get max flow for catchment consumption nodes based on DRBC data
        model['parameters'][f'max_flow_catchmentConsumption_{name}'] = {
            'type': 'constant',
            'url': '../input_data/sw_avg_wateruse_Pywr-DRB_Catchments.csv',
            'column': 'Total_CU_MGD',
            'index_col': 'node',
            'index': node_name
        }



        return model



    ### add major nodes to model, along with corresponding minor nodes, edges, & parameters
    model['nodes'] = []
    model['edges'] = []
    model['parameters'] = {}
    inflow_type = 'nhmv10'
    model = add_major_node(model, 'cannonsville', 'reservoir', inflow_type, backup_inflow_type, 'regulatory', 'delLordville', 117313.5018, 0.8, True)
    model = add_major_node(model, 'pepacton', 'reservoir', inflow_type, backup_inflow_type, 'regulatory', 'delLordville', 158947.009, 0.8, True)
    model = add_major_node(model, 'delLordville', 'river', inflow_type, backup_inflow_type, None, 'delMontague')
    model = add_major_node(model, 'neversink', 'reservoir', inflow_type, backup_inflow_type, 'regulatory', 'delMontague', 37026.34752, 0.8, True)
    model = add_major_node(model, 'wallenpaupack', 'reservoir', inflow_type, backup_inflow_type, 'starfit', 'delMontague', 70375.4208, 0.8, False)
    model = add_major_node(model, 'prompton', 'reservoir', inflow_type, backup_inflow_type, 'starfit', 'delMontague', 3022.12768, 0.8, False)
    model = add_major_node(model, 'shoholaMarsh', 'reservoir', inflow_type, backup_inflow_type, 'starfit', 'delMontague', 6889.60576, 0.8, False)
    model = add_major_node(model, 'mongaupeCombined', 'reservoir', inflow_type, backup_inflow_type, 'starfit', 'delMontague', 22697.65824, 0.8, False)
    model = add_major_node(model, 'delMontague', 'river', inflow_type, backup_inflow_type, 'regulatory', 'delTrenton')
    model = add_major_node(model, 'beltzvilleCombined', 'reservoir', inflow_type, backup_inflow_type, 'starfit', 'delTrenton', 38653.64704, 0.8, False)
    model = add_major_node(model, 'fewalter', 'reservoir', inflow_type, backup_inflow_type, 'starfit', 'delTrenton', 3022.12768, 0.8, False)
    model = add_major_node(model, 'merrillCreek', 'reservoir', inflow_type, backup_inflow_type, 'starfit', 'delTrenton', 11982.84192, 0.8, False)
    model = add_major_node(model, 'hopatcong', 'reservoir', inflow_type, backup_inflow_type, 'starfit', 'delTrenton', 12574.5872, 0.8, False)
    model = add_major_node(model, 'nockamixon', 'reservoir', inflow_type, backup_inflow_type, 'starfit', 'delTrenton', 18513.17376, 0.8, False)
    model = add_major_node(model, 'delTrenton', 'river', inflow_type, backup_inflow_type, 'regulatory', 'output_del')
    model = add_major_node(model, 'assunpink', 'reservoir', inflow_type, backup_inflow_type, 'starfit', 'outletAssunpink', 3296.86656, 0.8, False)
    model = add_major_node(model, 'outletAssunpink', 'river', inflow_type, backup_inflow_type, None, 'output_del')
    model = add_major_node(model, 'ontelaunee', 'reservoir', inflow_type, backup_inflow_type, 'starfit', 'outletSchuylkill', 3022.12768, 0.8, False)
    model = add_major_node(model, 'stillCreek', 'reservoir', inflow_type, backup_inflow_type, 'starfit', 'outletSchuylkill', 3022.12768, 0.8, False)
    model = add_major_node(model, 'blueMarsh', 'reservoir', inflow_type, backup_inflow_type, 'starfit', 'outletSchuylkill', 33856.28352, 0.8, False)
    model = add_major_node(model, 'greenLane', 'reservoir', inflow_type, backup_inflow_type, 'starfit', 'outletSchuylkill', 6551.4656, 0.8, False)
    model = add_major_node(model, 'outletSchuylkill', 'river', inflow_type, backup_inflow_type, None, 'output_del')
    model = add_major_node(model, 'marshCreek', 'reservoir', inflow_type, backup_inflow_type, 'starfit', 'outletChristina', 3022.12768, 0.8, False)
    model = add_major_node(model, 'outletChristina', 'river', inflow_type, backup_inflow_type, None, 'output_del')


    ### load remaining nodes from spreadsheet & add elements as dict items
    sheet = 'nodes'
    df = pd.read_csv(model_sheets_start + sheet + '.csv')
    for i in range(df.shape[0]):
        nodedict = {}
        for j, col in enumerate(df.columns):
            ### columns to ignore
            if col not in ('long', 'lat', 'notes', 'huc12', 'gages', 'nwm_feature_id'):
                val = df.iloc[i,j]
                if isinstance(val, int):
                    nodedict[col] = val
                elif isinstance(val, str):
                    ### if it's a list, need to convert from str to actual list
                    if '[' in val:
                        val = ast.literal_eval(val)
                    ### if it's "true" or "false", convert to bool
                    if val in ["\"true\"", "\"false\""]:
                        val = {"\"true\"": True, "\"false\"": False}[val]
                    ### if it's actually a number as string, convert to numeric
                    else:
                        try:
                            val = float(val)
                        except:
                            pass
                    nodedict[col] = val
                elif isinstance(val, float) or isinstance(val, np.float64):
                    if not np.isnan(val):
                        nodedict[col] = val
                else:
                    print(f'type not supported: {type(val)}, instance: {val}')
        model[sheet].append(nodedict)


    ### load remaining edges from spreadsheet & add elements as dict items
    sheet = 'edges'
    df = pd.read_csv(model_sheets_start + sheet + '.csv')
    for i in range(df.shape[0]):
        edge = []
        for j, col in enumerate(df.columns):
            if col not in ('type', 'notes'):
                val = df.iloc[i,j]
                if isinstance(val, int):
                    edge.append(val)
                elif isinstance(val, str):
                    ### if it's a list, need to convert from str to actual list
                    if '[' in val:
                        val = ast.literal_eval(val)
                    ### if it's "true" or "false", convert to bool
                    if val in ["\"true\"", "\"false\""]:
                        val = {"\"true\"": True, "\"false\"": False}[val]
                    edge.append(val)
                elif isinstance(val, float) or isinstance(val, np.float64):
                    if not np.isnan(val):
                        edge.append(val)
                else:
                    print(f'type not supported: {type(val)}, instance: {val}')
        model[sheet].append(edge)





    ### load remaining parameters from spreadsheet & add elements as dict items
    sheet = 'parameters'
    df = pd.read_csv(model_sheets_start + sheet + '.csv')
    for i in range(df.shape[0]):
        name = df.iloc[i,0]
        ### skip empty line
        if type(name) is not str:
            pass
        ### print elements from csv directly into json
        else:
            model[sheet][name] = {}
            for j, col in enumerate(df.columns[1:], start=1):
                val = df.iloc[i,j]

                ### fill in based on data type
                if isinstance(val, int):
                    model[sheet][name][col] = val
                elif isinstance(val, str):
                    ### if it's a list, need to convert from str to actual list
                    if '[' in val:
                        val = ast.literal_eval(val)
                    ### if it's "true" or "false", convert to bool
                    if val in ["\"true\"", "\"false\""]:
                        val = {"\"true\"": True, "\"false\"": False}[val]
                    ### if it's actually a number as string, convert to numeric
                    else:
                        try:
                            val = float(val)
                        except:
                            pass
                    model[sheet][name][col] = val
                elif isinstance(val, float) or isinstance(val, np.float64):
                    if not np.isnan(val):
                        model[sheet][name][col] = val
                else:
                    print(f'type not supported: {type(val)}, instance: {val}')


    ### save full model as json
    with open(model_full_file, 'w') as o:
        json.dump(model, o, indent=4)

