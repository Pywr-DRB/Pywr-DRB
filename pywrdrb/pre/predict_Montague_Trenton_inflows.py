import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import statsmodels.api as sm

sys.path.append('../')
sys.path.append('../pywrdrb/')
from utils.directories import input_dir
from plotting.plotting_functions import subset_timeseries
from pre.prep_input_data_functions import add_upstream_catchment_inflows

def predict_future_inflows(inflows, node, lag, use_log, remove_zeros, use_const, print_summary=False, plot_scatter=True):
    Y = inflows[node].iloc[lag:].values
    if use_const:
        X = np.ones((len(Y), 2))
        X[:, 1] = inflows[node].iloc[:-lag].values
        if remove_zeros:
            nonzeros = np.logical_and(Y > 0.01, X[:, 1] > 0.01)
            Y = Y[nonzeros]
            X = X[nonzeros, :]
    #             print(Y.min(), X[:,1].min())
    else:
        X = inflows[node].iloc[:-lag].values
        if remove_zeros:
            nonzeros = np.logical_and(Y > 0.01, X > 0.01)
            Y = Y[nonzeros]
            X = X[nonzeros]
    if use_log:
        Y += 0.001
        X += 0.001
        Y = np.log(Y)
        X = np.log(X)

    if plot_scatter:
        plt.figure()
        if use_const:
            plt.scatter(X[:, 1], Y)
        else:
            plt.scatter(X, Y)

        plt.title(node)
        plt.xlabel('Catchment inflow on day t')
        plt.ylabel(f'Catchment inflow on day t+{lag}')

    if use_const:
        lm = sm.OLS(Y, X, hasconst=True)
        lr = lm.fit()
        const, slope = lr.params[0], lr.params[1]
    else:
        lm = sm.OLS(Y, X, hasconst=False)
        lr = lm.fit()
        const, slope = 0, lr.params[0]

    # print(f'{node}, lag {lag}, R^2: {round(lr.rsquared, 4)}')
    if print_summary:
        print(lr.summary())
        print()

    if plot_scatter:
        if use_const:
            plt.scatter(X[:, 1], const + slope * X[:, 1])
        else:
            plt.scatter(X, slope * X)

    return const, slope


def get_flow_or_prediction(inflows, regressions, node, lag, idx, use_log, mode):
    ### mode: predict, perfect_foresight, same_day
    if mode == 'same_day':
        return inflows[node].iloc[idx]
    elif mode == 'perfect_foresight':
        if idx < inflows.shape[0] - lag:
            return inflows[node].iloc[idx + lag]
        else:
            return inflows[node].iloc[-1]
    elif mode in ['regression_disagg', 'regression_agg']:
        if (lag == 0) or (lag == -1 and idx == 0):
            return inflows[node].iloc[idx]
        elif lag == -1:
            return inflows[node].iloc[idx-1]
        elif lag > 0:
            const = regressions[(node, lag)]['const']
            slope = regressions[(node, lag)]['slope']
            value = inflows[node].iloc[idx]
            if use_log:
                prediction = np.exp(const + slope * np.log(value))
            else:
                prediction = const + slope * value
            return prediction


def predict_Montague_Trenton_inflows(dataset_label, start_date, end_date,
                                     use_log=False, remove_zeros=False, use_const=False):

    ### read in catchment inflows
    catchment_inflows = pd.read_csv(f'{input_dir}/catchment_inflow_{dataset_label}.csv')
    catchment_inflows.index = pd.DatetimeIndex(catchment_inflows['datetime'])
    catchment_inflows_training = subset_timeseries(catchment_inflows, start_date, end_date)

    ### first loop through lags of 1-4 days and get each node's regression that is needed for that lag.
    regressions = {}
    lag = 1
    lag_1_nodes = ['01436000', 'wallenpaupack', 'prompton', 'shoholaMarsh', 'mongaupeCombined', '01433500',
                   'delMontague', 'beltzvilleCombined', '01447800', 'fewalter', '01449800']
    for node in lag_1_nodes:
        const, slope = predict_future_inflows(catchment_inflows_training, node, lag, use_log=use_log,
                                              remove_zeros=remove_zeros, use_const=use_const)
        regressions[(node, lag)] = {'const': const, 'slope': slope}

    lag = 2
    lag_2_nodes = ['mongaupeCombined', '01433500', 'delMontague', 'beltzvilleCombined','01447800','fewalter','01449800', 'merrillCreek','hopatcong']
    for node in lag_2_nodes:
        const, slope = predict_future_inflows(catchment_inflows_training, node, lag, use_log=use_log,
                                              remove_zeros=remove_zeros, use_const=use_const)
        regressions[(node, lag)] = {'const': const, 'slope': slope}

    lag = 3
    lag_3_nodes = ['merrillCreek', 'hopatcong', 'nockamixon', 'delDRCanal']
    for node in lag_3_nodes:
        const, slope = predict_future_inflows(catchment_inflows_training, node, lag, use_log=use_log,
                                              remove_zeros=remove_zeros, use_const=use_const)
        regressions[(node, lag)] = {'const': const, 'slope': slope}

    lag = 4
    lag_4_nodes = ['nockamixon', 'delDRCanal']
    for node in lag_4_nodes:
        const, slope = predict_future_inflows(catchment_inflows_training, node, lag, use_log=use_log,
                                              remove_zeros=remove_zeros, use_const=use_const)
        regressions[(node, lag)] = {'const': const, 'slope': slope}


    ### now get 2-/1-day lagged predictions for total non-NYC flow for Montague, & 4-/3-day predictions for Trenton.
    ### do both prediction (with lagged linear regression) and perfect foresight (actual inflow)
    pred_nonnyc_gage_flow = pd.DataFrame({'datetime': catchment_inflows['datetime']})
    pred_node = 'delMontague'
    pred_lag = 2
    node_lags = [('01425000', 0), ('01417000', 0), ('delLordville', 0),
                 ('01436000', 1), ('wallenpaupack', 1), ('prompton', 1), ('shoholaMarsh', 1),
                 ('mongaupeCombined', 2), ('01433500', 2), ('delMontague', 2)]

    for mode in ['regression_disagg','perfect_foresight','same_day']:
        pred_nonnyc_gage_flow[f'{pred_node}_lag{pred_lag}_{mode}'] = np.zeros(catchment_inflows.shape[0])
        for node, lag in node_lags:
            pred_nonnyc_gage_flow[f'{pred_node}_lag{pred_lag}_{mode}'] += np.array(
                [get_flow_or_prediction(catchment_inflows, regressions, node, lag, idx, use_log, mode) for idx in
                 range(catchment_inflows.shape[0])])

    pred_node = 'delMontague'
    pred_lag = 1
    node_lags = [('01425000', -1), ('01417000', -1), ('delLordville', -1),
                 ('01436000', 0), ('wallenpaupack', 0), ('prompton', 0), ('shoholaMarsh', 0),
                 ('mongaupeCombined', 1), ('01433500', 1), ('delMontague', 1)]

    for mode in ['regression_disagg','perfect_foresight','same_day']:
        pred_nonnyc_gage_flow[f'{pred_node}_lag{pred_lag}_{mode}'] = np.zeros(catchment_inflows.shape[0])
        for node, lag in node_lags:
            pred_nonnyc_gage_flow[f'{pred_node}_lag{pred_lag}_{mode}'] += np.array(
                [get_flow_or_prediction(catchment_inflows, regressions, node, lag, idx, use_log, mode) for idx in
                 range(catchment_inflows.shape[0])])

    pred_node = 'delTrenton'
    pred_lag = 4
    node_lags = [('01425000', 0), ('01417000', 0), ('delLordville', 0),
                 ('01436000', 1), ('wallenpaupack', 1), ('prompton', 1), ('shoholaMarsh', 1),
                 ('mongaupeCombined', 2), ('01433500', 2), ('delMontague', 2),
                 ('beltzvilleCombined', 2), ('01447800', 2), ('fewalter', 2), ('01449800', 2),
                 ('hopatcong', 3), ('merrillCreek', 3),
                 ('nockamixon', 4), ('delDRCanal', 4)]

    for mode in ['regression_disagg','perfect_foresight','same_day']:
        pred_nonnyc_gage_flow[f'{pred_node}_lag{pred_lag}_{mode}'] = np.zeros(catchment_inflows.shape[0])
        for node, lag in node_lags:
            pred_nonnyc_gage_flow[f'{pred_node}_lag{pred_lag}_{mode}'] += np.array(
                [get_flow_or_prediction(catchment_inflows, regressions, node, lag, idx, use_log, mode) for idx in
                 range(catchment_inflows.shape[0])])

    pred_node = 'delTrenton'
    pred_lag = 3
    node_lags = [('01425000', -1), ('01417000', -1), ('delLordville', -1),
                 ('01436000', 0), ('wallenpaupack', 0), ('prompton', 0), ('shoholaMarsh', 0),
                 ('mongaupeCombined', 1), ('01433500', 1), ('delMontague', 1),
                 ('beltzvilleCombined', 1), ('01447800', 1), ('fewalter', 1), ('01449800', 1),
                 ('hopatcong', 2), ('merrillCreek', 2),
                 ('nockamixon', 3), ('delDRCanal', 3)]

    for mode in ['regression_disagg','perfect_foresight','same_day']:
        pred_nonnyc_gage_flow[f'{pred_node}_lag{pred_lag}_{mode}'] = np.zeros(catchment_inflows.shape[0])
        for node, lag in node_lags:
            pred_nonnyc_gage_flow[f'{pred_node}_lag{pred_lag}_{mode}'] += np.array(
                [get_flow_or_prediction(catchment_inflows, regressions, node, lag, idx, use_log, mode) for idx in
                 range(catchment_inflows.shape[0])])




    ### also do regression prediction on delMontague & delTrenton nonnyc gage flows in aggregate, rather than adding individual regressions
    mode = 'regression_agg'
    nonnyc_gage_flow = add_upstream_catchment_inflows(catchment_inflows)
    nonnyc_gage_flow_training = subset_timeseries(nonnyc_gage_flow, start_date, end_date)

    regressions = {}
    for pred_node, pred_lag in zip(('delMontague', 'delMontague', 'delTrenton', 'delTrenton'), (1, 2, 3, 4)):
        const, slope = predict_future_inflows(nonnyc_gage_flow_training, pred_node, pred_lag, use_log=use_log,
                                              remove_zeros=remove_zeros, use_const=use_const)
        regressions[(pred_node, pred_lag)] = {'const': const, 'slope': slope}

        pred_nonnyc_gage_flow[f'{pred_node}_lag{pred_lag}_{mode}'] = np.array(
            [get_flow_or_prediction(nonnyc_gage_flow, regressions, pred_node, pred_lag, idx, use_log, mode) for idx in
             range(catchment_inflows.shape[0])])

    ### save to csv
    pred_nonnyc_gage_flow.to_csv(f'{input_dir}/predicted_nonnyc_gage_flow_{dataset_label}.csv', index=False)
