import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from mpi4py import MPI


from pre.prep_input_data_functions import add_upstream_catchment_inflows
from utils.hdf5 import extract_realization_from_hdf5, get_hdf5_realization_numbers, export_ensemble_to_hdf5
from utils.directories import input_dir, fig_dir
from utils.lists import reservoir_list, majorflow_list
from utils.timeseries import get_rollmean_timeseries
from utils.timeseries import subset_timeseries

def regress_future_timeseries(timeseries, node, lag, 
                              use_log, remove_zeros, use_const,
                              print_summary=False, plot_scatter=False):
    """
    Predicting future timeseries (catchment inflows or interbasin diversions) 
    at a particular lag (days) using linear regression
    
    Args:
        timeseries (pd.DataFrame): The timeseries data, must include the node of interest.
        node (str): The node of interest.
        lag (int): The lag in days for the prediction.
        use_log (bool): Whether to use log-transformed values for the predictions.
        remove_zeros (bool): Whether to remove zero values from the predictions.
        use_const (bool): Whether to include a constant term in the regression.
        print_summary (bool): Whether to print the regression summary. Default is False.
        plot_scatter (bool): Whether to plot the scatter plot of the regression. Default is False.
        
    Returns:
        const (float): The constant term of the regression.
        slope (float): The slope of the regression.
    """

    Y = timeseries[node].iloc[lag:].values.astype(float)

    if use_const:
        X = np.ones((len(Y), 2))
        X[:, 1] = timeseries[node].iloc[:-lag].values
        if remove_zeros:
            nonzeros = np.logical_and(Y > 0.01, X[:, 1] > 0.01)
            Y = Y[nonzeros]
            X = X[nonzeros, :]
    else:
        if lag > 0:
            X = timeseries[node].iloc[:-lag].values
        else:
            X = timeseries[node].values

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
        plt.xlabel('Timeseries on day t')
        plt.ylabel(f'Timeseries on day t+{lag}')

    if use_const:
        lm = sm.OLS(Y, X, hasconst=True)
        lr = lm.fit()
        const, slope = lr.params[0], lr.params[1]
    else:
        lm = sm.OLS(Y, X, hasconst=False)
        lr = lm.fit()
        const, slope = 0, lr.params[0]

    if print_summary:
        print(lr.summary())
        print()

    if plot_scatter:
        if use_const:
            plt.scatter(X[:, 1], const + slope * X[:, 1])
        else:
            plt.scatter(X, slope * X)

    return const, slope


def get_known_or_predicted_value(timeseries, catchment_wc, 
                                 regressions, node, 
                                 lag, idx, use_log, 
                                 mode):
    """
    Retrieves future prediction or present/past value of 
    timeseries (catchment inflow or diversion).
    
    5 modes available.
        - perfect foresight: actual values at future date
        - same_day: assume value at future date is equal to value at current date
        - regression_disagg: separate regression for each catchment and add together afterwards (Montague & Trenton only. use regular regression for diversions.)
        - regression_agg: aggregate all catchment inflows for Mont/Tren and do aggregated regression.
    
    Args:  
        timeseries (pd.DataFrame): The timeseries data, must include the node of interest.
        catchment_wc (pd.DataFrame): The catchment water consumption data.
        regressions (dict): A dictionary containing regression models.
        node (str): The node of interest.
        lag (int): The lag in days for the prediction.
        idx (int): The index of the timeseries.
        use_log (bool): Whether to use log-transformed values for the predictions.
        mode (str): The mode for the prediction (see list in docstring for options).
        
    Returns:
        value (float): The predicted or known value.
    """
    assert mode in ['perfect_foresight', 'same_day', 'regression_disagg', 'regression_agg']

    if mode == 'same_day':
        timeseries_lag_prediction = timeseries[node].iloc[idx]
        timeseries_lagm1_prediction = timeseries[node].iloc[max(idx - 1, 0)]
    elif mode == 'perfect_foresight':
        if idx < timeseries.shape[0] - lag:
            timeseries_lag_prediction = timeseries[node].iloc[idx + lag]
            timeseries_lagm1_prediction = timeseries[node].iloc[idx + lag - 1]
        else:
            timeseries_lag_prediction = timeseries[node].iloc[-1]
            timeseries_lagm1_prediction = timeseries[node].iloc[-1]
    elif mode in ['regression_disagg', 'regression_agg']:
        if lag <= 0:
            timeseries_lag_prediction = timeseries[node].iloc[max(idx+lag, 0)]
            timeseries_lagm1_prediction = timeseries[node].iloc[max(idx+lag-1, 0)]
        elif lag > 0:
            ### first predict timeseries at idx+lag
            const = regressions[(node, lag)]['const']
            slope = regressions[(node, lag)]['slope']
            timeseries_t = timeseries[node].iloc[idx]
            if use_log:
                eps = 0.001
                timeseries_t = timeseries_t + eps if timeseries_t < eps else timeseries_t 
                timeseries_lag_prediction = np.exp(const + slope * np.log(timeseries_t))
            else:
                timeseries_lag_prediction = const + slope * timeseries_t
            ### now predict timeseries at idx+lag-1
            if lag == 1:
                timeseries_lagm1_prediction = timeseries_t
            else:
                const = regressions[(node, lag-1)]['const']
                slope = regressions[(node, lag-1)]['slope']
                if use_log:
                    eps = 0.001
                    timeseries_t = eps if timeseries_t < eps else timeseries_t 
                    timeseries_lagm1_prediction = np.exp(const + slope * np.log(timeseries_t))
                else:
                    timeseries_lagm1_prediction = const + slope * timeseries_t

    if node in reservoir_list + majorflow_list:
        ### now predict net catchment inflows minus consumption (which is based on ratio of lag-1 inflows)
        pywr_node = f'reservoir_{node}' if node in reservoir_list else f'link_{node}'
        withdrawal_const = catchment_wc.loc[pywr_node, 'Total_WD_MGD']
        consumption_ratio = catchment_wc.loc[pywr_node, 'Total_CU_WD_Ratio']
        consumption_prediction = min(timeseries_lag_prediction,
                                     consumption_ratio * min(timeseries_lagm1_prediction, withdrawal_const))
        value = timeseries_lag_prediction - consumption_prediction
    else:
        ### for nyc/nj demands, just use the lag value
        value = timeseries_lag_prediction

    return value


### function for creating lagged prediction datasets for catchment inflows & NJ diversions
def predict_inflows_diversions(dataset_label, start_date, end_date,
                               use_log=True, remove_zeros=False, use_const=False,
                               realization=None, ensemble_inflows=False,
                               save_predictions=True, return_predictions=False, make_figs=False,
                               catchment_inflows=None):
    """
    Creates lagged prediction datasets for catchment inflows & NJ diversions.
    
    Args:
        dataset_label (str): The label for the dataset to use.
        start_date (str): The start date for the predictions.
        end_date (str): The end date for the predictions.
        use_log (bool): Whether to use log-transformed values for the predictions. Default is True.
        remove_zeros (bool): Whether to remove zero values from the predictions. Default is False.
        use_const (bool): Whether to include a constant term in the regression. Default is False.
        realization (str): The realization number for the ensemble inflows. Default is None.
        ensemble_inflows (bool): Whether ensemble inflows are being used. Default is False.
        save_predictions (bool): Whether to save the predictions. Default is True.
        return_predictions (bool): Whether to return the predictions. Default is False.
        make_figs (bool): Whether to make figures of the predictions. Default is False.
        catchment_inflows (pd.DataFrame): The catchment inflow data. Default is None.
        
    Returns:
        predicted_timeseries (pd.DataFrame): The predicted timeseries.
    """

    ### read in catchment inflows and withdrawals/consumptions
    if ensemble_inflows and catchment_inflows is None:
        ensemble_input_dir = input_dir
        ensemble_input_dir += 'synthetic_ensembles/' if 'syn' in dataset_label else 'historic_ensembles/'

        ensemble_filename = f'{ensemble_input_dir}/catchment_inflow_{dataset_label}.hdf5'
        catchment_inflows = extract_realization_from_hdf5(ensemble_filename, realization, stored_by_node=True)
        
        if 'datetime' in catchment_inflows.columns:
            catchment_inflows.drop('datetime', axis=1, inplace=True)
        catchment_inflows = catchment_inflows.astype(float)       
        catchment_inflows['datetime'] = catchment_inflows.index 
    else:
        catchment_inflows = pd.read_csv(f'{input_dir}/catchment_inflow_{dataset_label}.csv')
        catchment_inflows.index = pd.DatetimeIndex(catchment_inflows['datetime'])
    
    catchment_inflows_training = subset_timeseries(catchment_inflows, start_date, end_date)

    # Withdrawals are currently the same across ensemble realizations
    catchment_wc = pd.read_csv(f'{input_dir}/sw_avg_wateruse_Pywr-DRB_Catchments.csv')
    catchment_wc.index = catchment_wc['node']


    ### first loop through lags of 1-4 days and get each node's regression that is needed for that lag.
    regressions = {}
    lag = 1
    lag_1_nodes = ['01436000', 'wallenpaupack', 'prompton', 'shoholaMarsh',
                   'mongaupeCombined', '01433500', 'delMontague', 'beltzvilleCombined', '01447800', 'fewalter',
                   '01449800',
                   'merrillCreek', 'hopatcong', 'nockamixon', 'delDRCanal']
    for node in lag_1_nodes:
        const, slope = regress_future_timeseries(catchment_inflows_training, node, lag, use_log=use_log,
                                              remove_zeros=remove_zeros, use_const=use_const)
        regressions[(node, lag)] = {'const': const, 'slope': slope}

    lag = 2
    lag_2_nodes = ['mongaupeCombined', '01433500', 'delMontague', 'beltzvilleCombined','01447800','fewalter',
                   '01449800', 'merrillCreek', 'hopatcong', 'nockamixon', 'delDRCanal']
    for node in lag_2_nodes:
        for l in [lag, lag-1]:
            const, slope = regress_future_timeseries(catchment_inflows_training, node, l, use_log=use_log,
                                                  remove_zeros=remove_zeros, use_const=use_const)
            regressions[(node, l)] = {'const': const, 'slope': slope}

    lag = 3
    lag_3_nodes = ['merrillCreek', 'hopatcong', 'nockamixon', 'delDRCanal']
    for node in lag_3_nodes:
        for l in [lag, lag-1]:
            const, slope = regress_future_timeseries(catchment_inflows_training, node, l, use_log=use_log,
                                                  remove_zeros=remove_zeros, use_const=use_const)
            regressions[(node, l)] = {'const': const, 'slope': slope}

    lag = 4
    lag_4_nodes = ['delDRCanal']
    for node in lag_4_nodes:
        for l in [lag, lag-1]:
            const, slope = regress_future_timeseries(catchment_inflows_training, node, l, use_log=use_log,
                                                  remove_zeros=remove_zeros, use_const=use_const)
            regressions[(node, l)] = {'const': const, 'slope': slope}


    ### now get 2-/1-day lagged predictions for total non-NYC flow for Montague, & 1-4-day predictions for Trenton.
    ### do both prediction (with lagged linear regression) and perfect foresight (actual inflow)
    predicted_timeseries = pd.DataFrame({'datetime': catchment_inflows['datetime']})
    pred_node = 'delMontague'
    for pred_lag in [2,1]:
        node_lags = [('01425000', pred_lag-2), ('01417000', pred_lag-2), ('delLordville', pred_lag-2),
                     ('01436000', pred_lag-1), ('wallenpaupack', pred_lag-1), ('prompton', pred_lag-1),
                     ('shoholaMarsh', pred_lag-1),
                     ('mongaupeCombined', pred_lag), ('01433500', pred_lag), ('delMontague', pred_lag-1)]

        for mode in ['regression_disagg', 'perfect_foresight', 'same_day']:
            predicted_timeseries[f'{pred_node}_lag{pred_lag}_{mode}'] = np.zeros(catchment_inflows.shape[0])
            for node, lag in node_lags:
                predicted_timeseries[f'{pred_node}_lag{pred_lag}_{mode}'] += np.array(
                    [get_known_or_predicted_value(catchment_inflows, catchment_wc, regressions, node,
                                                  lag, idx, use_log, mode) for idx in
                     range(catchment_inflows.shape[0])])


    pred_node = 'delTrenton'
    for pred_lag in range(4, 0, -1):
        node_lags = [('01425000', pred_lag-4), ('01417000', pred_lag-4), ('delLordville', pred_lag-4),
                     ('01436000', pred_lag-3), ('wallenpaupack', pred_lag-3), ('prompton', pred_lag-3),
                     ('shoholaMarsh', pred_lag-3),
                     ('mongaupeCombined', pred_lag-2), ('01433500', pred_lag-2), ('delMontague', pred_lag-2),
                     ('beltzvilleCombined', pred_lag-2), ('01447800', pred_lag-2), ('fewalter', pred_lag-2),
                     ('01449800', pred_lag-2),
                     ('hopatcong', pred_lag-1), ('merrillCreek', pred_lag-1), ('nockamixon', pred_lag-1),
                     ('delDRCanal', pred_lag)]

        for mode in ['regression_disagg', 'perfect_foresight', 'same_day']:
            predicted_timeseries[f'{pred_node}_lag{pred_lag}_{mode}'] = np.zeros(catchment_inflows.shape[0])
            for node, lag in node_lags:
                predicted_timeseries[f'{pred_node}_lag{pred_lag}_{mode}'] += np.array(
                    [get_known_or_predicted_value(catchment_inflows, catchment_wc, regressions, node,
                                                  lag, idx, use_log, mode) for idx in
                     range(catchment_inflows.shape[0])])


    ### also do regression prediction on delMontague & delTrenton nonnyc gage flows in aggregate, rather than adding individual regressions
    mode = 'regression_agg'
    nonnyc_gage_flow = add_upstream_catchment_inflows(catchment_inflows, exclude_NYC=True)
    nonnyc_gage_flow_training = subset_timeseries(nonnyc_gage_flow, start_date, end_date)

    regressions = {}
    for pred_node, pred_lag in zip(('delMontague', 'delMontague', 'delMontague',
                                    'delTrenton', 'delTrenton', 'delTrenton', 'delTrenton', 'delTrenton'),
                                   (0, 1, 2,
                                    0, 1, 2, 3, 4)):
        const, slope = regress_future_timeseries(nonnyc_gage_flow_training, pred_node, pred_lag, use_log=use_log,
                                              remove_zeros=remove_zeros, use_const=use_const)
        regressions[(pred_node, pred_lag)] = {'const': const, 'slope': slope}

    for pred_node, pred_lag in zip(('delMontague', 'delMontague',
                                    'delTrenton', 'delTrenton', 'delTrenton', 'delTrenton'), (1, 2, 1, 2, 3, 4)):
        predicted_timeseries[f'{pred_node}_lag{pred_lag}_{mode}'] = np.array(
            [get_known_or_predicted_value(catchment_inflows, catchment_wc, regressions, pred_node,
                                          pred_lag, idx, use_log, mode) for idx in range(catchment_inflows.shape[0])])

    ### finally, just assume a 7-day moving average of flows to try to smooth releases
    mode = 'moving_average'
    window = 7
    rollmean_timeseries = get_rollmean_timeseries(nonnyc_gage_flow, window)
    for pred_node, pred_lag in zip(('delMontague', 'delMontague', 'delMontague',
                                    'delTrenton', 'delTrenton', 'delTrenton', 'delTrenton', 'delTrenton'),
                                   (0, 1, 2,
                                    0, 1, 2, 3, 4)):
        predicted_timeseries[f'{pred_node}_lag{pred_lag}_{mode}'] = rollmean_timeseries[pred_node]




    ### now do predictions for 1-4 days for NJ diversions as well
    ### read in NJ diversions (used as demands)
    nj_diversions = pd.read_csv(f'{input_dir}/deliveryNJ_DRCanal_extrapolated.csv')
    nj_diversions.index = pd.DatetimeIndex(nj_diversions['datetime'])
    nj_diversions['demand_nj'] = nj_diversions['D_R_Canal']
    nj_diversions = subset_timeseries(nj_diversions, catchment_inflows.index[0], catchment_inflows.index[-1])
    nj_diversions_training = subset_timeseries(nj_diversions, start_date, end_date)
    pred_node = 'demand_nj'
    regressions = {}

    for pred_lag in [0,1,2,3,4]:
        const, slope = regress_future_timeseries(nj_diversions_training, pred_node, pred_lag, use_log=use_log,
                                                 remove_zeros=remove_zeros, use_const=use_const)
        regressions[(pred_node, pred_lag)] = {'const': const, 'slope': slope}
    for pred_lag in [1,2,3,4]:
        for mode in ['regression_disagg', 'perfect_foresight', 'same_day']:
            predicted_timeseries[f'{pred_node}_lag{pred_lag}_{mode}'] = np.array(
                [get_known_or_predicted_value(nj_diversions, catchment_wc, regressions, pred_node,
                                              pred_lag, idx, use_log, mode) for idx in range(nj_diversions.shape[0])])
        predicted_timeseries[f'{pred_node}_lag{pred_lag}_regression_agg'] = predicted_timeseries[f'{pred_node}_' + \
                                                                                f'lag{pred_lag}_regression_disagg']

    mode = 'moving_average'
    window = 7
    rollmean_timeseries = get_rollmean_timeseries(nj_diversions, window)
    for pred_lag in [0, 1, 2, 3, 4]:
        predicted_timeseries[f'{pred_node}_lag{pred_lag}_{mode}'] = rollmean_timeseries[pred_node]

    ### save to csv
    if save_predictions:
        predicted_timeseries.to_csv(f'{input_dir}/predicted_inflows_diversions_{dataset_label}.csv', index=False)
    if return_predictions:
        return predicted_timeseries


    ### plot performance at different locations & modes
    if make_figs:
        units = 'MCM/D'
        assert units in ['MCM/D', 'MGD']
        units_conversion = 0.0037854118 if units == 'MCM/D' else 1.
        loc_dict = {'delMontague':'Montague', 'delTrenton':'Trenton', 'demand_nj':'NJ diversion'}
        for mode in ['regression_disagg']:#, 'regression_agg', 'same_day', 'moving_average', 'perfect_foresight']
            fig, axs = plt.subplots(4, 3, figsize=(8, 8), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
            for row, lag in enumerate(range(1, 5)):
                for col, loc in enumerate(['delMontague','delTrenton','demand_nj']):
                    ax = axs[row,col]
                    if use_log:
                        ax.loglog('log')
                    if f'{loc}_lag{lag}_{mode}' in predicted_timeseries.columns:
                        ax.scatter(predicted_timeseries[f'{loc}_lag{lag}_perfect_foresight'] * units_conversion,
                                   predicted_timeseries[f'{loc}_lag{lag}_{mode}'] * units_conversion,
                                   color='cornflowerblue', alpha=0.2, zorder=1)
                        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
                        ax.plot([lims[0], lims[1]], [lims[0], lims[1]], color='k', alpha=1, lw=0.5, ls=':', zorder=2)
                        ax.annotate(f'{loc_dict[loc]}, {lag} day', xy=(0.01, 0.98), xycoords='axes fraction', ha='left',
                                    va='top')
                    else:
                        ax.tick_params(
                            axis='x',
                            which='both',
                            bottom=False,
                            labelbottom=False)
                        ax.tick_params(
                            axis='y',
                            which='both',
                            left=False,
                            labelleft=False)
                        ax.set_frame_on(False)
                    if (col == 0 and row <= 1) or (col == 1 and row > 1):
                        ax.set_ylabel(f'Predicted ({units})')
                    if (row == 3 and col > 0) or (col == 0 and row == 1):
                        ax.set_xlabel(f'Observed ({units})')

            plt.savefig(f'{fig_dir}/predict_flows_{mode}_{dataset_label}.png', dpi=400, bbox_inches='tight')




def predict_ensemble_inflows_diversions(dataset_label, start_date, end_date,
                                        use_log=True, remove_zeros=False, 
                                        use_const=False,
                                        input_dir=input_dir):
    """
    Makes predictions for inflows and diversions at non-NYC gage flows, 
    using the specified ensemble dataset. 
    Predictions are parallelizing using MPI. 
    Final predictions are saved to an HDF5 file.
    
    Args:
        dataset_label (str): The label for the dataset to use. 
        start_date (str): The start date for the predictions.
        end_date (str): The end date for the predictions.
        use_log (bool): Whether to use log-transformed values for the predictions. Default is True.
        remove_zeros (bool): Whether to remove zero values from the predictions. Default is False.
        use_const (bool): Whether to include a constant term in the regression. Default is False.
        input_dir (str): The input directory for the ensemble data. Default is input_dir.
        
    Returns:
        None
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    ensemble_pred_nonnyc_gage_flows = {}

    if 'obs_pub' in dataset_label:
        input_dir += 'historic_ensembles/'
    elif 'syn' in dataset_label:
        input_dir += 'synthetic_ensembles/'
    else:
        print(f'Invalid dataset label {dataset_label}.')
        print('Current implementation only supports input of types: obs_pub or syn.')
        return

    ensemble_filename = f'{input_dir}catchment_inflow_{dataset_label}.hdf5'
    realization_numbers = get_hdf5_realization_numbers(ensemble_filename)

    # Divide work among MPI processes
    realizations_per_proc = np.array_split(realization_numbers, size)[rank]
    num_realizations_per_proc = len(realizations_per_proc)
    
    # Each process works on its assigned part
    for i in realizations_per_proc:
        print(f'Process {rank}: Making predictions for realization {i+1} of {num_realizations_per_proc}')
        df_predictions = predict_inflows_diversions(dataset_label, start_date, end_date,
                                                    use_log=use_log, remove_zeros=remove_zeros, 
                                                    use_const=use_const,
                                                    ensemble_inflows=True, realization=str(i),
                                                    save_predictions=False, return_predictions=True,
                                                    make_figs=False)
        ensemble_pred_nonnyc_gage_flows[f'{i}'] = df_predictions.copy()

    # Gather all predictions to root process
    all_ensemble_pred = comm.gather(ensemble_pred_nonnyc_gage_flows, root=0)

    # Export to HDF5 by root process
    if rank == 0:
        combined_ensemble_pred = {key: val for d in all_ensemble_pred for key, val in d.items()}
        output_filename = f'{input_dir}predicted_inflows_diversions_{dataset_label}.hdf5'
        print('Exporting ensemble of inflows/diversions to hdf5.')
        export_ensemble_to_hdf5(combined_ensemble_pred, output_filename)
    return
