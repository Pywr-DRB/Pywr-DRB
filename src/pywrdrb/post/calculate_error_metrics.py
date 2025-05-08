"""
Calculate error metrics for simulated and observed streamflow data.

Overview
--------
This module provides functions for evaluating the performance of reservoir and major flow simulations
in the Pywr-DRB model framework. It computes a suite of error metrics comparing modeled data to
observed data at daily, monthly, yearly, and full timescales using metrics from HydroEval,
flow duration curve statistics, autocorrelation diagnostics, and roughness indicators.

Key Steps
---------
1. Subset modeled and observed time series to the desired time window.
2. Resample data to monthly or yearly resolution if needed.
3. Calculate performance metrics including NSE, KGE, autocorrelation, and FDC-based diagnostics.
4. Return a summary table of error metrics for all nodes, models, scenarios, and timescales.

Technical Notes
---------------
- Metrics include daily, monthly, yearly, and full-period variants.
- Uses HydroEval (https://github.com/hydro-informatics/hydroeval) for core metrics.
- Metrics include log-transformed variants for low flow sensitivity.
- Designed for evaluating Pywr-DRB reservoir and major river node outputs.
- Timeseries input assumed to be pandas Series indexed by datetime.

Links
-----
- https://github.com/pywrdrb/Pywr-DRB

Change Log
----------
Marilyn Smith, 2025-05-07, Added documentation and implemented full docstring formatting.
"""

import hydroeval as he
from scipy import stats
import pandas as pd
from pywrdrb.utils.lists import reservoir_list
import numpy as np

def subset_timeseries(data, start_date, end_date, end_inclusive=False):
    """
    Subset a time series to a specified date range.

    Parameters
    ----------
    data : pandas.Series
        Input time series indexed by datetime.
    start_date : str or pandas.Timestamp
        Start date for subsetting.
    end_date : str or pandas.Timestamp
        End date for subsetting.
    end_inclusive : bool, optional
        If True, includes the end date in the slice. Default is False.

    Returns
    -------
    pandas.Series
        Subsetted time series.
    """
    if isinstance(data, pd.Series):  # Check if the data is a pandas Series
        if start_date:
            data = data.loc[start_date:]
        if end_date:
            end_date = pd.Timestamp(end_date)  # Convert end_date to Timestamp
            if end_inclusive:
                data = data.loc[:end_date]
            else:
                data = data.loc[:end_date - pd.Timedelta(days=1)]  # Subtract one day for non-inclusive end
    else:
        # If data is not a pandas Series, return it as is (for cases where it's a numpy array or scalar)
        pass
    return data

def calculate_error_metrics(reservoir_downstream_gages, 
                            major_flows, 
                            models, 
                            output, 
                            nodes, 
                            scenarios, 
                            start_date=None, 
                            end_date=None, 
                            end_inclusive=False):
    """
    Compute error metrics for modeled vs. observed flows across nodes, models, and scenarios.

    Parameters
    ----------
    reservoir_downstream_gages : dict
        Dictionary of observed reservoir downstream gage flows.
    major_flows : dict
        Dictionary of observed major flow locations.
    models : list of str
        List of model names.
    output : object
        Simulation output object with reservoir and flow values.
    nodes : list of str
        Nodes to evaluate (reservoirs or major flows).
    scenarios : list of str
        Scenario identifiers.
    start_date : str or pandas.Timestamp, optional
        Start of evaluation period.
    end_date : str or pandas.Timestamp, optional
        End of evaluation period.
    end_inclusive : bool, optional
        Whether to include end_date in the evaluation window. Default is False.

    Returns
    -------
    pandas.DataFrame
        Multi-indexed dataframe with metrics by node, model, and scenario.
    """
    # Initialize an empty DataFrame for results
    results_metrics = pd.DataFrame()

    for node in nodes:
        print(f"\nProcessing node: {node}")

        # Determine which dataset to use based on node type
        if node in reservoir_list + ["NYCAgg"]:
            obs_results = reservoir_downstream_gages["obs"][node]
        else:
            obs_results = major_flows["obs"][node]
            
        # Iterate over models and scenarios
        for model in models:
            print(f"Processing model: {model}")
            for scenario in scenarios:
                print(f"\tProcessing scenario: {scenario}")

                # For reservoirs, access the simulated model data
                if node in reservoir_list + ["NYCAgg"]:
                    modeled_results = output.reservoir_downstream_gage[model][scenario][node]
                else:
                    modeled_results = output.major_flow[model][scenario][node]
                
                # Initialize dictionary to store error metrics
                resultsdict = {}
                # For each timescale, daily and monthly
                for timescale in ["D", "M", "Y","Full"]:
                    print(f"\t\tTimescale: {timescale}")


                    # Subset observed and modeled data based on start and end dates
                    obs = subset_timeseries(obs_results, start_date, end_date, end_inclusive)
                    modeled = subset_timeseries(modeled_results, start_date, end_date, end_inclusive)
                    
                    # Resample to monthly if timescale is "M"
                    if timescale == "M":
                        obs = obs.resample("M").mean()
                        modeled = modeled.resample("M").mean()
                    elif timescale == "Y":
                        obs = obs.resample("Y").mean()
                        modeled = modeled.resample("Y").mean()
                    elif timescale == "Full":
                        # No resampling needed, use the existing obs and modeled variables
                        pass


                    # Print the data to debug
                    print(f"\t\t\tObserved data (first 5): {obs.head()}")
                    print(f"\t\t\tModeled data (first 5): {modeled.head()}")

                    # Calculate error metrics
                    kge, r, alpha, beta = he.evaluator(he.kge, modeled, obs)
                    nse = he.evaluator(he.nse, modeled, obs)
                    logkge, logr, logalpha, logbeta = he.evaluator(he.kge, modeled, obs, transform="log")
                    lognse = he.evaluator(he.nse, modeled, obs, transform="log")

                    # Additional metrics like autocorrelation, roughness, FDC matches, etc.
                    autocorr1_obs_log = np.corrcoef(np.log(obs.iloc[1:]), np.log(obs.iloc[:-1]))[0, 1]
                    autocorr1_mod_log = np.corrcoef(np.log(modeled.iloc[1:]), np.log(modeled.iloc[:-1]))[0, 1]
                    rel_autocorr1 = autocorr1_mod_log / autocorr1_obs_log
                    autocorr7_obs_log = np.corrcoef(np.log(obs.iloc[7:]), np.log(obs.iloc[:-7]))[0, 1]
                    autocorr7_mod_log = np.corrcoef(np.log(modeled.iloc[7:]), np.log(modeled.iloc[:-7]))[0, 1]
                    rel_autocorr7 = autocorr7_mod_log / autocorr7_obs_log

                    # Roughness metric in log space
                    rel_roughness_log = np.std(np.log(modeled.iloc[1:].values) - np.log(modeled.iloc[:-1].values)) / \
                                        np.std(np.log(obs.iloc[1:].values) - np.log(obs.iloc[:-1].values))

                    # FDC horizontal and vertical match metrics
                    kss, _ = stats.ks_2samp(modeled, obs)
                    fdc_match_horiz = 1 - kss
                    obs_ordered = np.log(np.sort(obs))
                    modeled_ordered = np.log(np.sort(modeled))
                    fdc_range = max(obs_ordered.max(), modeled_ordered.max()) - min(obs_ordered.min(), modeled_ordered.min())
                    fdc_match_vert = 1 - np.abs(obs_ordered - modeled_ordered).max() / fdc_range

                    # Slope of FDC (log space)
                    sfdc_2575_obs_log = (np.log(np.quantile(obs, 0.75)) - np.log(np.quantile(obs, 0.25))) / 0.5
                    sfdc_0199_obs_log = (np.log(np.quantile(obs, 0.99)) - np.log(np.quantile(obs, 0.01))) / 0.98
                    sfdc_MinMax_obs_log = np.log(obs.max()) - np.log(obs.min())
                    sfdc_2575_mod_log = (np.log(np.quantile(modeled, 0.75)) - np.log(np.quantile(modeled, 0.25))) / 0.5
                    sfdc_0199_mod_log = (np.log(np.quantile(modeled, 0.99)) - np.log(np.quantile(modeled, 0.01))) / 0.98
                    sfdc_MinMax_mod_log = np.log(modeled.max()) - np.log(modeled.min())
                    sfdc_relBias_2575_log = sfdc_2575_mod_log / sfdc_2575_obs_log
                    sfdc_relBias_0199_log = sfdc_0199_mod_log / sfdc_0199_obs_log
                    sfdc_relBias_MinMax_log = sfdc_MinMax_mod_log / sfdc_MinMax_obs_log

                    # Store error metrics in a dictionary
                    resultsdict_inner = {
                        f"{timescale}_nse": nse[0],
                        f"{timescale}_kge": kge[0],
                        f"{timescale}_r": r[0],
                        f"{timescale}_alpha": alpha[0],
                        f"{timescale}_beta": beta[0],
                        f"{timescale}_lognse": lognse[0],
                        f"{timescale}_logkge": logkge[0],
                        f"{timescale}_logr": logr[0],
                        f"{timescale}_logalpha": logalpha[0],
                        f"{timescale}_logbeta": logbeta[0],
                        f"{timescale}_rel_autocorr1": rel_autocorr1,
                        f"{timescale}_rel_autocorr7": rel_autocorr7,
                        f"{timescale}_rel_roughness_log": rel_roughness_log,
                        f"{timescale}_fdc_match_horiz": fdc_match_horiz,
                        f"{timescale}_fdc_match_vert": fdc_match_vert,
                        f"{timescale}_sfdc_relBias_2575_log": sfdc_relBias_2575_log,
                        f"{timescale}_sfdc_relBias_0199_log": sfdc_relBias_0199_log,
                        f"{timescale}_sfdc_relBias_MinMax_log": sfdc_relBias_MinMax_log,
                    }

                    # Add to the main results dictionary
                    resultsdict.update(resultsdict_inner)
                
                # Add additional information about node, model, scenario
                resultsdict["node"] = node
                resultsdict["model"] = model
                resultsdict["scenario"] = scenario

                # Append the dictionary as a new row to the results DataFrame
                result_df = pd.DataFrame(resultsdict, index=[0])  # Create a DataFrame from resultsdict
                results_metrics = pd.concat([results_metrics, result_df], ignore_index=True)  # Concatenate

    # Set index to make it easier to query later
    results_metrics = results_metrics.set_index(["node", "model", "scenario"])
    
    return results_metrics
