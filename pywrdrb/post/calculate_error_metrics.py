# The script contains functions to calculate error metrics for the model outputs
# Marilyn Smith 

import hydroeval as he
from scipy import stats
import pandas as pd
from pywrdrb.utils.lists import reservoir_list
import numpy as np
from dataretrieval import nwis

def fetch_nwis_data(site_number, start_date, end_date):
    """
    Fetches daily streamflow data from NWIS.
    
    Parameters:
    site_number (str): The USGS site number to fetch data from.
    start_date (str): The start date for the data.
    end_date (str): The end date for the data.
    
    Returns:
    pd.Series: The average daily discharge data.
    """
    parameter_code = '00060'  # Discharge
    daily_streamflow = nwis.get_dv(sites=site_number, parameterCd=parameter_code, start=start_date, end=end_date)
    if daily_streamflow:
        data = daily_streamflow[0]
        data.index = data.index.date
        return data['00060_Mean']
    else:
        print(f"No data retrieved for site {site_number}")
        return None
    
def subset_timeseries(data, start_date, end_date, end_inclusive=False):
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
                            models, 
                            output, 
                            nodes, 
                            scenarios, 
                            start_date=None, 
                            end_date=None, 
                            end_inclusive=False,
                            fetch_nwis=False):
    # Initialize an empty DataFrame for results
    results_metrics = pd.DataFrame()

    for node in nodes:
        print(f"\nProcessing node: {node}")

        # Determine which dataset to use based on node type
        # Define the NWIS site numbers
        nwis_sites = {
            'prompton': '01430000',
        }

        try:
            obs_results = reservoir_downstream_gages['obs'][node]
        except KeyError:
            print(f"Data for '{node}' not found in 'obs'. Available keys: {reservoir_downstream_gages['obs'].keys()}")
            obs_results = None
    
        # Check if obs_data is None and fetch NWIS data if needed
        if obs_results is None and fetch_nwis:
            site_number = nwis_sites.get(node)  # Retrieve the site number from nwis_sites
            if site_number:
                obs_results = fetch_nwis_data(site_number, start_date, end_date)
                obs_results.index = pd.to_datetime(obs_results.index)
            
        # Iterate over models and scenarios
        for model in models:
            #print(f"Processing model: {model}")
            for scenario in scenarios:
                #print(f"\tProcessing scenario: {scenario}")

                # For reservoirs, access the simulated model data
                modeled_results = output.reservoir_downstream_gage[model][scenario][node]
                
                # Initialize dictionary to store error metrics
                resultsdict = {}
                # For each timescale, daily and monthly
                for timescale in ["D", "W", "M","A"]:
                    #print(f"\t\tTimescale: {timescale}")


                    # Subset observed and modeled data based on start and end dates
                    obs = subset_timeseries(obs_results, start_date, end_date, end_inclusive)
                    modeled = subset_timeseries(modeled_results, start_date, end_date, end_inclusive)

                    # **Filter out zero values from both observed and modeled data**
                    obs = obs.loc[obs > 0]  # Keep only positive values
                    modeled = modeled.loc[modeled > 0]  # Keep only positive values

                    # Ensure indices overlap for both datasets
                    common_index = obs.index.intersection(modeled.index)
                    obs = obs.loc[common_index]
                    modeled = modeled.loc[common_index]
                    
                    # Resample to monthly if timescale is "M"
                    if timescale == "W":
                        obs = obs.resample("W").mean()
                        modeled = modeled.resample("W").mean()
                    elif timescale == "M":
                        obs = obs.resample("ME").mean()
                        modeled = modeled.resample("ME").mean()
                    elif timescale == "A":
                        obs = obs.resample("YE").mean()
                        modeled = modeled.resample("YE").mean()
                    elif timescale == "D":
                        pass
                            
                    # Print the data to debug
                    #print(f"\t\t\tObserved data (first 5): {obs.head()}")
                    #print(f"\t\t\tModeled data (first 5): {modeled.head()}")

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
