#############################################################################

# Marilyn Simth Summer 2024 Flow Duration Contribution to the Reservoir
# This script is used to plot the individual contributions to the reservoir

#Things to include for future development
#Connect to figdir and colordict used in the other figures 
#need to turn into function 
#add the lag to the contirbution calcuaiton to see if that improves the accuracy 
#this is currently underconstruction 
#changed all get results to the new get results 


#############################################################################
#potentailly required packages
import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib import cm
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from scipy import stats

import hydroeval as he

from pywrdrb.pywr_drb_node_data import upstream_nodes_dict, downstream_node_lags, immediate_downstream_nodes_dict

# Custom modules
#from pywrdrb.utils.constants import mg_to_mcm
from pywrdrb.utils.lists import reservoir_list, reservoir_list_nyc, majorflow_list, seasons_dict 
from pywrdrb.utils.lists import drbc_lower_basin_reservoirs

from pywrdrb.utils.directories import input_dir, fig_dir, model_data_dir, spatial_data_dir
from pywrdrb.plotting.styles import base_model_colors, paired_model_colors, scatter_model_markers
from pywrdrb.plotting.styles import node_label_full_dict, model_label_dict
from pywrdrb.utils.timeseries import subset_timeseries

##GET THE DATA
output_dir = '../output_data/'
input_dir = '../input_data/'
print(f'Retrieving simulation data.')

#pywr results
pywr_models = ['nhmv10_withObsScaled']
for model in pywr_models:
    print(f'pywr_{model}')
    reservoir_downstream_gages[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, results_set='reservoir_downstream_gage', datetime_index=datetime_index)
    reservoir_releases[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, results_set='res_release', datetime_index=datetime_index)
    major_flows[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, results_set='major_flow', datetime_index=datetime_index)
    
pywr_models = [f'pywr_{m}' for m in pywr_models]

#base results 
base_model = ['obs', 'nhmv10_withObsScaled']

datetime_index = list(reservoir_downstream_gages.values())[0].index
for model in base_model:
    print(model)
    reservoir_downstream_gages[model], datetime_index = get_base_results(input_dir, model, results_set='reservoir_downstream_gage', datetime_index=datetime_index)
    reservoir_releases[model], datetime_index = get_base_results(input_dir, model, results_set='res_release', datetime_index=datetime_index)
    major_flows[model], datetime_index = get_base_results(input_dir, model, results_set='major_flow', datetime_index=datetime_index)


# List of nodes to process
nodes = ['01417000', '01425000', '01433500', '01436000', '01447800', '01449800', '01463620', '01470960', 'delDRCanal', 'delLordville', 'delMontague', 'delTrenton', 'outletAssunpink', 'outletSchuylkill']

# Simulation start and end dates
start_date = '1983-10-01'
end_date = '2020-09-30'

for node in nodes:
    print(f"Processing node: {node}")

    # Step 1: Get total simulated and observed flow
    total_sim_node_flow = subset_timeseries(major_flows['pywr_nhmv10_withObsScaled'][node], start_date, end_date)
    total_base_node_flow = subset_timeseries(major_flows['nhmv10_withObsScaled'][node], start_date, end_date)

    # Step 2: Identify contributing flows
    contributing = upstream_nodes_dict[node]

    # Initialize a DataFrame to store the results
    contribution_df = pd.DataFrame(index=total_sim_node_flow.index)

    # Step 3: Calculate percentage contributions relative to total simulated flow
    for upstream_reservoir in contributing:
        if upstream_reservoir in reservoir_releases['pywr_nhmv10_withObsScaled']:
            upstream_reservoir_data = subset_timeseries(reservoir_releases['pywr_nhmv10_withObsScaled'][upstream_reservoir], start_date, end_date)
            contribution_percentage = (upstream_reservoir_data / total_sim_node_flow) * 100
            contribution_df[upstream_reservoir] = contribution_percentage

    # Print the resulting DataFrame to console
    print(f"Contribution data for node: {node}")
    print(contribution_df)

    # Ensure the sum of contributions is close to 100%
    contribution_sums = contribution_df.sum(axis=1)
    if not all((contribution_sums - 100).abs() < 5):  # Allowing a small tolerance
        print("Warning: The sum of contributions is not close to 100% for all time steps.")
    else:
        print("The sum of contributions is close to 100% for all time steps.")

    # Diagnostic step: Identify major discrepancies

    # Calculate the difference from 100%
    difference_from_100 = contribution_sums - 100

    # Find the time steps with the largest discrepancies
    large_discrepancies = difference_from_100.abs().sort_values(ascending=False).head(10)
    print("Top 10 time steps with the largest discrepancies:")
    print(large_discrepancies)

    # Identify which reservoirs are responsible for discrepancies
    discrepancy_threshold = 5  # Define a threshold for significant discrepancy
    significant_discrepancies = difference_from_100[difference_from_100.abs() >= discrepancy_threshold]

    # Count occurrences of significant discrepancies for each reservoir
    reservoir_discrepancy_count = (contribution_df.loc[significant_discrepancies.index].isna().sum() + contribution_df.loc[significant_discrepancies.index] == 0).sum(axis=0)
    print("\nReservoirs responsible for significant discrepancies:")
    print(reservoir_discrepancy_count)

    # Plot the sum of contributions to visually inspect
    plt.figure(figsize=(14, 7))
    plt.plot(contribution_sums, label='Sum of Contributions')
    plt.axhline(y=100, color='r', linestyle='--', label='Expected 100%')
    plt.title(f'Sum of Reservoir Contributions Over Time for Node {node}')
    plt.xlabel('Time')
    plt.ylabel('Sum of Contributions (%)')
    plt.legend()
    plt.show()

    # Print detailed inspection for one of the largest discrepancies
    time_step_to_inspect = large_discrepancies.index[0]
    print(f"\nDetailed inspection for {time_step_to_inspect} for node {node}:")
    print(contribution_df.loc[time_step_to_inspect])
    print(f"Total simulated flow: {total_sim_node_flow.loc[time_step_to_inspect]}")

    # Save the DataFrame to a CSV file for each node
    #output_csv_path = f'../output_data/contribution_percentages_{node}.csv'
    #contribution_df.to_csv(output_csv_path)
    #print(f"Contribution percentages for node {node} saved to {output_csv_path}")

    # Print the resulting DataFrame to console
    #print(contribution_df)