#############################################################################

# Marilyn Simth Summer 2024 Flow Duration Curve Plotting
# This script is used to plot the flow duration curve for a specific reservoir.

#Things to include for future development
#Connect to figdir and colordict used in the other figures 



#############################################################################

import matplotlib.pyplot as plt

def plot_inflows(inflows, node, model, start_date=None, end_date=None, 
                 end_inclusive=False, colordict=None, save_fig=False, fig_dir=None,
                 log_scale=False):
    """
    Plots the inflow data for a specific node using major inflow data.
    
    Parameters:
    inflows (dict): Dictionary containing inflow data for each model.
    node (str): The node to plot the inflow data for.
    model (str): The model to use for plotting.
    start_date (str or None): The start date for the data to plot.
    end_date (str or None): The end date for the data to plot.
    end_inclusive (bool): Whether to include the end date in the data subset.
    colordict (dict or None): Dictionary of colors for plotting.
    save_fig (bool): Whether to save the figure.
    fig_dir (str or None): The directory to save the figure in.
    log_scale (bool): Whether to use a logarithmic scale for the y-axis.
    
    Returns:
    None
    """
    
    # Default colors if not provided
    if colordict is None:
        colordict = {model: 'blue'}
    
    # Function to get the correct data
    def get_fig_data(model, node):
        if node in inflows[model][0]:
            data = subset_timeseries(inflows[model][0][node], start_date, end_date)
        else:
            print(f'get_fig_data() not set for node {node}')
            data = None
        return data

    # Retrieve the inflow data for the specified node
    inflow_data = get_fig_data(model, node)
    
    if inflow_data is None:
        print(f"Cannot make plot for node: {node} - Data not available.")
        return

    # Plotting the inflow data
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(inflow_data.index, inflow_data.values, color=colordict[model], label=model)

    # Formatting the plot
    ax.set_xlabel('Date')
    ax.set_ylabel('Inflow (MCM)')
    ax.legend()
    ax.grid(True)
    plt.title(f'Inflow Time Series for {node}')
    
    # Apply log scale if requested
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Log Inflow (MCM)')
    
    # Save the figure if required
    if save_fig and fig_dir:
        fig.savefig(f'{fig_dir}/inflow_{node}.png', bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.close()
