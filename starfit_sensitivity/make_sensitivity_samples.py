#####
#Script to make the LHS samples and export to hdf5
####

#import the necessary libraries
import numpy as np
import pandas as pd
import SALib.sample.latin as latin

# set path to pywrdrb
path_to_pywrdrb = '../'
import sys
sys.path.append(path_to_pywrdrb)

# Import custom modules and functions from pywrdrb
from pywrdrb.utils.directories import model_data_dir

# Reservoirs of interest for sensitivity analysis
reservoirs_of_interest = ['modified_blueMarsh', 'modified_fewalter', 'modified_beltzvilleCombined', 'prompton']

# add a print statement 
print(f"Reservoirs of interest: {reservoirs_of_interest}")

# Parameter bounds
parameter_bounds = {
    'NORhi_alpha': [-0.2461, -0.2139], 'NORhi_beta': [-2.1507, -1.8693], 'NORhi_max': [93.0000, 100], 
    'NORhi_min': [0.0000, 2.0000], 'NORhi_mu': [14.0244, 16.1356], 'NORlo_alpha': [0.1860, 0.2140],
    'NORlo_beta': [-4.3656, -3.7944], 'NORlo_max': [14.4336, 16.6064], 'NORlo_min': [11.4483, 13.1717],
    'NORlo_mu': [11.6994, 13.4606], 'Release_alpha1': [-4.08, 240], 'Release_alpha2': [-0.5901, 84.7844],
    'Release_beta1': [-1.2104, 83.9024], 'Release_beta2': [-52.3545, 0.4454], 'Release_max': [-1, 2],
    'Release_min': [-1, 0], 'Release_c': [-1.4, 63.516], 'Release_p1': [0, 17.02], 'Release_p2': [0, 0.957]
}

# Number of LHS samples per reservoir
n_samples = 2000

# add a print statement 
print(f"Number of LHS samples per reservoir: {n_samples}")

# Set random seed for reproducibility
np.random.seed(42)

# Generate LHS samples
def generate_lhs_samples(param_ranges, n_samples):
    problem = {
        'num_vars': len(param_ranges),
        'names': list(param_ranges.keys()),
        'bounds': list(param_ranges.values())
    }
    samples = latin.sample(problem, n_samples)
    return samples

# Empty list to hold individual reservoir DataFrames
reservoir_dfs = []

# Loop through each reservoir and generate samples
for reservoir in reservoirs_of_interest:
    # Generate LHS samples for the current reservoir
    lhs_samples = generate_lhs_samples(parameter_bounds, n_samples)
    
    # Create a DataFrame for the LHS samples
    lhs_df = pd.DataFrame(lhs_samples, columns=parameter_bounds.keys())
    
    # Add columns for reservoir name and sample ID
    lhs_df['reservoir'] = reservoir
    lhs_df['sample_id'] = np.arange(1, n_samples+1)
    
    # Reorder columns so 'reservoir' and 'sample_id' are first
    lhs_df = lhs_df[['reservoir', 'sample_id'] + [col for col in lhs_df.columns if col not in ['reservoir', 'sample_id']]]
    
    # Append to list of DataFrames
    reservoir_dfs.append(lhs_df)

# Combine all reservoir DataFrames into one
reservoir_samples = pd.concat(reservoir_dfs, ignore_index=True)
print(f"Combined DataFrame shape: {reservoir_samples.shape}")

#save the reservoir_samples to a csv file in the model data dir
reservoir_samples.to_csv(f"{model_data_dir}reservoir_samples.csv", index=False)
print("Sensitivity samples saved to 'sensitivity_samples.csv'")


# Get the constants for the reservoirs to match pywrdrb parameter requriements 
#subset the default dataframe to only include GRanD_CAP_MG,GRanD_MEANFLOW_MGD,Adjusted_CAP_MG,Adjusted_MEANFLOW_MGD,Max_release,
#and the reservoir name
reservoir_constants = pd.read_csv(f"{model_data_dir}drb_model_istarf_conus.csv")
required_columns = ['reservoir', 'GRanD_CAP_MG', 'GRanD_MEANFLOW_MGD', 'Adjusted_CAP_MG', 'Adjusted_MEANFLOW_MGD', 'Max_release']
reservoir_constants = reservoir_constants[required_columns]
#reservoir_constants.head()
print(f"Reservoir Constants subset shape: {reservoir_constants.shape}")

#Create dataframe for unsampled reservoirs to be used in the hdf5 file
# All reservoirs that are used in the model (including those not under SA)
all_reservoirs =  ['cannonsville' 'pepacton' 'neversink' 'wallenpaupack_original'
    'blueMarsh' 'beltzville' 'beltzvilleCombined' 'wallenpaupack'
    'prompton_original' 'shoholaMarsh_original' 'mongaupeCombined_original'
    'fewalter_original' 'merrillCreek_original' 'hopatcong_original'
    'nockamixon_original' 'assunpink_original' 'ontelaunee_original'
    'stillCreek_original' 'greenLane_original' 'prompton' 'shoholaMarsh'
    'mongaupeCombined' 'fewalter' 'merrillCreek' 'hopatcong' 'nockamixon'
    'assunpink' 'vanSciver' 'ontelaunee' 'stillCreek' 'greenLane'
    'modified_blueMarsh' 'modified_fewalter' 'modified_beltzvilleCombined']

default_df = pd.read_csv(f"{model_data_dir}drb_model_istarf_conus.csv")

#drop these columns 'GRanD_ID', 'GRanD_NAME', 'GRanD_CAP_MCM','GRanD_MEANFLOW_CUMECS', 'Obs_MEANFLOW_CUMECS', 'fit', 'match','notes'
default_df = default_df.drop(columns=['GRanD_ID', 'GRanD_NAME', 'GRanD_CAP_MCM','GRanD_MEANFLOW_CUMECS', 'Obs_MEANFLOW_CUMECS', 'fit', 'match','notes'])

# Filter the default_df for the reservoirs in reservoir list that are not in reservoirs of interest
unsampled_reservoirs = default_df[~default_df['reservoir'].isin(reservoirs_of_interest)]
print(f"Unsampled reservoirs DataFrame shape: {unsampled_reservoirs.shape}")


# Create the hdf5 file

def create_scenario_data_hdf5(reservoir_samples, reservoir_constants,unsampled_reservoirs, hdf5_filename, default_df):
    # Merge the DataFrames on 'reservoir' column for the SA samples
    merged_df = pd.merge(reservoir_samples, reservoir_constants, on='reservoir')
    
    # Create an HDF5 file and set the group name to 'starfit'
    with pd.HDFStore(hdf5_filename, mode='w') as store:
        # Save scenario 0 with default parameters
        store.put('/starfit/scenario_0', default_df)
        print(f"Scenario 0 (default parameters) saved to HDF5. Shape: {default_df.shape}")

        # Loop through each unique sample_id
        for sample_id in reservoir_samples['sample_id'].unique():
            # Subset the merged_df to the chosen sample_id
            sample_df = merged_df[merged_df['sample_id'] == sample_id]
            
            # Merge the sample df with the unsampled_reservoirs (which excludes reservoirs of interest)
            final_df = pd.concat([sample_df, unsampled_reservoirs], ignore_index=True)
            
            # Define the HDF5 path for each scenario under the 'starfit' group
            hdf5_path = f'/starfit/scenario_{sample_id}'
            
            # Save the DataFrame into the HDF5 file
            store.put(hdf5_path, final_df)
            
            # Optionally, print out feedback for each scenario
            print(f"Scenario {sample_id} saved to HDF5. Shape: {final_df.shape}")


hdf5_filename = f'{model_data_dir}scenarios_data.h5'

create_scenario_data_hdf5(reservoir_samples, reservoir_constants, unsampled_reservoirs, hdf5_filename, default_df)
print(f"Scenario data saved to {hdf5_filename}")