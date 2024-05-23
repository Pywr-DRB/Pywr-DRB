"""
Organize data records into appropriate format for Pywr-DRB.

Observed records (USGS gages) & WRF-Hydro modeled estimates for multiple WRF-Hydro configurations.

"""
 
import numpy as np
import pandas as pd


from pywr_drb_node_data import obs_site_matches, wrf_hydro_site_matches

from utils.directories import input_dir

from pre.disaggregate_DRBC_demands import disaggregate_DRBC_demands
from pre.extrapolate_NYC_NJ_diversions import extrapolate_NYC_NJ_diversions
from pre.predict_inflows_diversions import predict_inflows_diversions
from pre.prep_input_data_functions import match_gages

use_mpi = True

if use_mpi:
    import mpi4py as MPI
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Gets the rank (ID) of the current process
    size = comm.Get_size()  # Gets the total number of processes
else:
    rank = 0
    size = 1

if __name__ == "__main__":

    ### set random seed for consistency
    np.random.seed(1)

    # Dates depending on config climate:
    date_ranges = {
        '1960s': ('1959-10-01', '1969-12-31'),
        'aorc': ('1979-10-01', '2021-12-31'),
        '2050s': ('2051-10-01', '2061-12-31'),
    }
    
    # change daterange for 2050s to be same as 1960s for now
    # will need to setup future forecasts of other inputs to 
    # run the 2050s otherwise...
    date_ranges['2050s'] = date_ranges['1960s']
    
    # WRF-Hydro model configuration to use 
    # must have corresponding CSV in input_dir/modeled_gages/ already (see Input-Data-Retrival repo)
    
    # List of WRF-Hydro configurations to prepare data 
    prep_configs = []
    calibration = 'calib'
    landcover = 'nlcd2016'
    
    for climate in ['1960s', 'aorc', '2050s']:
        c = {
            'climate': climate,
            'calibration': calibration,
            'landcover': landcover
        }
        prep_configs.append(c)

    # Distribute tasks using a round-robin approach
    rank_configs = [prep_configs[i] for i in range(rank, len(prep_configs), size)]

    for config in rank_configs:
        print(f'Preparing data for {config["climate"]} {config["calibration"]} {config["landcover"]}')
    
        start_data = date_ranges[config['climate']][0]
        end_date = date_ranges[config['climate']][1]
        
        dataset_label = f'wrf{config["climate"]}_{config["calibration"]}_{config["landcover"]}'
        source_fname = f'{input_dir}modeled_gages/streamflow_daily_{dataset_label}.csv'
    
    
        # load WRF-Hydro model outputs
        df_wrf_hydro = pd.read_csv(source_fname, index_col=0, parse_dates=True) 
        
        # change 2050s datetime index to match 1960s
        if config['climate'] == '2050s':
            df_wrf_hydro.index = pd.date_range(start_data, end_date, freq='D')         
    
        ### match USGS gage sites to Pywr-DRB model nodes & save inflows to csv file in format expected by Pywr-DRB
        match_gages(df_wrf_hydro, dataset_label=dataset_label, 
                    site_matches_id= wrf_hydro_site_matches)
        
        ### create predicted future Montague & Trenton inflows & NJ diversions based on lagged regressions, for scheduling NYC releases
        # 75% of the data is used for training, 25% for testing
        start_date_training = start_data
        training_frac = int(len(df_wrf_hydro.index) * 0.75)
        training_end_year = df_wrf_hydro.index[training_frac].year
        end_date_training = f'{training_end_year}-12-31'
        
        predict_inflows_diversions(dataset_label, 
                                   start_date_training, end_date_training, make_figs=True)
        
    ### create hybrid datasets (not implemented for WRF-Hydro, but should be possible)
    # create_hybrid_modeled_observed_datasets('nhmv10', df_nhm.index)

    # Get NYC & NJ diversions. for time periods we dont have historical record, extrapolate by seasonal relationship to flow.
    # Don't need to run this every time; it is already saved in input_dir for 1952-2022
    # download_USGS_data_NYC_NJ_diversions()    
    # extrapolate_NYC_NJ_diversions('nyc', make_figs=True)
    # extrapolate_NYC_NJ_diversions('nj', make_figs=True)
    
    ### get catchment demands based on DRBC data
    # sw_demand = disaggregate_DRBC_demands() # only need to run this once