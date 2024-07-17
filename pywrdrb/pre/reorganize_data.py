"""
This script contsins functions to reorganize the NHMv1.0 and NWMv2.1 data originally provided by Aubrey Duggar (NCAR).
Structures it into a pd.DataFrame-friendly format, and saves it to csv files in the input_data/modeled_gages directory.
"""

import pandas as pd

from pywrdrb.pre.prep_input_data_functions import read_modeled_estimates
from pywrdrb.utils.directories import input_dir

def combine_nwmv21_datasets():
    """Combines three different NWMv2.1 outputs into a single DataFrame, and exports it to a csv file.
    The datasets include NWMv2.1 modeled lake inflows, reach flows, and flows at USGS gauge locations.

    Returns:
        None: The combined dataset is exported input_data/modeled_gages/streamflow_daily_nwmv21_mgd.csv
    """
    
    # Shared column names
    sep= ','
    date_label= 'UTC_date'
    
    start_date= '1979-10-01'
    end_date= '2020-12-31'

    ## Load modeled flows at USGS gauges
    filename= f'{input_dir}modeled_gages/streamflow_daily_nwmv21.txt'

    nwm_gauge_flow = read_modeled_estimates(filename, sep='\t', 
                                            date_label= 'UTC_date', 
                                            site_label= 'site_no',
                                            streamflow_label= 'q_cms_mod',
                                            start_date= start_date, end_date= end_date)

    ## Load reach streamflow data
    filename= f'{input_dir}modeled_gages/streamflow_daily_1979_2020_nwmv21_reaches.csv'
    
    nwm_reach_flow = read_modeled_estimates(filename, sep, date_label, 
                                            site_label= 'link', 
                                            streamflow_label= 'q_cms',
                                            start_date= start_date, end_date= end_date)
    
    ## Load lake inflow data
    filename= f'{input_dir}modeled_gages/lakes_daily_1979_2020_nwmv21.csv'
    nwm_lake_inflow = read_modeled_estimates(filename, sep, date_label, 
                                            site_label= 'feature_id', 
                                            streamflow_label= 'inflow',
                                            start_date= start_date, end_date= end_date)

    # Concatenate into a single DataFrame
    all_nwm_flows= pd.concat([nwm_gauge_flow, nwm_reach_flow, nwm_lake_inflow], axis= 1)
    all_nwm_flows.index= pd.to_datetime(all_nwm_flows.index)

    # Export
    all_nwm_flows.to_csv(f'{input_dir}modeled_gages/streamflow_daily_nwmv21_mgd.csv', index_label= 'date')
    
    return None
    


if __name__ == '__main__':
    
    # Run
    combine_nwmv21_datasets()
    