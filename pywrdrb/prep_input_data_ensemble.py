"""
Organize data records into appropriate format for Pywr-DRB.

This script handles ensemble inflow datasets. 

Predictions of inflows and diversions are made individually for each ensemble member. 
The predictions are parallelized using MPI.

This script can be used to prepare inputs from the following datasets:
- PUB reconstruction ensembles (obs_pub_*_ensemble)
- Syntehtically generated ensembles (syn_*_ensemble)

"""
 
from pre.disaggregate_DRBC_demands import disaggregate_DRBC_demands
from pre.extrapolate_NYC_NJ_diversions import extrapolate_NYC_NJ_diversions, download_USGS_data_NYC_NJ_diversions
from pre.predict_inflows_diversions import predict_ensemble_inflows_diversions


if __name__ == "__main__":

    ## List of ensemble datasets to process
    ensemble_datasets = [
        'obs_pub_nhmv10_ObsScaled_ensemble', 
        'obs_pub_nwmv21_ObsScaled_ensemble']


    ## Optional data processing steps
    # Generally, these only need to be run once.  Assuming already run, set to False.
    download_flow_data_for_diversion_extrapolation = False
    extrapolate_diversions = False
    disaggregate_demands = False

    ## Get NYC & NJ diversions. 
    # For time periods w/ no historical record, extrapolate by seasonal relationship to flow.
    if download_flow_data_for_diversion_extrapolation:
        download_USGS_data_NYC_NJ_diversions()    
    if extrapolate_diversions:
        extrapolate_NYC_NJ_diversions('nyc')
        extrapolate_NYC_NJ_diversions('nj')
        
    ## Spatially disaggregate DRBC demands to match PywrDRB node catchments
    # Source data is from DRBC report
    if disaggregate_demands:
        sw_demand = disaggregate_DRBC_demands()

    ## Predict future Montague & Trenton inflows & NJ diversions based on lagged regressions
    # Predictions are used to inform NYC releases on day t needed to meet flow targets on days t+n (n=1,..4)    
    for dataset in ensemble_datasets:
        print(f'Making inflow & diversion predictions for {dataset}...')
        predict_ensemble_inflows_diversions(dataset, '1945/01/01', '2022/12/31')
