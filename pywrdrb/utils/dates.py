"""
Contains information about dates of interest:
- start and end dates for different model datasets
"""


# dictionary with {"dataset": (start_date, end_date)}
model_date_ranges = {}

for nxm in ['nhmv10', 'nwmv21']:
    
    # Datasets from Hamilton et al. (2024)
    model_date_ranges[nxm] = ('1983-10-01', '2016-12-31')
    model_date_ranges[f'{nxm}_withObsScaled'] = ('1983-10-01', '2016-12-31')

    # Historic reconstructions
    model_date_ranges[f'obs_pub_{nxm}_ObsScaled'] = ('1945-01-01', '2022-12-31')
    model_date_ranges[f'obs_pub_{nxm}_ObsScaled_ensemble'] = ('1945-01-01', '2022-12-31')

    # Synthetic ensembles
    model_date_ranges[f'syn_obs_pub_{nxm}_ObsScaled_ensemble'] = ('1945-01-01', '2021-12-31')

# WRF-Hydro simulations
# not yet implemented