"""
Contains information about the date ranges for internally available datasets.

Overview: 
It's pretty simple. Contains start and end dates for different datasets. 

Technical Notes: 
- This can probably be improved and/or removed in the future:
    - i.e., it may be added to the Options class or as a sepearate data class
    - but we should think about what we want before making changes.
- The `model_date_ranges` are currently only used in the notebook tutorials
- The `temp_pred_date_range` is used in the LSTM temperature parameter classes.
    
Links: 
- NA
 
Change Log:
TJA, 2025-05-05, Add docs.
"""

# dictionary with {"dataset": (start_date, end_date)}
model_date_ranges = {}

for nxm in ["nhmv10", "nwmv21"]:
    # Datasets from Hamilton et al. (2024)
    model_date_ranges[nxm] = ("1983-10-01", "2016-12-31")
    model_date_ranges[f"{nxm}_withObsScaled"] = ("1983-10-01", "2016-12-31")

# WRF-Hydro simulations
model_date_ranges["wrf1960s_calib_nlcd2016"] = ("1959-10-01", "1969-12-31")
model_date_ranges["wrf2050s_calib_nlcd2016"] = ("1959-10-01", "1969-12-31")
model_date_ranges["wrfaorc_calib_nlcd2016"] = ("1979-10-01", "2021-12-31")

# Reconstruction datasets
model_date_ranges["pub_nhmv10_BC_withObsScaled"] = ("1945-01-01", "2023-12-31")

## The date range where temperature prediction LSTM is able to be run
temp_pred_date_range = ("1982-04-03", "2021-04-15")
