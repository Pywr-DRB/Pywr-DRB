Tasks 
1. naming rules
2. streamline the observation updates
3. streamline the final model inputs update (include inflows and all other constants


Test the inflow creation ABC
Complete all data
test model builder


Historic reservoir opt
Look like to core model dont need anything from this folder except get_result()


it is unclear to me the current data arrangement in the modeled_gages/ and the folder naming logic


usgs_gages/ are only used for diversions extrapolations
=> oberservations
Are we expecting uses to do this?






New Structure

Each inflow type should have its own folder under flows/
Q: I only see the following functions to output the hybrid dataset but no pure nhm nwm

	create_hybrid_modeled_observed_datasets("nhmv10", df_nhm.index)
    create_hybrid_modeled_observed_datasets("nwmv21", df_nwm.index)
I found "match_gages()"
But I dont understand obs and obs_pub

gauge flow is still the inflows?

observation by node 
observation by gauge
 


# get_result.py  => dir need to be updated


