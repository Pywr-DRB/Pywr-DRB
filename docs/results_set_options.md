# Results Set Table

For extracting results after running a simulation, below are the following currently available results set options when using the `pywrdrb.Data.load_output()` functionality.

| Results Set                     | Description                                                  |
|---------------------------------|--------------------------------------------------------------|
| `"reservoir_downstream_gage"`     | Flow at the nearest gauge below each reservoir. This is majority reservoir release plus any unmanaged flows which enter the gauge catchment below the reservoir (magnitude depends how far from the reservoir the gauge is).                           |
| `"res_storage"`                    | Reservoir storages.                                    |
| `"major_flow"`                     | Flow at major node locations including USGS gauges (e.g., Montague, and Trenton gauges) and confluence points (e.g., confluence of Schuykill and Delaware Rivers).                         |
| `"res_release"`                    | Reservoir releases.                |
| `"downstream_release_target"`      | Downstream total release targets for NYC reservoirs. This is the total release from NYC reservoirs, unless the reservoir has sufficient storage to meet the target release.             |
| `"inflow"`                         | Inflow data for each node. This is reservoir inflow, or marginal catchment inflow for downstream nodes.                              |
| `"catchment_withdrawal"`           | Daily withdrawal from each catchment.                                |
| `"catchment_consumption"`          | Daily consumptive use from each catchment.                               |
| `"ffmp_level_boundaries"`          | NYC reservoir operational levels, as defined by the FFMP. This is constant for every year of simulation, and is not inflow-dependent.           |
| `"mrf_targets"`                     | Minimum regulated flow (MRF) targets at Montague and Trenton. Note that the NYC reservoirs are used to maintain the Montague target as per the US Supreme Court Decrees, while Trenton flow target is primarily met using USACE reservoirs with some constrained annual support (\<=6.09BG) from NYC reservoirs.                        |
| `"nyc_release_components"`         | NYC reservoir release components including flood releases, conservation releases, downstream contributions, etc.|
| `"lower_basin_mrf_contributions"`  | Contributions to the Trenton equivalent flow target from the USACE lower basin reservoirs  |
| `"ibt_demands"`                    | Inter-basin transfer (IBT) demands to NYC and NJ.             |
| `"ibt_diversions"`                 | IBT diversions for NYC and NJ                                 |
| `"temperature"`                    | Streamflow temperature (degrees C) at the Lordville gauge using an LSTM predictive model.  |
| `"salinity"`                    | Salt front location (in units of river miles upstream from the Delaware Estuary non-tidal zone) modeled using an LSTM predictive model.|
