# Results Set Table

For extracting results after running a simulation, below are the following currently available results set options when using the get_pywrdrb_results set function.

[`get_pywrdrb_results()`](https://pywr-drb.github.io/Pywr-DRB/api/generated/pywrdrb.post.get_pywrdrb_results.html#pywrdrb.post.get_pywrdrb_results)

| Results Set                     | Description                                                  |
|---------------------------------|--------------------------------------------------------------|
| `reservoir_downstream_gage`     | Flow data below each reservoir                                |
| `res_storage`                    | Reservoir storage dynamics                                    |
| `major_flow`                     | Flow at major flow points of interest                         |
| `res_release`                    | Reservoir releases including outflow and spill                |
| `downstream_release_target`      | Downstream release targets for specific reservoirs            |
| `inflow`                         | Inflow data for each catchment                                |
| `catchment_withdrawal`           | Withdrawal from each catchment                                |
| `catchment_consumption`          | Consumption from each catchment                               |
| `prev_flow_catchmentWithdrawal`  | Previous flow at catchment withdrawal points                  |
| `max_flow_catchmentWithdrawal`   | Maximum flow at catchment withdrawal points                   |
| `max_flow_catchmentConsumption`  | Maximum flow at catchment consumption points                  |
| `res_level`                      | Reservoir drought levels                                      |
| `ffmp_level_boundaries`          | Flood frequency management program level boundaries           |
| `mrf_target`                     | Management of river basin (DRB) targets                       |
| `nyc_release_components`         | NYC release components including targets, flood releases, etc.|
| `lower_basin_mrf_contributions`  | Contributions to lower basin management of river basin (DRB)  |
| `ibt_demands`                    | IBT (Inter-basin transfer) demands for NYC and NJ             |
| `ibt_diversions`                 | IBT diversions for NYC and NJ                                 |
| `mrf_targets`                    | MRF (Montague River Flow) targets                             |
| `all_mrf`                        | All MRF-related data                                          |
| `temperature`                    | Temperature data                                              |
