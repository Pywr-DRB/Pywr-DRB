# Running Pywr-DRB Simulations


## Executables

### [`prep_input_data.py`](../API_References/prep_input_data.md)

This module prepares input-streamflow data files from different potential sources, and saves data in Pywr-ready formats. Sources include observed streamflows, modeled flows from a reconstructed historic period using [prediction in ungauged basins,](../Supplemental/pub.md) NHMv1.0, NWMv2.1, and WEAP (Aug 23, 2022, version) modeled streamflow data. For more information on different inflow datasets, see the [Data Summary page.](../Supplemental/data_summary.md)

Example:

```bash
python3 -W ignore prep_input_data.py
```

All of the processed data will be stored in `Pywr-DRB/input_data/`.

### [`drb_run_sim.py`](../API_References/drb_run_sim.md)

The `drb_run_sim.py` script is used to run a simulation of the Pywr-DRB model using a specified streamflow input data type. After data has been prepared, the [`drb_make_model()`](../API_References/drb_make_model.md) function constructs a JSON file defining the Pywr-DRB model (`Pywr-DRB/drb_model_full.json`). 

The model is run by calling the `drb_run_sim.py` file from the command line followed by an `inflow_type` argument.

The inflow type options are `obs_pub`, `nhmv10`, `nwmv21`, `nwmv21_withLakes`, and `WEAP_23Aug2022_gridmet`. For descriptions of these different inflow data, see the [Data Summary](../../Supplemental/data_summary.md) page.

Example:

```bash
python3 -W ignore drb_sun_sim.py <inflow_type> 
```

Once the model is constructed, the simulation is run and the simulation results will be stored in `Pywr-DRB/output_data/drb_output_<inflow_type>.hdf5`. See [Model Outputs](../Interpret_Results/model_outputs.md) for a description of the data contained within the output.


### [`drb_make_figs.py`](../API_References/api_references.md)

This script uses several plotting functions (stored in `Pywr-DRB.plotting`) to generate comparative figures for analysis. Executing this script after performing a simulation will result in figures being generated and stored in `Pywr-DRB/figs/`.

```{note}
The `drb_make_figs.py` script will only work if simulations have successfully been completed each of the four inflow datasets.
```
