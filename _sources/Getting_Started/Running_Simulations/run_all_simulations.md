# Run all model simulations

## Executing the model

The model is run by calling the `drb_run_sim.py` file from the command line, with arguments for `inflow_type` and `backup_inflow_type`.

The inflow type options are `obs_pub`, `nhmv10`, `nwmv21`, `nwmv21_withLakes`, and `WEAP_23Aug2022_gridmet`. For descriptions of these different inflow data, see the [Data Summary](../../Supplemental/data_summary.md) page.

Example:

```BASH
cd <model base directory>
python3 drb_run_sim.py <inflow_type> <backup_inflow_type>
```

```{note}
The `backup_inflow_type` data is only necessary when running `WEAP_gridmet` as the main `inflow_type`.
```

## Run all simulations

To run the Pywr-DRB model using all available streamflow inputs (historic reconstruction, NHM, NWM, and WEAP) run the following command:

```BASH
sh drb_run_all.sh
```

Results will be stored in the `output_data/drb_output_<inflow_type>.hdf5` folder, where `<inflow_type>` is the user-specified argument, as discussed above.

See [Model Outputs](../Interpret_Results/model_outputs.md) for a description of the data contained within the output.

## Generate figures

Several plotting functions are available to visualize the model results. The `drb_make_figs.py` module produces these figures using the latest model outputs. All figures can be produced if all model simulations have been run, by executing the command:

```bash
python3 drb_make_figs.py
```

See [Visualizing Results](../Interpret_Results/output_figures.md) for descriptions of the resulting figures.
