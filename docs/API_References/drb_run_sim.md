# `drb_run_sim.py`

The module is used to execute a simulation of the Pywr-DRB model.

Two command line arguments set the `inflow_type` and `backup_inflow_type` variables. These variables are used to specify the data source and backup data source for the inflow data used in the model. For more information on available inflow data, see the [Data Summary page.](../Supplemental/data_summary.md)

The code then imports some custom parameters for the model and generates a JSON file with model data using the `drb_make_model` function. The model is then loaded from the JSON file using the `Model.load` method.

A `TablesRecorder` is added to the model to record the results of the model run to an output file specified by the `output_filename` variable. Finally, the model is run and the resulting statistics are converted to a pandas DataFrame.

## Function calls:
- [`drb_make_model()`](./drb_make_model.md)
> This function is called and used to construct the JSON file defining the Pywr-DRB model.
