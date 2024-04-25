# Model Outputs

Simulation results are stored in the `output_data/` folder, and are named `drb_output_<inflow_type>.hdf5` where `inflow_type` corresponds to one of the naming conventions listed in [Data Summary](../../Supplemental/data_summary.md).

The function [`get_pywr_results()`](../../API_References/drb_make_figs.md) is used to transform the `hdf5` file into a `pd.DataFrame` containing simulated results at different nodes.

