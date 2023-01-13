from pywr.model import Model
from pywr.recorders import TablesRecorder
from drb_make_model import drb_make_model
from custom_pywr import FfmpNycRunningAvgParameter, FfmpNjRunningAvgParameter
import sys

### specify inflow type from command line args
inflow_type = sys.argv[1]
backup_inflow_type = sys.argv[2]

# inflow_type = 'WEAP_23Aug2022_gridmet'  ### nhmv10, nwmv21, nwmv21_withLakes, obs, WEAP_23Aug2022_gridmet
# backup_inflow_type = 'nhmv10'  ## for WEAP inflow type, we dont have all reservoirs. use this secondary type for missing.

model_filename = "model_data/drb_model_full.json"
if 'WEAP' in inflow_type:
    output_filename = f"output_data/drb_output_{inflow_type}_{backup_inflow_type}.hdf5"
else:
    output_filename = f"output_data/drb_output_{inflow_type}.hdf5"

### import custom pywr params
FfmpNycRunningAvgParameter.register()  # register the name so it can be loaded from JSON
FfmpNjRunningAvgParameter.register()  # register the name so it can be loaded from JSON

### make model json files
drb_make_model(inflow_type, backup_inflow_type)

### Load the model
model = Model.load(model_filename)

### Add a storage recorder
TablesRecorder(model, output_filename, parameters=[p for p in model.parameters if p.name])

### Run the model
stats = model.run()
stats_df = stats.to_dataframe()




# ### plot data
# for name, df in TablesRecorder.generate_dataframes(output_filename):
#     # df.columns = ["Very low", "Low", "Central", "High", "Very high"]
#     df.columns = ["Central"]
#
#     # if name.split('_')[0] in ("reservoir"):
#     # if name.split('_')[0] in ("reservoir", "outflow", "flow", "link", "flowtarget", "demand"):
#     # if 'target' in name.split('_'):
#     # reservoir = 'neversink'
#     # if name in ('reservoir_'+reservoir, 'flow_'+reservoir, 'outflow_'+reservoir, 'link_'+reservoir+'_nyc'):
#     if name in ('max_flow_ffmp_delivery_nj', 'drought_factor_delivery_nyc', 'delivery_nj', 'demand_nj'):
#     # if 'factor_trenton' in name or 'outflow_trenton' in name or 'target_trenton' in name: # or name == 'max_flow_delivery_nyc':
#         fig, (ax1, ax2) = plt.subplots(
#             figsize=(12, 4), ncols=2, sharey="row", gridspec_kw={"width_ratios": [3, 1]}
#         )
#         df.plot(ax=ax1)
#         df.quantile(np.linspace(0, 1)).plot(ax=ax2)
#
#         if name.startswith("reservoir"):
#             ax1.set_ylabel("Volume [MG]")
#         else:
#             ax1.set_ylabel("Flow [MGD]")
#
#         for ax in (ax1, ax2):
#             ax.set_title(name)
#             ax.grid(True)
#         plt.tight_layout()
#         print(name, df.min(), df.max())
#         if ext is not None:
#             fig.savefig(f"figs/{name}.{ext}", dpi=300)
