from pywr.model import Model
from pywr.recorders import TablesRecorder
from drb_make_model import drb_make_model
from custom_pywr import *
import sys

### specify inflow type from command line args
inflow_type = sys.argv[1]
backup_inflow_type = sys.argv[2]

# inflow_type = 'nhmv10'  ### nhmv10, nwmv21, nwmv21_withLakes, obs, obs_pub, WEAP_23Aug2022_gridmet
# backup_inflow_type = 'nhmv10'  ## for WEAP inflow type, we dont have all reservoirs. use this secondary type for missing.

model_filename = "model_data/drb_model_full.json"
if 'WEAP' in inflow_type:
    output_filename = f"output_data/drb_output_{inflow_type}_{backup_inflow_type}.hdf5"
else:
    output_filename = f"output_data/drb_output_{inflow_type}.hdf5"

### import custom pywr params - register the name so it can be loaded from JSON
FfmpNycRunningAvgParameter.register()
FfmpNjRunningAvgParameter.register()
VolBalanceNYCDemandTarget.register()
VolBalanceNYCDemandFinal.register()
VolBalanceNYCDownstreamMRFTargetAgg.register()
VolBalanceNYCDownstreamMRFTarget.register()
VolBalanceNYCDownstreamMRFFinal.register()
NYCCombinedReleaseFactor.register()

### make model json files
drb_make_model(inflow_type, backup_inflow_type)

### Load the model
model = Model.load(model_filename)

### Add a storage recorder
TablesRecorder(model, output_filename, parameters=[p for p in model.parameters if p.name])

### Run the model
stats = model.run()
stats_df = stats.to_dataframe()
