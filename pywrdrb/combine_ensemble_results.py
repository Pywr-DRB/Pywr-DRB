import sys
import os
import glob
import warnings

from utils.directories import output_dir, model_data_dir
from utils.hdf5 import combine_batched_hdf5_outputs
from utils.options import inflow_type_options

### specify inflow type from command line args
inflow_type_options = [
    "obs_pub",
    "nhmv10",
    "nwmv21",
    "WEAP_29June2023_gridmet",
    "obs_pub_nhmv10_ObsScaled",
    "obs_pub_nwmv21_ObsScaled",
    "obs_pub_nhmv10_ObsScaled_ensemble",
    "obs_pub_nwmv21_ObsScaled_ensemble",
    "syn_obs_pub_nhmv10_ObsScaled_ensemble",
    "syn_obs_pub_nwmv21_ObsScaled_ensemble",
]
use_mpi_options = [None, "", "True", "False"]

inflow_type = sys.argv[1]
assert (
    inflow_type in inflow_type_options
), f"Invalid inflow_type specified. Options: {inflow_type_options}"


if len(sys.argv) > 2:
    use_mpi = sys.argv[2]
    assert (
        use_mpi in use_mpi_options
    ), f"Invalid use_mpi specified. Options: {use_mpi_options}"
    if use_mpi == "True":
        use_mpi = True
    else:
        use_mpi = False

if use_mpi:
    batched_filenames = glob.glob(
        f"{output_dir}drb_output_{inflow_type}_rank*_batch*.hdf5"
    )
    batched_modelnames = glob.glob(
        f"{model_data_dir}drb_model_full_{inflow_type}_rank*.json"
    )
else:
    batched_filenames = glob.glob(f"{output_dir}drb_output_{inflow_type}_batch*.hdf5")
    batched_modelnames = []


### Combine batched output files
try:
    combined_output_filename = f"{output_dir}drb_output_{inflow_type}.hdf5"
    combine_batched_hdf5_outputs(
        batch_files=batched_filenames, combined_output_file=combined_output_filename
    )

    # Delete batched files
    print("Deleting individual batch results files")
    for file in batched_filenames:
        os.remove(file)
    for file in batched_modelnames:
        os.remove(file)
except Exception as e:
    warnings.warn(f"Error combining batched files: {e}")
    pass
