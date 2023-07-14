import h5py
import sys
import os
import glob

from utils.directories import output_dir, model_data_dir, input_dir
from utils.hdf5 import get_hdf5_realization_numbers, combine_batched_hdf5_outputs

### specify inflow type from command line args
inflow_type_options = ['obs_pub', 'nhmv10', 'nwmv21', 'WEAP_29June2023_gridmet',
                       'obs_pub_nhmv10_NYCScaled', 'obs_pub_nwmv21_NYCScaled',
                       'obs_pub_nhmv10_NYCScaled_ensemble', 'obs_pub_nwmv21_NYCScaled_ensemble']
inflow_type = sys.argv[1]
assert(inflow_type in inflow_type_options), f'Invalid inflow_type specified. Options: {inflow_type_options}'


# Combine outputs into single HDF5
print(f'Combining all ensemble results files to single HDF5 file.')
batched_filenames = glob.glob(f'{output_dir}drb_output_{inflow_type}_rank*_batch*.hdf5')
# print(batched_filenames)
combined_output_filename = f'{output_dir}drb_output_{inflow_type}.hdf5'
combine_batched_hdf5_outputs(batch_files=batched_filenames, combined_output_file=combined_output_filename)

# Delete batched files
print('Deleting individual batch results files')
for file in batched_filenames:
    os.remove(file)