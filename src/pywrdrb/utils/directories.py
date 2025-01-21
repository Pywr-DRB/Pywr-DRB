"""
This script is used to identify the absolute path to the pywrdrb project directory.

This is used to ensure stability in relative paths throughout the project, regardless of the
current working directory in which a given script is run.
"""

####!!!!! Should be removed from the project !!!!!####
# See the snippet from src/pywrdrb/__init__.py for a better approach

import os

#### OLD CODE ####

# Absolute directory to the pywrdrb folder
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))

input_dir = os.path.realpath(os.path.join(ROOT_DIR, "../input_data/")) + "/"
output_dir = os.path.realpath(os.path.join(ROOT_DIR, "../output_data/")) + "/"

# lots of scripts use this
fig_dir = os.path.realpath(os.path.join(ROOT_DIR, "../figs/")) + "/"

# Only used in the 
# input_data/historical_reservoir_opt/plot_modified_starfit_comparison.py 
# mb, ffmp, starfit
# plotting/ensemble_plots.py, plotting_functions.py .....
model_data_dir = os.path.realpath(os.path.join(ROOT_DIR, "model_data/")) + "/"

# Only used in the pre/disaggregate_DRBC_demands.py, plotting/plot_flow_contribution.py,
# and plotting.plotting_functions.py script
spatial_data_dir = os.path.realpath(os.path.join(ROOT_DIR, "../DRB_spatial/")) + "/"

# Only used in the pre/prep_input_data_functions.py script
weap_dir = os.path.realpath(
    os.path.join(ROOT_DIR, "../input_data/WEAP_29June2023_gridmet/")
)
