"""
This script is used to identify the absolute path to the pywrdrb project directory.

This is used to ensure stability in relative paths throughout the project, regardless of the 
current working directory in which a given script is run.
"""

import os

# Absolute directory to the pywrdrb folder
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '../'))

input_dir = os.path.realpath(os.path.join(ROOT_DIR, '../input_data/')) + '/'
output_dir = os.path.realpath(os.path.join(ROOT_DIR, '../output_data/')) + '/'
fig_dir = os.path.realpath(os.path.join(ROOT_DIR, '../figs/')) + '/'
model_data_dir = os.path.realpath(os.path.join(ROOT_DIR, 'model_data/')) + '/'
