# Pywr model for Delaware River Basin

***In development***

This is the repository for the pywr implementation of a Delaware River Basin (DRB) water management model. Better documentation will be forthcoming, but for now here are the basic steps to run the model and analysis.

### Python installation
You will need Python 3 - I am running 3.8.10 and haven't done extensive compatability testing with other versions, though I expect it should be broadly stable for 3.6+ and perhaps earlier. You will also need to install numpy, matplotlib, click, and pywr packages. Additionally, to run the geospatial analysis, you will need to install gdal on your system followed by the geopandas, fiona, shapely, and contextily packages.

### Geospatial analysis
The ``DRB_spatial/`` directory contains the Jupyter Notebook ``DRB_spatial.ipynb`` that creates some (rudimentary for now) maps of the system, which can be helpful for visualizing the node network used in the pywr simulation. To recreate the maps, you will need to download additional geospatial data, as outlined in the separate README file in that directory.

### Downloading streamflow data
The ``get_usgs_inflows.py`` script downloads historical discharge timeseries for many streamflow gages associated with reservoirs and other important river reaches in the DRB. These are stored in the ``input_data/`` directory. *Note: this takes a while, USGS servers are slow.*

### Prepping streamflow data
The ``drb_inflow_prep.py`` script will loop through each input dataset, extract and clean up the desired information, and store all in a single file, ``input_data/inflows_clean.csv``. *Note: This is currently broken, need to go back and fix, but you can still use the inflows_clean.csv file. Also note that at present, I am just using constant streamflows for the 4 catchments in the simplified model, until I can verify that everything is working properly.*

### Creating the "model" file
Pywr builds the system model from a JSON file with a specific format. However, this format is rather difficult to read and develop for large, complex models. For this reason, the DRB model is mostly defined in an Excel file, ``model_data/drb_model_sheets.xlsx``, with different sheets for nodes, edges, parameters rules curves, etc. The ``drb_make_model.py`` script is used to extract this information and transform it into the necessary JSON format, ``model_data/drb_model_full.json``. 

This script also uses parametric information stored in the Excel file to build pywr representations of the STARFIT data-driven control rules for many reservoirs. See Turner et al, "Water storage and release policies for all large reservoirs of conterminous United States", Journal of Hydrology, 2021.

*Note: There seems to be an issue with the STARFIT implementation and/or unit inconsistencies, as I am getting strange results on our simple test problem at the moment.*

### Running the model
The ``drb.py`` script can be used to run the model by supplying the command line argument "run" (i.e. ``python3 drb.py run``). Results will be stored as ``output_data/drb_output.hdf5``.

The same script can also be used to visualize the output, using the "figures" command line argument. Note that the types of data to visualize are currently hard-coded in line 42. 


