# Pywr-DRB: a Pywr model for Delaware River Basin

***In development***

This is the repository for the pywr implementation of a Delaware River Basin (DRB) water management model. Better documentation will be forthcoming, but for now here are the basic steps to run the model and analysis.

### Python installation
You will need Python 3 - I am running 3.8.10 and haven't done extensive compatability testing with other versions, though I expect it should be broadly stable for 3.6+ and perhaps earlier. You will also need to install numpy, matplotlib, click, and pywr packages. Additionally, to run the geospatial analysis, you will need to install gdal on your system followed by the geopandas, fiona, shapely, and contextily packages.

### Geospatial analysis
The ``DRB_spatial/`` directory contains the Jupyter Notebook ``DRB_spatial.ipynb`` that creates some (rudimentary for now) maps of the system, which can be helpful for visualizing the node network used in the pywr simulation. To recreate the maps, you will need to download additional geospatial data, as outlined in the separate README file in that directory.

### Prepping data, looping over simulations with different input files, and analyzing data
Running ``drb_run_all.sh`` from the command line will do three things. First, it will run ``prep_input_data.py``, which preps input data files on observed streamflows, modeled flows from NHMv1.0, NWMv2.1, and WEAP (Aug 23, 2022, version), and saves data in Pywr-ready formats. Second, it loops over these four input data types and runs each through the Pywr model, using ``drb_run_sim.py``. Lastly, it analyzes the results and creates the figures from my 10/24/2023 USGS seminar plus some other close variants.

### More info forthcoming...

