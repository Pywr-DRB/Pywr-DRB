# Pywr-DRB: An open-source Python model for water availability and drought risk assessment in the Delaware River Basin

Pywr-DRB is an open-source Python model for exploring the role of reservoir operations, transbasin diversions, minimum flow targets, and other regulatory rules on water availability and drought risk in the DRB. Pywr-DRB is designed to flexibly draw on streamflow estimates from a variety of emerging data resources, such as the National Water Model, the National Hydrologic Model, and hybrid datasets blending modeled and observed data. Pywr-DRB bridges state-of-the-art advances in large-scale hydrologic modeling with an open-source representation of the significant role played by the basin's evolving water infrastructure and management institutions. 

For more details, see the following paper:

Hamilton, A.L., Amestoy, T.J., & P.M. Reed. (2024). Pywr-DRB: An open-source Python model for water availability and drought risk assessment in the Delaware River Basin. *(In Review)*.

## Setup

First clone `diagnostic_paper` branch of the Pywr-DRB repository from GitHub:

```bash
git clone -b diagnostic_paper https://github.com/Pywr-DRB/Pywr-DRB.git
```

Next, create and activate a Python virtual environment using your favorite package manager (pip or conda) and install the dependencies listed in ``requirements.txt``.

## Running Pywr-DRB model diagnostic experiment
To run the model diagnostic experiment from Hamilton et al. (2024) paper listed above, simply run the following script from the command line:

```bash
sh pywrdrb_run_diagnostics_paper.sh
```

This script has three main parts:

1. Run ``pywrdrb/prep_input_data.py``: this script runs various data preparation operations to prepare inputs needed by Pywr-DRB.
    a. Note: Some of the data used by prep_input_data.py is harvested and organized in a [separate Pywr-DRB repository](https://github.com/Pywr-DRB/Input-Data-Retrieval), as described in [this training notebook](https://github.com/Pywr-DRB/Pywr-DRB/blob/master/notebooks/Tutorial%2002%20Prepare%20Input%20Data.ipynb).
2. Run ``pywrdrb/run_historic_simulation.py`` for each of four inflow datasets: (NHM v1.0, NWM v2.1, hybrid NHM v1.0, hybrid NWM v2.1). This is the main simulation run.
    b. Note: More details on the simulation model can be found in Hamilton et al. (2024) as well as [this training notebook](https://github.com/Pywr-DRB/Pywr-DRB/blob/master/notebooks/Tutorial%2001%20Introduction%20to%20PywrDRB.ipynb)
3. Run ``pywrdrb/make_figs_diagnostics_paper.py``: this script runs postprocessing and creates all figures for Hamilton et al. (2024).

## Acknowledgements

This research was funded by the U.S. Geological Survey (USGS) Water Availability and Use Science Program as part of the Water Resources Mission Area Predictive Understanding of Multiscale Processes Project (USGS Grant Number G21AC10668). The authors thank Hedeff Essaid and Noah Knowles from USGS and Aubrey Dugger and David Yates from the National Center for Atmospheric Research (NCAR) for providing data and feedback that improved this work. The views expressed in this work are those of the authors and do not reflect the views or policies of the USGS or NCAR.
