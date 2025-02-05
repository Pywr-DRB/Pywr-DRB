# Pywr-DRB: An open-source Python model for water availability and drought risk assessment in the Delaware River Basin

Pywr-DRB is an open-source Python model for exploring the role of reservoir operations, transbasin diversions, minimum flow targets, and other regulatory rules on water availability and drought risk in the DRB. Pywr-DRB is designed to flexibly draw on streamflow estimates from a variety of emerging data resources, such as the National Water Model, the National Hydrologic Model, and hybrid datasets blending modeled and observed data. Pywr-DRB bridges state-of-the-art advances in large-scale hydrologic modeling with an open-source representation of the significant role played by the basin's evolving water infrastructure and management institutions.

For more details, see the following paper:

Hamilton, A. L., Amestoy, T. J., & Reed, Patrick. M. (2024). Pywr-DRB: An open-source Python model for water availability and drought risk assessment in the Delaware River Basin. Environmental Modelling & Software, 106185. https://doi.org/10.1016/j.envsoft.2024.106185

## Installation

```bash
pip install git+https://github.com/Pywr-DRB/Pywr-DRB.git
```

## Getting start

### Create a minimum example
```python
import pywrdrb

###### Create a model ######
# Initialize a model builder
mb = pywrdrb.ModelBuilder(
    inflow_type='nhmv10_withObsScaled', 
    start_date="1983-10-01",
    end_date="1985-12-31"
)

# Make a model
mb.make_model()

# Output model.json file
model_filename = r"your working location\model.json"
mb.write_model(model_filename)


###### Run a simulation ######
# Load the model using Model inherited from pywr
model = pywrdrb.Model.load(model_filename)

# Add a recorder inherited from pywr
output_filename = r"your working location\model_output.hdf5"
pywrdrb.TablesRecorder(
    model, output_filename, parameters=[p for p in model.parameters if p.name]
)

# Run a simulation
stats = model.run()


###### Post process ######
# Load model_output.hdf5 and turn it into dictionary
output_dict = pywrdrb.hdf5_to_dict(output_filename)
```

## Advanced usage

### Customizing options
Default optional settings.

```python
# Print out the default optional settings
mb.options.list()
#NSCENARIOS: 1
#inflow_ensemble_indices: None
#use_hist_NycNjDeliveries: True
#predict_temperature: False
#temperature_torch_seed: 4
#predict_salinity: False
#salinity_torch_seed: 4
#run_starfit_sensitivity_analysis: False
#sensitivity_analysis_scenarios: []
#initial_volume_frac: 0.8
```

Customize optional settings
```python
mb = pywrdrb.ModelBuilder(
    inflow_type='nhmv10_withObsScaled', 
    start_date="1983-10-01",
    end_date="1985-12-31",
    options={
        "predict_temperature": True
    }
)
```

### Customizing directory
In pywrdrb, we use a global directory instance to store the directories. The default 
directories can be viewed by:

```python
mb.dirs.list()
# or
pywrdrb.get_directory().list()
```

For advanced usage, those directories can be assiged by the following:
```python
mb = pywrdrb.ModelBuilder(
    inflow_type='nhmv10_withObsScaled', 
    start_date="1983-10-01",
    end_date="1985-12-31",
    input_dir="new_input_dir"
)

# or 
mb.set_directory(input_dir="new_input_dir")

# or you may run the following code before using ModelBuilder
pywrdrb.set_directory(input_dir="new_input_dir")
```

## Acknowledgements

This research was funded by the U.S. Geological Survey (USGS) Water Availability and Use Science Program as part of the Water Resources Mission Area Predictive Understanding of Multiscale Processes Project (USGS Grant Number G21AC10668). The authors thank Hedeff Essaid and Noah Knowles from USGS and Aubrey Dugger and David Yates from the National Center for Atmospheric Research (NCAR) for providing data and feedback that improved this work. The views expressed in this work are those of the authors and do not reflect the views or policies of the USGS or NCAR.
