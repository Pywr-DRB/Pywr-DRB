[![DOI](https://zenodo.org/badge/479150651.svg)](https://doi.org/10.5281/zenodo.10720011)
[![Docs](https://github.com/Pywr-DRB/Pywr-DRB/actions/workflows/deploy.yml/badge.svg)](https://pywr-drb.github.io/Pywr-DRB/intro.html)
![Test](https://github.com/Pywr-DRB/Pywr-DRB/actions/workflows/run_tests.yml/badge.svg)

# Pywr-DRB: An open-source Python model for water availability and drought risk assessment in the Delaware River Basin

Pywr-DRB is an open-source Python model for exploring the role of reservoir operations, transbasin diversions, minimum flow targets, and other regulatory rules on water availability and drought risk in the DRB. Pywr-DRB is designed to flexibly draw on streamflow estimates from a variety of emerging data resources, such as the National Water Model, the National Hydrologic Model, and hybrid datasets blending modeled and observed data. Pywr-DRB bridges state-of-the-art advances in large-scale hydrologic modeling with an open-source representation of the significant role played by the basin's evolving water infrastructure and management institutions.

# Installation

```bash
pip install git+https://github.com/Pywr-DRB/Pywr-DRB.git
```

# A minimum example

## 1. Import Required Packages
```python
import pywrdrb                        # Pywr-DRB package 
import matplotlib.pyplot as plt       # Plotting package for visualizations
import os                             # Standard package to handle file paths
```

## 2. Define Working Directory
```python
# Replace this path with your preferred working directory.
wd = r""  # Output files (model JSON, output HDF5) will be saved here.
```

## 3. Build the Pywr-DRB Model
```python
# Create a ModelBuilder instance with inflow data type and time period
mb = pywrdrb.ModelBuilder(
    inflow_type='nwmv21_withObsScaled',  # Use hybrid version of NWM v2.1 inflow inputs
    start_date="1983-10-01",
    end_date="1985-12-31"
)

# Generate the model structure
mb.make_model()

# Save the model configuration to JSON
model_filename = os.path.join(wd, "my_model.json")
mb.write_model(model_filename)
```

## 4. Load the Model and Attach an Output Recorder
```python
# Load the model from the saved JSON file
model = pywrdrb.Model.load(model_filename)

# Define the HDF5 output file to store simulation results
output_filename = os.path.join(wd, "my_model.hdf5")

# Create an OutputRecorder to log all named parameters
recorder = pywrdrb.OutputRecorder(
    model=model,
    output_filename=output_filename,
    parameters=[p for p in model.parameters if p.name]
)
```

## 5. Run the Model
```python
# Execute the simulation
stats = model.run()
```

## 6. Load Simulation Outputs
```python
# Instantiate a Data object to access results
data = pywrdrb.Data()

# Load major flows and reservoir storage results from the HDF5 file
results_sets = ['major_flow', 'res_storage']
data.load_output(output_filenames=[output_filename], results_sets=results_sets)

# Extract the dataframes for plotting
df_major_flow = data.major_flow["my_model"][0]
df_res_storage = data.res_storage["my_model"][0]
```

## 7. Plot Major Streamflows
```python
fig, ax = plt.subplots(figsize=(5, 4))
df_major_flow[['delMontague', 'delTrenton']].plot(ax=ax)
ax.set_ylabel("Streamflow (mgd)")
ax.set_xlabel("Date")
ax.set_title("Major Streamflow: Montague & Trenton")
plt.tight_layout()
plt.show()
```
![](https://github.com/Pywr-DRB/Pywr-DRB/blob/master/docs/images/readme_streamflow.png)

## 8. Plot Reservoir Storage
```python
# Define reservoirs to plot
reservoirs = ['cannonsville', 'pepacton', 'neversink']

# Create subplots for each reservoir
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5, 5), sharex=True)
for ax, res in zip(axes, reservoirs):
    df_res_storage[res].plot(ax=ax)
    ax.set_ylabel(f"{res}\nstorage\n(mg)")
    ax.set_title(res.capitalize())

plt.xlabel("Date")
plt.suptitle("Reservoir Storage Over Time", y=0.96)
plt.tight_layout()
plt.show()
```
![](https://github.com/Pywr-DRB/Pywr-DRB/blob/master/docs/images/readme_storage.png)


# Documentation
[Tutorials](https://pywr-drb.github.io/Pywr-DRB/examples/examples.html) and [API references](https://pywr-drb.github.io/Pywr-DRB/api/api.html) can be found [![Docs](https://github.com/Pywr-DRB/Pywr-DRB/actions/workflows/deploy.yml/badge.svg)](https://pywr-drb.github.io/Pywr-DRB/intro.html) (click).

# DOIs & Citations


If you are using the package, we kindly ask that you acknowledge both the model and the associated publication.

1. Model: Lin, C.Y., Amestoy, T., Smith, M., Hamilton, A., & Reed, P. (2025). Pywr‑DRB v2.0.0 [Software]. Zenodo. https://doi.org/10.5281/zenodo.15659955

2. Paper: Hamilton, A. L., Amestoy, T. J., & Reed, Patrick. M. (2024). Pywr-DRB: An open-source Python model for water availability and drought risk assessment in the Delaware River Basin. Environmental Modelling & Software, 106185. https://doi.org/10.1016/j.envsoft.2024.106185

# Acknowledgements
This research was funded by the U.S. Geological Survey (USGS) as part of the Water Resources Mission area (USGS Grant Numbers G21AC10668, G23AC00678, and G24AC00124). The authors thank Hedeff Essaid and Noah Knowles from USGS and Aubrey Dugger and David Yates from the National Center for Atmospheric Research (NCAR) for providing data and feedback that improved this work. The views expressed in this work are those of the authors and do not reflect the views or policies of the USGS or NCAR.

