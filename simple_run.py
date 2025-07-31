# =============================================================================
# Pywr-DRB Getting Started Guide: Build, Run, and Visualize a DRB Water System Model
# =============================================================================

# This guide walks new users through creating a Delaware River Basin water system model 
# using the `pywrdrb` package. We'll build the model, run a simulation, and 
# visualize key outputs like major streamflows and reservoir storage.

# -----------------------------------------------------------------------------
# 1. Import Required Packages
# -----------------------------------------------------------------------------

import pywrdrb                        # Pywr-DRB package 
import matplotlib.pyplot as plt       # Plotting package for visualizations
import os                             # Standard package to handle file paths

# -----------------------------------------------------------------------------
# 2. Define Working Directory
# -----------------------------------------------------------------------------

# Replace this path with your preferred working directory.
wd = r""  # Output files (model JSON, output HDF5) will be saved here.

# -----------------------------------------------------------------------------
# 3. Build the Pywr-DRB Model
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# 4. Load the Model and Attach an Output Recorder
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# 5. Run the Model
# -----------------------------------------------------------------------------

# Execute the simulation
stats = model.run()

# -----------------------------------------------------------------------------
# 6. Load Simulation Outputs
# -----------------------------------------------------------------------------

# Instantiate a Data object to access results
data = pywrdrb.Data()

# Load major flows and reservoir storage results from the HDF5 file
results_sets = ['major_flow', 'res_storage']
data.load_output(output_filenames=[output_filename], results_sets=results_sets)

# Extract the dataframes for plotting
df_major_flow = data.major_flow["my_model"][0]
df_res_storage = data.res_storage["my_model"][0]

# -----------------------------------------------------------------------------
# 7. Plot Major Streamflows
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(5, 4))
df_major_flow[['delMontague', 'delTrenton']].plot(ax=ax)
ax.set_ylabel("Streamflow (mgd)")
ax.set_xlabel("Date")
ax.set_title("Major Streamflow: Montague & Trenton")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 8. Plot Reservoir Storage
# -----------------------------------------------------------------------------

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