import pywrdrb
import time
import tracemalloc
import os
import csv
from datetime import datetime

# Suppress FutureWarnings and DeprecationWarnings
import warnings
from tables import NaturalNameWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=NaturalNameWarning)

# Setup
pn = pywrdrb.get_pn_object()

# Move up three levels
wd = pn.get().parents[2] / "temp_results/profiling"

# Create the "temp/profiling" directory
wd.mkdir(parents=True, exist_ok=True)
print(f"Working directory created: {wd}")

inflow_types = ['nhmv10_withObsScaled', 'nwmv21_withObsScaled', 'nhmv10', 'nwmv21', 'wrfaorc_withObsScaled', 'pub_nhmv10_BC_withObsScaled']  
recorder_types = ['TablesRecorder', 'PywrdrbRecorder']
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
results_file = os.path.join(wd, f'profiling_results_{timestamp}.csv')

# Write CSV header
with open(results_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "inflow_type",
        "recorder_type",
        "model_creation",
        "model_write",
        "model_load",
        "simulation",
        "peak_memory_MB"
    ])

def format_duration(seconds):
    minutes = int(seconds // 60)
    sec = seconds % 60
    return f"{minutes:02}:{sec:05.2f}"

# Loop over different inflow types
for inflow_type in inflow_types:
    for recorder_type in recorder_types:
        try:
            print(f"Profiling inflow_type = {inflow_type}; recorder_type = {recorder_type}")

            # --- Model Creation ---
            print("\tMaking model...")
            t0 = time.perf_counter()
            mb = pywrdrb.ModelBuilder(
                inflow_type=inflow_type,  
                start_date="2001-10-01", # Run 10 years for consistency.
                end_date="2010-9-30"
            )
            mb.make_model()
            t1 = time.perf_counter()

            model_filename = os.path.join(wd, f'model_{inflow_type}.json')
            mb.write_model(model_filename)
            t2 = time.perf_counter()

            # --- Model Loading ---
            print("\tLoading model...")
            model = pywrdrb.Model.load(model_filename)
            t3 = time.perf_counter()

            # --- Simulation with memory tracking ---
            print("\tAdding recorder...")
            output_filename = os.path.join(wd, f'model_output_{inflow_type}.hdf5')
            if recorder_type == 'NumpyArrayParameterRecorder':
                # New recorder type defualt in pywrdrb
                recorder = pywrdrb.OutputRecorder(
                    model=model,
                    output_filename=output_filename,
                    parameters=[p for p in model.parameters if p.name]
                    )
            elif recorder_type == 'TablesRecorder':
                pywrdrb.recorder.TablesRecorder(
                    model, output_filename, parameters=[p for p in model.parameters if p.name]
                )

            tracemalloc.start()
            t4 = time.perf_counter()
            print("\tRunning model...")
            stats = model.run()
            t5 = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Write results to CSV
            with open(results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    inflow_type,
                    recorder_type,
                    format_duration(t1 - t0),
                    format_duration(t2 - t1),
                    format_duration(t3 - t2),
                    format_duration(t5 - t4),
                    round(peak / (1024 ** 2), 2)  # Convert to MB
                ])
            print("\tCompleted successfully.")
        except Exception as e:
            print(f"Error with inflow_type = {inflow_type} and recorder_type = {recorder_type}: {e}")
            with open(results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    inflow_type,
                    recorder_type,
                    -9999,
                    -9999,
                    -9999,
                    -9999,
                    -9999  # Convert to MB
                ])
            

print(f"\nProfiling complete. Results saved to:\n{results_file}")

#%%
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
#results_file = os.path.join(wd, 'profiling_results_2025-05-12_1629.csv')
df = pd.read_csv(results_file)

def to_seconds(time_str):
    mins, secs = map(float, time_str.split(":"))
    return mins * 60 + secs

time_cols = ["model_creation", "model_write", "model_load", "simulation"]
for col in time_cols:
    df[col] = df[col].apply(to_seconds)

# Convert to long format
df_long = pd.melt(
    df,
    id_vars=["inflow_type", "recorder_type"],
    value_vars=time_cols + ["peak_memory_MB"],
    var_name="metric",
    value_name="Sec"
)

# Preview
# df_long.head()

# Separate memory and time data
df_time = df_long[df_long["metric"] != "peak_memory_MB"]

# Split simulation and non-simulation metrics
df_sim = df_time[df_time["metric"] == "simulation"]
df_others = df_time[df_time["metric"] != "simulation"]

# Combine to get consistent x positions
all_metrics = df_time["metric"].unique()
x_order = [m for m in ['model_creation', 'model_write', 'model_load', 'simulation'] if m in all_metrics]

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(5, 5))
ax2 = ax1.twinx()

# Plot non-simulation bars on ax1
sns.barplot(
    data=df_others,
    x="metric",
    y="Sec",
    hue="recorder_type",
    ax=ax1,
    order=x_order
)

# Plot simulation bars on ax2
sns.barplot(
    data=df_sim,
    x="metric",
    y="Sec",
    hue="recorder_type",
    ax=ax2,
    dodge=True,
    alpha=0.6,
    order=x_order,
    legend=False
)

# X-tick management
ax1.set_xticklabels(x_order, rotation=45)
ax1.set_xlabel("Step")
ax1.set_ylabel("Time (sec)")

ax1.axvline(2.5, c="gray", ls="--")

# Merge legends
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[:2], labels[:2], title="Recorder Type", loc='upper left')

# After plotting
sec_to_min = lambda sec: sec / 60
ymin, ymax = ax2.get_ylim()
ticks = ax2.get_yticks()

ax2.set_ylim(ymin, ymax)
ax2.set_yticks(ticks)
ax2.set_yticklabels([f"{tick/60:.1f}" for tick in ticks])
ax2.set_ylabel("10-year Simulation Time (min)")

plt.tight_layout()
plt.savefig(os.path.join(wd, f"TimeProfiling_{timestamp}.jpg"), dpi=300, bbox_inches='tight')
#plt.show()

# --- Boxplot for memory metric ---
# Filter memory data
df_memory = df_long[df_long["metric"] == "peak_memory_MB"]

# Create a new figure
plt.figure(figsize=(2, 5))
ax = sns.barplot(
    data=df_memory,
    x="metric",
    y="Sec",  # Assuming memory in MB is still stored in the 'Sec' column
    hue="recorder_type",
    dodge=True,
    legend=False
)

# Set labels and title
ax.set_xlabel("")
ax.set_ylabel("Peak Memory Usage During Simulation (MB)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(wd, f"PeakMemoryProfiling_{timestamp}.jpg"), dpi=300, bbox_inches='tight')
#plt.show()
