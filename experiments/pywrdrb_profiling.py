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
recorder_types = ['TablesRecorder', 'NumpyArrayParameterRecorder']
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
results_file = os.path.join(wd, f'profiling_results_{timestamp}.csv')

# Write CSV header
with open(results_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "inflow_type",
        "recorder_type"
        "model_creation_time_sec",
        "model_write_time_sec",
        "model_load_time_sec",
        "simulation_time_sec",
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
import pywrdrb

pn = pywrdrb.get_pn_object()

# Make/Set the working directory
pn.mkdir("temp/profiling")

wd = pn.get("temp/profiling")


def main():

    ###### Create a model ######
    # Initialize a model builder
    mb = pywrdrb.ModelBuilder(
        inflow_type='nwmv21_withObsScaled',  # Use hybrid version of NWM v2.1 inflow inputs
        start_date="1983-10-01",
        end_date="1985-12-31"
    )
    
    # # Make a model
    mb.make_model()
    
    # Output model.json file
    model_filename = rf"{wd}\model.json"
    mb.write_model(model_filename)
    
    
    ###### Run a simulation ######
    # Load the model using Model inherited from pywr
    model_filename = rf"{wd}\model.json"
    model = pywrdrb.Model.load(model_filename)
    
    # Add a recorder inherited from pywr
    output_filename = rf"{wd}\model_output.hdf5"
    pywrdrb.TablesRecorder(
        model, output_filename, parameters=[p for p in model.parameters if p.name]
    )
    
    # Run a simulation
    stats = model.run()
    
    
    ###### Post process ######
    # Load model_output.hdf5 and turn it into dictionary
    output_dict = pywrdrb.hdf5_to_dict(output_filename)


# Profile when running the script directly
if __name__ == "__main__":
    import cProfile
    import pstats

    # Profile the `main()` function
    cProfile.run('main()', 'profiling_output')

    # Save and display profiling report
    with open(rf'{wd}\profiling_report.txt', 'w') as f:
        p = pstats.Stats('profiling_output', stream=f)
        p.sort_stats('cumulative').print_stats()
    
    print(rf"Profiling complete. Report saved to {wd}\profiling_report.txt.")