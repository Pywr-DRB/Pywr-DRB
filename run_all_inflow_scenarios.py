"""
Run Pywr-DRB simulation for all available inflow scenarios.

Builds and runs the model for each inflow type, saving outputs
to output_data/<inflow_type>.hdf5.
"""

import os
import pywrdrb
from pywrdrb.utils.dates import model_date_ranges

# Output directory
output_dir = "output_data"
os.makedirs(output_dir, exist_ok=True)

for inflow_type, (start_date, end_date) in model_date_ranges.items():
    print(f"\n{'='*60}")
    print(f"Running: {inflow_type} ({start_date} to {end_date})")
    print(f"{'='*60}")

    # Build model
    mb = pywrdrb.ModelBuilder(
        inflow_type=inflow_type,
        start_date=start_date,
        end_date=end_date,
    )
    mb.make_model()

    # Write model JSON (temporary)
    model_file = os.path.join(output_dir, f"{inflow_type}_model.json")
    mb.write_model(model_file)

    # Load and run
    model = pywrdrb.Model.load(model_file)
    output_file = os.path.join(output_dir, f"{inflow_type}.hdf5")
    recorder = pywrdrb.OutputRecorder(
        model=model,
        output_filename=output_file,
        parameters=[p for p in model.parameters if p.name],
    )
    stats = model.run()
    print(f"Finished: {inflow_type} -> {output_file}")

    # Clean up temporary model JSON
    os.remove(model_file)

print(f"\nAll scenarios complete. Outputs saved to {output_dir}/")
