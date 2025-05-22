import cProfile
import pstats
import pywrdrb
import os

import logging
logging.basicConfig(level=logging.INFO) # Disable debug messages

# Define working directory
wd = "../"


if __name__ == "__main__":

    inflow_type = 'nwmv21_withObsScaled'
    
    # Create and build model
    mb = pywrdrb.ModelBuilder(
        inflow_type=inflow_type,
        start_date="1983-10-01",
        end_date="2010-12-31"  # Using shorter period for profiling
    )
    mb.make_model()
    
    # Save model configuration
    model_filename = os.path.join(wd, f"{inflow_type}.json")
    mb.write_model(model_filename)
    
    # Load model and set up recorder
    model = pywrdrb.Model.load(model_filename)
    output_filename = os.path.join(wd, f"{inflow_type}.hdf5")
    recorder = pywrdrb.OutputRecorder(
        model=model,
        output_filename=output_filename,
    )

    def run_model():            
        # Run model
        stats = model.run()
        return model, stats


    # Run with profiling
    cProfile.run('run_model()', 'profile_stats')

    # Analyze results
    p = pstats.Stats('profile_stats')
    p.strip_dirs().sort_stats('cumulative').print_stats(30)