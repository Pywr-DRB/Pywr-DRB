import cProfile
import pstats
import io
import pywrdrb
import os
from datetime import datetime

import logging
logging.basicConfig(level=logging.WARNING)


if __name__ == "__main__":
    
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define working directory and output files
    wd = f"{os.path.dirname(os.path.abspath(__file__))}/../temp_results/profiling/"

    profile_stats_file = os.path.join(wd, f"profile_stats_{timestamp}.prof")
    profile_output_file = os.path.join(wd, f"detailed_profile_report_{timestamp}.txt")
    
    inflow_type = 'nwmv21_withObsScaled'
    
    # Create and build model (outside profiling for cleaner results)
    print("Building model...")
    mb = pywrdrb.ModelBuilder(
        inflow_type=inflow_type,
        start_date="1983-10-01", 
        end_date="2016-12-31"
    )
    mb.make_model()
    
    # Save model configuration
    model_filename = os.path.join(wd, f"{inflow_type}_{timestamp}.json")
    mb.write_model(model_filename)
    
    # Load model and set up recorder
    print("Loading model and setting up recorder...")
    model = pywrdrb.Model.load(model_filename)
    output_filename = os.path.join(wd, f"{inflow_type}_{timestamp}.hdf5")
    recorder = pywrdrb.OutputRecorder(
        model=model,
        output_filename=output_filename,
    )

    def run_simulation():
        """Function to profile - contains only the simulation run"""
        return model.run()

    # Run detailed profiling
    print(f"Starting detailed profiling...")
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run the simulation
    stats = run_simulation()
    
    profiler.disable()
    
    # Save raw profile data
    profiler.dump_stats(profile_stats_file)
    print(f"Raw profile data saved to: {profile_stats_file}")
    
    # Generate comprehensive text report
    print(f"Generating detailed report...")
    
    with open(profile_output_file, 'w') as f:
        # Redirect output to both file and console
        pr_stats = pstats.Stats(profiler)
        
        # Header information
        f.write(f"Detailed Profile Report for pywrdrb Simulation\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Inflow Type: {inflow_type}\n")
        f.write(f"Model File: {model_filename}\n")
        f.write(f"Output File: {output_filename}\n")
        f.write("="*80 + "\n\n")
        
        # Capture stats output to string buffer
        s = io.StringIO()
        
        # 1. Top functions by cumulative time (most important bottlenecks)
        pr_stats.stream = s
        pr_stats.strip_dirs().sort_stats('cumulative').print_stats(50)
        
        f.write("TOP 50 FUNCTIONS BY CUMULATIVE TIME:\n")
        f.write("-" * 50 + "\n")
        f.write(s.getvalue())
        f.write("\n\n")
        
        # 2. Top functions by total time (CPU intensive functions)
        s = io.StringIO()
        pr_stats.stream = s
        pr_stats.strip_dirs().sort_stats('tottime').print_stats(50)
        
        f.write("TOP 50 FUNCTIONS BY TOTAL TIME:\n")
        f.write("-" * 50 + "\n")
        f.write(s.getvalue())
        f.write("\n\n")
        
        # 3. Functions called most frequently
        s = io.StringIO()
        pr_stats.stream = s
        pr_stats.strip_dirs().sort_stats('ncalls').print_stats(50)
        
        f.write("TOP 50 MOST FREQUENTLY CALLED FUNCTIONS:\n")
        f.write("-" * 50 + "\n")
        f.write(s.getvalue())
        f.write("\n\n")
        
        # 8. All functions with detailed call information (comprehensive view)
        s = io.StringIO()
        pr_stats.stream = s
        pr_stats.strip_dirs().sort_stats('cumulative').print_stats()
        
        f.write("COMPLETE FUNCTION LISTING (ALL FUNCTIONS):\n")
        f.write("-" * 50 + "\n")
        f.write(s.getvalue())
    
    print(f"Comprehensive profile report saved to: {profile_output_file}")
    
    # Also print summary to console
    print("\nSUMMARY - Top 15 functions by cumulative time:")
    print("-" * 60)
    pstats.Stats(profiler).strip_dirs().sort_stats('cumulative').print_stats(15)
    
    # Cleanup temporary files
    if os.path.exists(model_filename):
        os.remove(model_filename)
    if os.path.exists(output_filename):
        os.remove(output_filename)
    
    print(f"\nProfiling complete. Detailed results in: {profile_output_file}")