import pywrdrb
wd = r"C:\Users\CL\Desktop\wd"

def main():

    ###### Create a model ######
    # Initialize a model builder
    # mb = pywrdrb.ModelBuilder(
    #     inflow_type='nhmv10_withObsScaled', 
    #     start_date="1983-10-01",
    #     end_date="2010-12-31"
    #     )
    
    # # Make a model
    # mb.make_model()
    
    # # Output model.json file
    # model_filename = rf"{wd}\model.json"
    # mb.write_model(model_filename)
    
    
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