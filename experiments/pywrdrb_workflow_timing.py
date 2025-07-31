"""
This script is used to get simple timing info for each step in the simulation workflow:

1. pywrdrb.ModelBuilder.make_model()
2. pywrdrb.ModelBuilder.write_model()
3. pywrdrb.Model.load()
4. Setup pywrdrb.OutputRecorder
5. pywrdrb.Model.run()

"""
import pywrdrb
import os
import time
wd = "." 

if __name__ == "__main__":


    # Settings
    inflow_type = 'nwmv21_withObsScaled'  # Use hybrid version of NWM v2.1 inflow inputs
    start_date = "1983-10-01"
    end_date = "2016-12-31" 
    model_filename = os.path.join(wd, "my_model.json")
    output_filename = os.path.join(wd, "my_model.hdf5")

    print(f"### Beginning timing test for pywrdrb with {inflow_type} inflow | {start_date} - {end_date}###")


    mb_start = time.time()
    # Create a ModelBuilder instance with inflow data type and time period
    mb = pywrdrb.ModelBuilder(
        inflow_type=inflow_type,
        start_date=start_date,
        end_date=end_date
    )

    mb.make_model()
    mb_end = time.time()

    # write to json
    mbw_start = time.time()
    mb.write_model(model_filename)
    mbw_end = time.time()

    # Load the model from the saved JSON file
    ml_start = time.time()
    model = pywrdrb.Model.load(model_filename)
    ml_end = time.time()

    # Setup output recorder
    mor_start = time.time()
    recorder = pywrdrb.OutputRecorder(
        model=model,
        output_filename=output_filename,
    )
    mor_end = time.time()

    # Execute the simulation
    mr_start = time.time()
    stats = model.run()
    mr_end = time.time()

    # Print summary of timing results
    print(f"### Timing Results for pywrdrb with {inflow_type} inflow | {start_date} - {end_date}###")
    print(f"1. ModelBuilder.make_model() took {mb_end - mb_start:.4f} seconds")
    print(f"2. ModelBuilder.write_model() took {mbw_end - mbw_start:.4f} seconds")
    print(f"3. Model.load() took {ml_end - ml_start:.4f} seconds")
    print(f"4. OutputRecorder setup took {mor_end - mor_start:.4f} seconds")
    print(f"5. Model.run() took {mr_end - mr_start:.4f} seconds")

