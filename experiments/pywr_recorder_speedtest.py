from pprint import pprint
import pywrdrb
import time
import tempfile
import shutil
import os
from tqdm import tqdm

# Create a temporary directory
temp_dir = tempfile.mkdtemp(dir=os.path.dirname(__file__))
wd = temp_dir
#wd = r"C:\Users\CL\Desktop\wd"

# pprint(pywrdrb.get_directory())
timer = {}
inflow_types = ["nhmv10", "nwmv21", "nhmv10_withObsScaled", "nwmv21_withObsScaled"]

try:
    # =============================================================================
    # Create model files
    # =============================================================================
    for inflow_type in tqdm(inflow_types, desc="Creating model files"):
        timer[inflow_type] = {}
        #inflow_type = "nhmv10_withObsScaled"
        
        ###### Create a model ######
        #Initialize a model builder
        mb = pywrdrb.ModelBuilder(
            inflow_type=inflow_type, 
            start_date="1983-10-01",
            end_date="2016-12-31"
            )
        
        # Make a model
        mb.make_model()
        
        # Output model.json file
        model_filename = rf"{wd}\{inflow_type}.json"
        mb.write_model(model_filename)

    # =============================================================================
    # Run a simulation
    # =============================================================================

    # Run a simulation with NumpyArrayParameterRecorder
    for inflow_type in tqdm(inflow_types, desc="NumpyArrayParameterRecorder"):
        start_time = time.time()
        model_filename = rf"{wd}\{inflow_type}.json"
        model = pywrdrb.Model.load(model_filename)
        
        recorder_dict = {}
        for p in model.parameters:
            if p.name:
                recorder_dict[p.name] = pywrdrb.NumpyArrayParameterRecorder(model, p)
        
        # Run a simulation
        stats = model.run()
        
        pywrdrb.dict_to_hdf5(recorder_dict, rf"{wd}\NumpyArrayParameterRecorder_{inflow_type}.h5")
        #output_dict = pywrdrb.hdf5_to_dict(rf"{wd}\NumpyArrayParameterRecorder_{inflow_type}.h5")

        end_time = time.time()
        timer[inflow_type]['NumpyArrayParameterRecorder'] = end_time - start_time
    
    # Run a simulation with TablesRecorder
    for inflow_type in tqdm(inflow_types, desc="TablesRecorder"):
        start_time = time.time()
        model_filename = rf"{wd}\{inflow_type}.json"
        model = pywrdrb.Model.load(model_filename)
        
        output_filename = rf"{wd}\{inflow_type}.h5"
        pywrdrb.TablesRecorder(
            model, output_filename, parameters=[p for p in model.parameters if p.name]
        )
        
        # Run a simulation
        stats = model.run()
        
        end_time = time.time()
        timer[inflow_type]['TablesRecorder'] = end_time - start_time
finally:
    pprint(timer)
    # Delete the temporary directory
    shutil.rmtree(temp_dir)

r"""
{'nhmv10': {'NumpyArrayParameterRecorder': 59.57,
            'TablesRecorder': 565.73 = 9.43 mins},
 'nhmv10_withObsScaled': {'NumpyArrayParameterRecorder': 84.91,
                          'TablesRecorder': 671.22 = 11.19 mins},
 'nwmv21': {'NumpyArrayParameterRecorder': 58.31,
            'TablesRecorder': 450.57 = 7.51 mins},
 'nwmv21_withObsScaled': {'NumpyArrayParameterRecorder': 80.88,
                          'TablesRecorder': 668.77 = 11.15 mins}}
"""



