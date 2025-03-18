from pprint import pprint
import time
import tempfile
import shutil
import os

import pywrdrb
from pywrdrb.recorders import OutputRecorder

# Create a temporary directory
temp_dir = tempfile.mkdtemp(dir=os.path.dirname(__file__))
wd = temp_dir

timer = {}
inflow_types = ["nhmv10", "nwmv21", "nhmv10_withObsScaled", "nwmv21_withObsScaled"]

try:
    # =============================================================================
    # Create model files
    # =============================================================================
    for inflow_type in inflow_types:
        timer[inflow_type] = {}
        
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

    # Run a simulation with Custom OutputRecorder
    for inflow_type in inflow_types:
        start_time = time.time()
        model_filename = rf"{wd}\{inflow_type}.json"
        model = pywrdrb.Model.load(model_filename)
        
        output_filename = rf"{wd}\{inflow_type}.hdf5"
        recorder = OutputRecorder(
            model, output_filename, 
            parameters=[p for p in model.parameters if p.name]
        )
        # Run a simulation
        stats = model.run()
        

        end_time = time.time()
        timer[inflow_type]['Custom OutputRecorder'] = end_time - start_time
    
    # Run a simulation with TablesRecorder
    for inflow_type in inflow_types:
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
On Trevor's laptop, the output is:
{'nhmv10': {'Custom OutputRecorder': 35.02632451057434,
            'TablesRecorder': 156.96190094947815},
 'nhmv10_withObsScaled': {'Custom OutputRecorder': 34.599191427230835,
                          'TablesRecorder': 135.11984395980835},
 'nwmv21': {'Custom OutputRecorder': 34.69260334968567,
            'TablesRecorder': 147.52072262763977},
 'nwmv21_withObsScaled': {'Custom OutputRecorder': 34.78731966018677,
                          'TablesRecorder': 125.03494048118591}}
"""



