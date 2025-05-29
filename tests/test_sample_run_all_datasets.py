"""
Runs the complete simulation workflow for all pre-packaged datasets.

This is a large integrated test.
"""

import pytest
import pywrdrb
from pywrdrb.utils.dates import model_date_ranges
from pywrdrb.path_manager import get_pn_object

pn = get_pn_object()
flowtype_opts = [i for i in pn.flows.list(type="folder") if i[0] != "_"]

def test_sample_run_all_datasets():
    """Run sample_run.py workflow for all datasets."""

    # Loop through available flowtypes
    for flowtype in flowtype_opts:
        
        
        model_filename = f"{flowtype}_model.json"
        output_filename = f"{flowtype}_output.hdf5"
        
        try:
            start, end = model_date_ranges[flowtype]
        except Exception as e:
            raise ValueError(f"Flowtype '{flowtype}' not found in model_date_ranges: {e}")
        
        # Make the model
        try:
            mb = pywrdrb.ModelBuilder(
                inflow_type=flowtype,
                start_date=start,
                end_date=start
            )
            mb.make_model()
            mb.write_model(model_filename)
        except Exception as e:
            raise RuntimeError(f"Failed to create model for flowtype '{flowtype}': {e}")

        # load model
        try:
            model = pywrdrb.Model.load(model_filename)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from '{model_filename}': {e}")
        
        # make output recorder
        try:
            recorder = pywrdrb.OutputRecorder(
                model=model,
                output_filename=output_filename,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create output recorder for model '{model_filename}': {e}")
        
        # run
        try:
            stats = model.run()
        except Exception as e:
            raise RuntimeError(f"Failed to run model '{model_filename}': {e}")
        
        # load output
        try:
            data = pywrdrb.Data()
            data.load_output(output_filenames=[output_filename])
        except Exception as e:
            raise RuntimeError(f"Failed to load output from '{output_filename}': {e}")
        
        print(f"Successfully ran full model workflow for flowtype '{flowtype}'.") 