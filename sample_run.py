import h5py
import pywrdrb

mb = pywrdrb.ModelBuilder(
    inflow_type='nhmv10_withObsScaled', 
    start_date="1983-10-01",
    end_date="2016-12-31",
    options={
        "inflow_ensemble_indices": None,
        "use_hist_NycNjDeliveries": True,  
        "predict_temperature":False, 
        "temperature_torch_seed": 4,
        "predict_salinity":False, 
        "salinity_torch_seed": 4,
        "run_starfit_sensitivity_analysis": False,
        "sensitivity_analysis_scenarios": [],
        })

mb.make_model()


model_filename = r"C:\Users\CL\Desktop\wd\test.json"
mb.write_model(model_filename)


#timer.start()
### Load the model
model = pywrdrb.Model.load(model_filename)


### Add a storage recorder
output_filename = r"C:\Users\CL\Desktop\wd\test.hdf5"
pywrdrb.TablesRecorder(
    model, output_filename, parameters=[p for p in model.parameters if p.name]
)


### Run the model
stats = model.run()

def hdf5_to_dict(file_path):
    def recursive_dict(group):
        d = {}
        for key, item in group.items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                d[key] = item[()]
            elif isinstance(item, h5py._hl.group.Group):
                d[key] = recursive_dict(item)
        return d

    with h5py.File(file_path, 'r') as f:
        data_dict = recursive_dict(f)

    return data_dict
out = hdf5_to_dict(output_filename)
