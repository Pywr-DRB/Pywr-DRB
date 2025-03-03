from pprint import pprint
import pywrdrb

#%%
pn = pywrdrb.get_pn_object()
pn_config = pywrdrb.get_pn_config()

pn_config = pywrdrb.get_pn_config()
pn_config["flows/my_data"] = pn_config["flows/nhmv10"]

pywrdrb.load_pn_config(pn_config)



#%%
# Now we can use the custom inflow type
mb = pywrdrb.ModelBuilder(
    inflow_type='my_data', 
    diversion_type='nhmv10',
    start_date="1983-10-01",
    end_date="1985-12-31"
    )

# Make a model (you are expected to see error here)
mb.make_model()

#%%
wd = r"C:\Users\CL\Desktop\wd"

# pprint(pywrdrb.get_directory())

###### Create a model ######
#Initialize a model builder
mb = pywrdrb.ModelBuilder(
    inflow_type='nhmv10_withObsScaled', 
    start_date="1983-10-01",
    end_date="1985-12-31"
    )

# Make a model
mb.make_model()

model_dict = mb.model_dict
#%%
# Output model.json file
model_filename = rf"{wd}\model.json"
mb.write_model(model_filename)


# ###### Run a simulation ######
# # Load the model using Model inherited from pywr
model_filename = rf"{wd}\model.json"
model = pywrdrb.Model.load(model_filename)

# Add a recorder inherited from pywr
output_filename = rf"{wd}\model_output_.hdf5"
pywrdrb.TablesRecorder(
    model, output_filename, parameters=[p for p in model.parameters if p.name]
)

# Run a simulation
stats = model.run()


###### Post process ######
# Load model_output.hdf5 and turn it into dictionary
output_dict = pywrdrb.hdf5_to_dict(output_filename)
