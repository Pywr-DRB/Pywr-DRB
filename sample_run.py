#%%
from pprint import pprint
import pywrdrb
from pywrdrb import OutputRecorder

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


wd = r"./"


# pprint(pywrdrb.get_directory())

###### Create a model ######
#Initialize a model builder
mb = pywrdrb.ModelBuilder(
    inflow_type='nhmv10_withObsScaled', 
    start_date="1983-10-01",
    end_date="2016-12-31"
    )

# Make a model
mb.make_model()

model_dict = mb.model_dict
#%%
# Output model.json file
model_filename = rf"{wd}\model.json"
mb.write_model(model_filename)
#%%

# ###### Run a simulation ######
# # Load the model using Model inherited from pywr
model = pywrdrb.Model.load(model_filename)

# Add a recorder inherited from pywr
output_filename = rf"{wd}\model_output.hdf5"

#%%


recorder = OutputRecorder(
    model, output_filename, 
    parameters=[p for p in model.parameters if p.name]
)

# Run a simulation
stats = model.run()


###### Post process ######
# Load simulation results

# Setup data loader object
data = pywrdrb.Data(print_status=True)

# specify the datatypes and results_sets to load
datatypes = ['outputs']
results_sets = ['major_flow', 'res_storage']

# Load the data
data.load(datatypes=datatypes,
          output_filenames= [output_filename], 
          results_sets=results_sets)
