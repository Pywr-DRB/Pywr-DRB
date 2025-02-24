#%%
from pprint import pprint
import pywrdrb
from pywrdrb import OutputRecorder

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

# data = pywrdrb.Data(print_status=True)

# datatypes = ['outputs']
# results_sets = ['major_flow', 'res_storage']


# data.load(datatypes=datatypes,
#           output_filenames= [output_filename], 
#           results_sets=results_sets)
