import pywrdrb

###### Create a model ######
# Initialize a model builder
mb = pywrdrb.ModelBuilder(
    inflow_type='nhmv10_withObsScaled', 
    start_date="1983-10-01",
    end_date="1985-12-31"
    )

# Make a model
mb.make_model()

# Output model.json file
model_filename = r"your working location\model.json"
mb.write_model(model_filename)


###### Run a simulation ######
# Load the model using Model inherited from pywr
model = pywrdrb.Model.load(model_filename)

# Add a recorder inherited from pywr
output_filename = r"your working location\model_output.hdf5"
pywrdrb.TablesRecorder(
    model, output_filename, parameters=[p for p in model.parameters if p.name]
)

# Run a simulation
stats = model.run()


###### Post process ######
# Load model_output.hdf5 and turn it into dictionary
output_dict = pywrdrb.hdf5_to_dict(output_filename)
