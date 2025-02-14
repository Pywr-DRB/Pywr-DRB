from pywr.recorders import Recorder, NumpyArrayParameterRecorder, TablesRecorder
import h5py 

class OutputRecorder(Recorder):
    """
    Records the output of the model to a file.
    
    Args:
    - model: The Pywr model dict.
    - output_filename: The output filename to write to.
    - parameters: A list of parameters to record.
    
    """
    def __init__(self, 
                 model, 
                 output_filename, 
                 parameters, 
                 **kwargs):
        super().__init__(model, **kwargs)
        
        self.output_filename = output_filename
        self.parameters = parameters
        self.saved_parameters = []
        
        self.recorder_dict = {}
        for p in self.parameters:
            if p.name:
                self.recorder_dict[p.name] = NumpyArrayParameterRecorder(model, p)
                self.saved_parameters.append(p)

    def setup(self):
        """Sets up the recorder."""
        for recorder in self.recorder_dict.values():
            recorder.setup()
        
    def reset(self):
        """Resets the recorder."""
        for recorder in self.recorder_dict.values():
            recorder.reset()




    # Create a new HDF5 file
    def to_hdf5(self):
        output_dict = {}
        datetime = None
        for name, recorder in self.recorder_dict.items():
            #print(f"Data for {name}:")
            df = recorder.to_dataframe()
            datetime = df.index.values

            d = df.reset_index(drop=True)
            d.columns = [int(col) for col in d.columns.get_level_values(0)]    
            d = d.to_dict(orient="list")
            
            output_dict[name] = d

            # add time
            output_dict["time"] = datetime

        # Create a new HDF5 file
        with h5py.File(self.output_filename, 'w') as hdf:
            # Loop through each variable in the dictionary
            for var, indices in output_dict.items():
                # Create a group for each variable
                group = hdf.create_group(var)
                # Loop through each index in the variable
                for idx, values in indices.items():
                    # Convert index to string to use as dataset name
                    dataset_name = str(idx)
                    # Create a dataset for each index
                    group.create_dataset(dataset_name, data=values)