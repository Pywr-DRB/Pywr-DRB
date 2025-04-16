from pywr.core import Node, Parameter
from pywr.recorders import Recorder
from pywr.recorders import NumpyArrayNodeRecorder, NumpyArrayParameterRecorder
from pywr.recorders import TablesRecorder # Just to put here in case user wants to use it
import h5py 

__all__ = ["OutputRecorder"]

class OutputRecorder(Recorder):
    """
    A custom implementation of the pywr.Recorder class 
    that records model parameter and node values during simulation
    and exports the data to an HDF5 file at the end.
    
    Uses the NumpyArrayNodeRecorder and NumpyArrayParameterRecorder classes
    to record data for each parameter and node, which is efficient.
    
    Manually adds a "time" dataset to the HDF5 file to store the datetime index,
    which is expected to be available in the pywrdrb loading methods.
    
    
    Args:
    - model: The Pywr model instance.
    - output_filename: The output filename to write to.
    - parameters: A list of parameters to record. If None, all parameters with names will be recorded.
    - nodes: A list of nodes to record. If None, all nodes with names will be recorded.
    - kwargs: Additional keyword arguments to be passed to the pywr.Recorder class.
    
    Returns:
    - None
    """
    def __init__(self, 
                 model, 
                 output_filename, 
                 nodes=None,
                 parameters=None, 
                 **kwargs):
        super().__init__(model, **kwargs)
        
        self.output_filename = output_filename
        self.parameters = parameters
        self.nodes = nodes
        
        if parameters is None:
            self.parameters = self._get_model_parameter_names()
        
        if nodes is None:
            self.nodes = self._get_model_node_names()
            
        self.recorder_dict = {}
        for p in self.parameters:
            self.recorder_dict[p.name] = NumpyArrayParameterRecorder(model, p)
        for n in self.nodes:
            self.recorder_dict[n.name] = NumpyArrayNodeRecorder(model, n)

    def _get_model_node_names(self):
        """
        Get a list of all pywr.core.Node objects which have names in the model instance.
        Each node object has value attributes that are recorded during simulation.

        Returns:
            list[pywr.core.Node]: A list of pywr.core.Node objects which have names in the model instance.
        """
        node_names = [n for n in self.model.nodes.values() if n.name]
        return node_names
    
    def _get_model_parameter_names(self):
        """
        Get a list of all pywr.core.Parameter objects which have names in the model instance.
        Each parameter value will be recorded during simulation.

        Returns:
            list[pywr.core.Parameters]: A list of pywr.core.Parameter objects which have names in the model instance.
        """
        parameter_names = [p for p in self.model.parameters if p.name]
        return parameter_names

    def setup(self):
        """Sets up the recorders."""
        for recorder in self.recorder_dict.values():
            recorder.setup()
        
    def reset(self):
        """Reset the recorders."""
        for recorder in self.recorder_dict.values():
            recorder.reset()

    def after(self):
        """Performed after each timestep."""        
        for recorder in self.recorder_dict.values():
            recorder.after()
    
    def finish(self):
        """
        Saves data to an hdf5. 
        Performed at the end of the simulation.
        """
        self.to_hdf5()
        for recorder in self.recorder_dict.values():
            recorder.finish()

    # Create a new HDF5 file
    def to_hdf5(self):
        """
        Saves all data from the recorders to an HDF5 file.
        Manually adds a "time" dataset to the HDF5 file to store the datetime index.
        
        Returns:
            None
        """
        output_dict = {}
        
        # get and save datetime
        datetime = self.model.timestepper.datetime_index
        datetime = [str(d) for d in datetime]
        output_dict["time"] = datetime
        
        # loop through parameter recorders and save data
        for name, recorder in self.recorder_dict.items():
            #print(f"Data for {name}:")
            data = recorder.data
            output_dict[name] = data


        # Create a new HDF5 file
        with h5py.File(self.output_filename, 'w') as hdf:
            
            # Loop through each variable in the dictionary
            for name, data in output_dict.items():
        
                # Create a dataset for each variable
                try:
                    hdf.create_dataset(name, data=data)            
                except Exception as e:
                    print(f"OutputRecorder.to_hdf5() error during create_dataset() with {var} and {idx}")
                    print(e)
                    
OutputRecorder.register()