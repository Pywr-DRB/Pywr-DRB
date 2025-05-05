"""
Used to efficiently store and save pywrdrb simulation data. 

Overview: 
During simulation, 'Recorder' classes are used to store and finally save data.
This custom recorder is designed to be efficient (quick) while tracking all
the model parameters and nodes of interest. The data is saved to an HDF5 file. 

Technical Notes: 
- This custom recorder is based on the pywr.Recorder class, which is a base class for all recorders in Pywr.
- It uses the following predefined pywr recorders:
    - NumpyArrayNodeRecorder
    - NumpyArrayParameterRecorder
    - NumpyArrayStorageRecorder
- By default, if nodes=None and parameters=None, all nodes and parameters with names will be recorded.
- The recorder_dict dictionary is used to store the recorders for each parameter and node.
- Future modifications should consider additional efficiency gains.

Links: 
- NA
 
Change Log:
TJA, 2025-05-05, Add docstrings.
"""

from pywr.core import Node, Parameter
from pywr.recorders import Recorder
from pywr.recorders import NumpyArrayNodeRecorder, NumpyArrayParameterRecorder, NumpyArrayStorageRecorder
from pywr.recorders import TablesRecorder # Just to put here in case user wants to use it
import h5py 
from pywrdrb.utils.lists import reservoir_list

__all__ = ["OutputRecorder"]

class OutputRecorder(Recorder):
    """
    Used to record and save data from the pywr model simulation.
    
    This uses the pywr.Recorder class as a base class and is designed to be efficient (quick) while tracking all
    variables from the model. The data is saved to an HDF5 file.    
    Manually adds a "time" dataset to the HDF5 file to store the datetime index,
    which is expected to be available in the pywrdrb loading methods.
    
    Methods
    -------
    _get_model_node_names()
        Get a list of all pywr.core.Node objects from the model instance.
    _get_model_parameter_names()
        Get a list of all pywr.core.Parameter objects from the model instance.
    setup()
        Sets up the recorders.
    reset()
        Reset the recorders.
    after()
        Performed after each timestep.
    finish()
        Saves data to an HDF5 file.
    to_hdf5()
        Saves all data from the recorders to an HDF5 file. Used by finish().
    
    Attributes
    ----------
    model : pywrdrb.Model
        The pywrdrb model instance to record data from.
    output_filename : str
        The name of the output HDF5 file to save the data to, at the end of simulation.
    nodes : list[pywr.core.Node], optional
        A list of pywr.core.Node objects to record data from. If None, all nodes with names will be recorded.
    parameters : list[pywr.core.Parameter], optional
        A list of pywr.core.Parameter objects to record data from. If None, all parameters with names will be recorded.
    recorder_dict : dict
        A dictionary of recorders for each parameter and node.
    reservoir_list : list
        A list of reservoir names to determine when Storage recorder should be used.
    """
    def __init__(self, 
                 model, 
                 output_filename, 
                 nodes=None,
                 parameters=None, 
                 **kwargs):
        """
        Initialize the OutputRecorder.
        
        Different NumpyArray recorders are initialized for different variables in the model.
        
        Parameters
        ----------
        model : pywrdrb.Model
            The pywrdrb model instance to record data from.
        output_filename : str
            The name of the output HDF5 file to save the data to, at the end of simulation.
        nodes : list[pywr.core.Node], optional
            A list of pywr.core.Node objects to record data from. If None, all nodes with names will be recorded.
        parameters : list[pywr.core.Parameter], optional
            A list of pywr.core.Parameter objects to record data from. If None, all parameters with names will be recorded.
        **kwargs : optional
            Additional keyword arguments to pass to the base pywr.Recorders class (unused).
            
        Returns
        -------
        None        
        """
        
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
            
            # for reservoir nodes, use NumpyArrayStorageRecorder
            if n.name.split("_")[0] == "reservoir" and n.name.split("_")[1] in reservoir_list:
                self.recorder_dict[n.name] = NumpyArrayStorageRecorder(model, 
                                                                       n, 
                                                                       proportional=False)
            # for other nodes, use NumpyArrayNodeRecorder
            else:
                self.recorder_dict[n.name] = NumpyArrayNodeRecorder(model, n)

    def _get_model_node_names(self):
        """Get a list of all pywr.core.Node objects in the model instance.
        
        Each node object has value attributes that are recorded during simulation.

        Parameters
        ----------
        None


        Returns
        -------
        list[pywr.core.Node]: A list of pywr.core.Node objects which have names in the model instance.
        """
        node_names = [n for n in self.model.nodes.values() if n.name]
        return node_names
    
    def _get_model_parameter_names(self):
        """Get a list of all pywr.core.Parameter objects in the model instance.
        
        Each parameter value will be recorded during simulation.

        Parameters
        ----------
        None 
        
        Returns
        -------
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
        """Saves data to an hdf5. 
        
        This is automatically done by pywr at the end of the simulation.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        self.to_hdf5()
        for recorder in self.recorder_dict.values():
            recorder.finish()

    # Create a new HDF5 file
    def to_hdf5(self):
        """Saves all data from the recorders to an HDF5 file.
        
        Manually adds a "time" dataset to the HDF5 file to store the datetime index.
        
        Parameters
        ----------
        None
        
        Returns
        -------
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
                    print(f"OutputRecorder.to_hdf5() error during create_dataset() with {name} and data type {type(data)}")
                    print(e)


OutputRecorder.register()