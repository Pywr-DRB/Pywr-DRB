"""
This script defines a custom Pywr Parameter used to provide ensembles of 
flow timeseries to a given node.
"""

import numpy as np
import pandas as pd
import h5py

from pywr.parameters import Parameter, load_parameter, TablesArrayParameter

from utils.hdf5 import get_hdf5_realization_numbers, extract_realization_from_hdf5
from utils.directories import input_dir


class FlowEnsemble(Parameter):
    """This parameter provides access to inflow ensemble timeseries during the simulation period.

    Args:
        model (Model): The Pywr model.
        name (str): The node name.
        inflow_type (str): The dataset label; Options: 'obs_pub', 'nhmv10', 'nwmv21'
        inflow_ensemble_indices (list): Indices of the inflow ensemble to be used.
        **kwargs: Additional keyword arguments.

    Returns:
        None
    """    
    def __init__(self, model, name, inflow_type, inflow_ensemble_indices, **kwargs):
        super().__init__(model, **kwargs)
        
        filename = f'{input_dir}/historic_ensembles/catchment_inflow_{inflow_type}.hdf5'
        
        # Load from hfd5 specific realizations
        with h5py.File(filename, 'r') as file:
            node_inflow_ensemble = file[name]
            column_labels = node_inflow_ensemble.attrs['column_labels']
            
            # Get timeseries
            data = {}
            for label in column_labels:
                data[label] = node_inflow_ensemble[label][:]
            
            datetime = node_inflow_ensemble['date'][:].tolist()
            
        # Store in DF
        inflow_df = pd.DataFrame(data, index = datetime)
        inflow_df.index = pd.to_datetime(inflow_df.index.astype(str))
            
        ## Match ensemble indices to columns 
        # inflow_ensemble_indices is a list of integers; 
        # We need to 1) verify that the indices are included in the df
        # 2) find the columns corresponding to these realization IDs 
        inflow_ensemble_columns = []
        for real_id in inflow_ensemble_indices:
            assert(f'realization_{real_id}' in inflow_df.columns),f'The specified inflow_ensemble_index {real_id} is not available in the HDF file.'
            inflow_ensemble_columns.append(np.argwhere(inflow_df.columns == f'realization_{real_id}')[0][0])
            
        self.inflow_ensemble_indices = inflow_ensemble_indices
        self.inflow_column_indices = inflow_ensemble_columns
        self.inflow_ensemble = inflow_df.iloc[:, inflow_ensemble_columns]
        
        
    def setup(self):
        """Perform setup operations for the parameter."""
        super().setup()
        
    def value(self, timestep, scenario_index):
        """Return the current flow across scenarios for the specified timestep and scenario index.

        Args:
            timestep (Timestep): The timestep being evaluated.
            scenario_index (ScenarioIndex): The index of the scenario.

        Returns:
            float: The flow value for the specified timestep and scenario.
        """
        s_id = self.inflow_ensemble_indices[scenario_index.global_id]
        return self.inflow_ensemble.loc[timestep.datetime, f"realization_{s_id}"]

    @classmethod
    def load(cls, model, data):
        name = data.pop("node")
        inflow_ensemble_indices = data.pop("inflow_ensemble_indices")
        inflow_type = data.pop("inflow_type")
        return cls(model, name, inflow_type, inflow_ensemble_indices, **data)

FlowEnsemble.register()
