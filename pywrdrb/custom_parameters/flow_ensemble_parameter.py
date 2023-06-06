"""
This script defines a custom Pywr Parameter used to provide ensembles of 
flow timeseries to a given node.
"""

import numpy as np
import pandas as pd

from pywr.parameters import Parameter, load_parameter

from utils.directories import input_dir


class FlowEnsemble(Parameter):
    """Custom Pywr parameter class. This parameter provides access to inflow ensemble timeseries during the simulation period. 
    
    :param name: The node name.
    :type name: str
    
    :param inflow_type: The dataset label; Options: 'obs_pub', 'nhmv10', 'nwmv21'
    :type inflow_type: str
    
    :returns: None
    :rtype: None
    """
    
    def __init__(self, model, name, inflow_type, inflow_ensemble_indices, **kwargs):
        super().__init__(model, **kwargs)
        
        self.inflow_ensemble = pd.read_csv(f'{input_dir}/synthetic_ensembles/{inflow_type}/ensemble_inflow_{name}.csv', index_col=0, parse_dates=True)
        
        N_SCENARIOS = len(inflow_ensemble_indices)
        self.inflow_ensemble_indices = inflow_ensemble_indices
        self.inflow_ensemble = self.inflow_ensemble.iloc[:, inflow_ensemble_indices]
        
        
    def setup(self):
        ### allocate an array to hold the parameter state
        super().setup()
        
    def value(self, timestep, scenario_index):
        ### return the current flow across scenarios
        #if self.name == 'flow_cannonsville' and ((scenario_index.global_id == 1) or (scenario_index.global_id == 0)):
        #    print(f'Returning {self.inflow_ensemble.loc[timestep.datetime, f"scenario_{scenario_index.global_id}"]} for {self.name} in scenario {scenario_index.global_id} at {timestep.datetime}')
        s_id = self.inflow_ensemble_indices[scenario_index.global_id]
        return self.inflow_ensemble.loc[timestep.datetime, f"scenario_{s_id}"]

    @classmethod
    def load(cls, model, data):
        name = data.pop("node")
        inflow_ensemble_indices = data.pop("inflow_ensemble_indices")
        inflow_type = data.pop("inflow_type")
        return cls(model, name, inflow_type, inflow_ensemble_indices, **data)

FlowEnsemble.register()
