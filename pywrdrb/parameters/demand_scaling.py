"""
Withdrawl scaling factor
"""

import numpy as np
import pandas as pd
import h5py

from pywr.parameters import Parameter, load_parameter, TablesArrayParameter
from pywrdrb.utils.directories import input_dir


class NodeWithdrawlScalingFactor(Parameter):
    """
    Args:
        model (Model): The Pywr model.
        node (str): The node node.
        withdrawl_scaling_indices (list): List of indices for samples from the CSV.
        filename (str): The name of the CSV file containing scaling samples
        **kwargs: Additional keyword arguments.

    Returns:
        None
    """    
    def __init__(self, model, node, withdrawl_scaling_indices, filename, 
                 **kwargs):
        super().__init__(model, **kwargs)
        
        ### LOAD CSV WITH SAMPLES
        filename = pd.read_csv( f'{input_dir}/Node_Withdrawl_Scaling_Factor.csv', index_col=0)

        #Specifcy ensemble indicies 
        #withdrawl_scaling_indices = [1,2,3,4,5]
        
        
    def setup(self):
        """Perform setup operations for the parameter."""
        super().setup()
        
    def value(self, timestep, scenario_index):
        """

        Args:
            timestep (Timestep): The timestep being evaluated. (but there is no timestep???)
            scenario_index (ScenarioIndex): The index of the scenario.

        Returns:
            float: The withdrawal scaling factor value for the specified timestep, scenario, and node.
        """
        # pywr scenario index
        s_id = self.withdrawl_scaling_indices[scenario_index.global_id]

        # Load the CSV file to access the withdrawal scaling factors
        withdrawal_scaling_data = pd.read_csv(self.filename, index_col=0)

         # Get the withdrawal scaling factor for the specified scenario and node
        withdrawal_scaling_factor = withdrawal_scaling_data.loc[s_id, scenario_index.node]

        return withdrawal_scaling_factor

        ### RETURN NODE-Scenario specific value 

    @classmethod
    def load(cls, model, data):
        node = data.pop("node")
        withdrawl_scaling_indices = data.pop("withdrawal_scaling_indices")
        filename = data.pop("filename")
        return cls(model, node, withdrawl_scaling_indices, filename, **data)


NodeWithdrawlScalingFactor.register()