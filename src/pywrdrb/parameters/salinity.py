# Necessary evil for lstm to find files
import os
import numpy as np
import pandas as pd
from pywr.parameters import Parameter, load_parameter

from pywrdrb.utils.constants import cms_to_mgd
from pywrdrb.utils.dates import temp_pred_date_range
from pywrdrb.utils.directories import ROOT_DIR

class SalinityLSTM():
    def __init__(self, start_date, torch_seed):
       
        # creating an instance of an LSTM model
        #print("Creating an instance of an BMI_LSTM model object")
        self.lstm = None

    def forecast(self, n, flow_Trenton, flow_Schuylkill, timestep=None):     
        salt_front_river_mile = 95
        return salt_front_river_mile
    
    def predict(self, flow_Trenton, flow_Schuylkill, timestep=None): 
        salt_front_river_mile = 95
        return salt_front_river_mile

class SalinityModel(Parameter):
    def __init__(self, model, torch_seed, **kwargs):
        super().__init__(model, **kwargs)
        self.torch_seed = torch_seed
        self.salinity_model = SalinityLSTM(start_date=model.timestepper.start, torch_seed=torch_seed)

    def value(self, timestep, scenario_index):
        pass
        return -99
    
    @classmethod
    def load(cls, model, data):
        torch_seed = data.pop("torch_seed")
        return cls(model, torch_seed, **data)
SalinityModel.register()

class SaltFrontRiverMile(Parameter):
    def __init__(self, model, salinity_model, node_Trenton_flow, node_Schuylkill_flow, **kwargs):
        super().__init__(model, **kwargs)
        self.salinity_lstm = salinity_model
        self.node_Trenton_flow = node_Trenton_flow
        self.node_Schuylkill_flow = node_Schuylkill_flow

        self.children.add(salinity_model)
    
    def value(self, timestep, scenario_index):

        flow_Trenton = self.node_Trenton_flow.flow #or prev_flow
        flow_Schuylkill = self.node_Schuylkill_flow.flow

        salt_front_river_mile = self.salinity_lstm.salinity_model.predict(flow_Trenton, flow_Schuylkill, timestep)

        return salt_front_river_mile
    
    @classmethod
    def load(cls, model, data):

        node_Trenton_flow = model.nodes["link_delTrenton"]
        node_Schuylkill_flow = model.nodes["link_outletSchuylkill"]
        salinity_model = load_parameter(model, "salinity_model")
        return cls(model, salinity_model, node_Trenton_flow, node_Schuylkill_flow,  **data)
SaltFrontRiverMile.register()

class SaltFrontAdjustFactor(Parameter):
    def __init__(self, model, mrf, salt_front_river_mile, 
                 rm_factor_mrf_92_5,
                 rm_factor_mrf_87,
                 rm_factor_mrf_82_9,
                 rm_factor_mrf_below_82_9, 
                 level_index, **kwargs):
        super().__init__(model, **kwargs)
        self.mrf = mrf
        self.salt_front_river_mile = salt_front_river_mile
        self.rm_factor_mrf_92_5 = rm_factor_mrf_92_5
        self.rm_factor_mrf_87 = rm_factor_mrf_87
        self.rm_factor_mrf_82_9 = rm_factor_mrf_82_9
        self.rm_factor_mrf_below_82_9 = rm_factor_mrf_below_82_9
        self.level_index = level_index

        # To ensure cannonsville_release & pepacton_release are updated before this parameter
        self.children.add(salt_front_river_mile)
        self.children.add(rm_factor_mrf_92_5)
        self.children.add(rm_factor_mrf_87)
        self.children.add(rm_factor_mrf_82_9)
        self.children.add(rm_factor_mrf_below_82_9)
        self.children.add(level_index)
    
    def value(self, timestep, scenario_index):
        if self.level_index.get_value(scenario_index) == 6: # Drought emergency (Level 5)
            salt_front_river_mile = self.salt_front_river_mile.get_value(scenario_index)
            if salt_front_river_mile >= 92.5:
                return self.rm_factor_mrf_92_5.get_value(scenario_index)
            elif salt_front_river_mile >= 87:
                return self.rm_factor_mrf_87.get_value(scenario_index)
            elif salt_front_river_mile >= 82.9:
                return self.rm_factor_mrf_82_9.get_value(scenario_index)
            else:
                return self.rm_factor_mrf_below_82_9.get_value(scenario_index)
        else:
            return 1.0
    
    @classmethod
    def load(cls, model, data):
        mrf = data.pop("mrf")
        salt_front_river_mile = load_parameter(model, "salt_front_river_mile")
        rm_factor_mrf_92_5 = load_parameter(model, f"salt_front_adjust_factor_92_5_mrf_{mrf}")
        rm_factor_mrf_87 = load_parameter(model, f"salt_front_adjust_factor_87_mrf_{mrf}")
        rm_factor_mrf_82_9 = load_parameter(model, f"salt_front_adjust_factor_82_9_mrf_{mrf}")
        rm_factor_mrf_below_82_9 = load_parameter(model, f"salt_front_adjust_factor_below_82_9_mrf_{mrf}")
            
        level_index = load_parameter(model, "drought_level_agg_nyc")

        return cls(model, mrf, salt_front_river_mile, 
                   rm_factor_mrf_92_5,
                   rm_factor_mrf_87,
                   rm_factor_mrf_82_9,
                   rm_factor_mrf_below_82_9,
                   level_index,
                    **data) 
SaltFrontAdjustFactor.register()