"""
This file contains different class objects which are used to construct custom Pywr parameters.

The parameters created here are used to implement the STARFIT-inferred reservoir operations 
policy at non-NYC reservoirs.
"""
import numpy as np
import pandas as pd
import math

from pywr.parameters import Parameter, load_parameter
from pywr.parameters import AnnualHarmonicSeriesParameter

from utils.directories import model_data_dir

### Load STARFIT parameter values
starfit_params = pd.read_csv(f'{model_data_dir}drb_model_istarf_conus.csv', sep = ',', index_col=0)

def get_reservoir_capacity(reservoir):
    return float(starfit_params['Adjusted_CAP_MG'].loc[starfit_params['reservoir'] == reservoir].iloc[0])


class STARFITReservoirRelease(Parameter):
    '''
    Custom Pywr parameter class to implement STARFIT reservoir policy.
    '''
    def __init__(self, model, storage_node, flow_parameter, **kwargs):
        super().__init__(model, **kwargs)
        
        self.node = storage_node
        self.name = storage_node.name.split('_')[1]
        self.inflow = flow_parameter
        
        # Add children
        self.children.add(flow_parameter)
        
        # Modifications to
        self.remove_R_max = True
        self.linear_below_NOR = False
        use_adjusted_storage = True
        self.WATER_YEAR_OFFSET = 0
        
        # Pull data from node
        if use_adjusted_storage:
            self.S_cap = starfit_params.loc[self.name, 'Adjusted_CAP_MG']
        else:
            self.S_cap = starfit_params.loc[self.name, 'GRanD_CAP_MG']
        
        # Store STARFIT parameters
        self.NORhi_mu = starfit_params.loc[self.name, 'NORhi_mu']
        self.NORhi_min = starfit_params.loc[self.name, 'NORhi_min']
        self.NORhi_max = starfit_params.loc[self.name, 'NORhi_max']
        self.NORhi_alpha = starfit_params.loc[self.name, 'NORhi_alpha']
        self.NORhi_beta = starfit_params.loc[self.name, 'NORhi_beta']
        
        self.NORlo_mu = starfit_params.loc[self.name, 'NORlo_mu']
        self.NORlo_min = starfit_params.loc[self.name, 'NORlo_min']
        self.NORlo_max = starfit_params.loc[self.name, 'NORlo_max']
        self.NORlo_alpha = starfit_params.loc[self.name, 'NORlo_alpha']
        self.NORlo_beta = starfit_params.loc[self.name, 'NORlo_beta']
        
        self.Release_alpha1 = starfit_params.loc[self.name, 'Release_alpha1']
        self.Release_alpha2 = starfit_params.loc[self.name, 'Release_alpha2']
        self.Release_beta1 = starfit_params.loc[self.name, 'Release_beta1']
        self.Release_beta2 = starfit_params.loc[self.name, 'Release_beta2']
        
        self.Release_c = starfit_params.loc[self.name, 'Release_c']
        self.Release_p1 = starfit_params.loc[self.name, 'Release_p1']
        self.Release_p2 = starfit_params.loc[self.name, 'Release_p2']
        
        self.I_bar = starfit_params.loc[self.name, 'GRanD_MEANFLOW_MGD']
        
        self.R_max = 999999 if self.remove_R_max else ((starfit_params.loc[self.name, 'Release_max']+1)*self.I_bar)
        self.R_min = (starfit_params.loc[self.name, 'Release_min'] +1)* self.I_bar

        
    def standardize_inflow(self, inflow):
        return (inflow - self.I_bar) / self.I_bar        
    
    def calculate_percent_storage(self, storage):
        return (storage / self.S_cap)
    
    def get_NORhi(self, timestep):
        c = math.pi*(timestep.dayofyear + self.WATER_YEAR_OFFSET)/365  
        NORhi = (self.NORhi_mu + self.NORhi_alpha * math.sin(2*c) +
                 self.NORhi_beta * math.cos(2*c))
        if (NORhi <= self.NORhi_max) and (NORhi >= self.NORhi_min):
            return NORhi/100
        elif (NORhi > self.NORhi_max):
            return self.NORhi_max/100
        else:
            return self.NORhi_min/100
        
    def get_NORlo(self, timestep):
        c = math.pi*(timestep.dayofyear + self.WATER_YEAR_OFFSET)/365
        NORlo = (self.NORlo_mu + self.NORlo_alpha * math.sin(2*c) +
                 self.NORlo_beta * math.cos(2*c))
        if (NORlo <= self.NORlo_max) and (NORlo >= self.NORlo_min):
            return NORlo/100
        elif (NORlo > self.NORlo_max):
            return self.NORlo_max/100
        else:
            return self.NORlo_min/100 
        
    def get_harmonic_release(self, timestep):
        c = math.pi*(timestep.dayofyear + self.WATER_YEAR_OFFSET)/365
        R_avg_t = self.Release_alpha1*math.sin(2*c) + self.Release_alpha2*math.sin(4*c) + self.Release_beta1*math.cos(2*c) + self.Release_beta2*math.cos(4*c)
        return R_avg_t


    def calculate_release_adjustment(self, S_hat, I_hat,
                                     NORhi_t, NORlo_t):
        # Calculate normalized value within NOR
        A_t = (S_hat - NORlo_t) / (NORhi_t)
        epsilon_t = self.Release_c + self.Release_p1 * A_t + self.Release_p2 * I_hat
        return epsilon_t
    
    
    def calculate_target_release(self, harmonic_release, epsilon,
                                 NORhi, NORlo, S_hat, I):
        if (S_hat <= NORhi) and (S_hat >= NORlo):
            target = min((self.I_bar * (harmonic_release + epsilon) + self.I_bar), self.R_max)
        elif (S_hat > NORhi):
            target = min((self.S_cap * (S_hat - NORhi) + I*7)/7, self.R_max)
        else:
            if self.linear_below_NOR:
                target = (self.I_bar * (harmonic_release + epsilon) + self.I_bar) * (1 - (NORlo - S_hat)/NORlo)
            else:
                target = self.R_min
        return target
        
        
    def value(self, timestep, scenario_index):
        # Get current storage and inflow conditions
        inflow_t = self.inflow.get_value(scenario_index)
        storage_t = self.node.volume
        
        S_hat_t = self.calculate_percent_storage(storage_t)
        I_hat_t = self.standardize_inflow(inflow_t)
        
        NORhi_t = self.get_NORhi(timestep)
        NORlo_t = self.get_NORlo(timestep)
        
        seasonal_release_t = self.get_harmonic_release(timestep)
            
        # Get adjustment from seasonal release
        epsilon_t = self.calculate_release_adjustment(S_hat_t, I_hat_t, NORhi_t, NORlo_t)
        
        # Get target release
        target_release = self.calculate_target_release(S_hat = S_hat_t,
                                                       I = inflow_t,
                                                       NORhi=NORhi_t,
                                                       NORlo=NORlo_t,
                                                       epsilon=epsilon_t,
                                                       harmonic_release=seasonal_release_t)
        
        # Get actual release subject to constraints
        release = max(min(target_release, inflow_t + storage_t), (inflow_t + storage_t - self.S_cap)) 
        if (S_hat_t <= 0.01) or (storage_t < 50):
            print(f'{self.node.name} Going to zero storage')
        return max(0, release)
        
        
    @classmethod
    def load(cls, model, data):
        name = data.pop("node")
        storage_node = model.nodes[f'reservoir_{name}']
        flow_parameter = load_parameter(model, f'flow_{name}')
        return cls(model, storage_node, flow_parameter, **data)
    
# Register the parameter for use with Pywr
STARFITReservoirRelease.register()