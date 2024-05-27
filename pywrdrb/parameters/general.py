"""
Defines custom Pywr parameters used to implement generic helpful functions. 

LaggedReservoirRelease:
    Provides access to a previous node flow or parameter value. 
    This is useful for calculating release for N timesteps ago based on rolling avg parameters 
    for N & (N-1) timesteps.
"""

from pywr.parameters import Parameter, load_parameter 


class LaggedReservoirRelease(Parameter):
    """
    Pywr doesnt have a parameter to return a previous (>1 timesteps) node flow or parameter value. But we can
    calculate release for N timesteps ago based on rolling avg parameters for N & (N-1) timesteps.
    
    Attributes:
        lag (int): number of timesteps ago
        roll_mean_lag_outflow (Parameter): rolling mean outflow parameter for N timesteps ago
        roll_mean_lagMinus1_outflow (Parameter): rolling mean outflow parameter for (N-1) timesteps ago
        roll_mean_lag_spill (Parameter): rolling mean spill parameter for N timesteps ago
        roll_mean_lagMinus1_spill (Parameter): rolling mean spill parameter for (N-1) timesteps ago
        
    Methods:
        value(timestep, scenario_index): returns the release value for N timesteps ago
        load(model, data): loads the parameter from the model dictionary
    """
    def __init__(self, model, lag, 
                 roll_mean_lag_outflow, 
                 roll_mean_lagMinus1_outflow, 
                 roll_mean_lag_spill,
                 roll_mean_lagMinus1_spill, **kwargs):
        super().__init__(model, **kwargs)
        self.lag = lag
        self.roll_mean_lag_outflow = roll_mean_lag_outflow
        self.roll_mean_lag_spill = roll_mean_lag_spill

        self.children.add(roll_mean_lag_outflow)
        self.children.add(roll_mean_lag_spill)

        if lag > 1:
            self.roll_mean_lagMinus1_outflow = roll_mean_lagMinus1_outflow
            self.roll_mean_lagMinus1_spill = roll_mean_lagMinus1_spill
            self.children.add(roll_mean_lagMinus1_outflow)
            self.children.add(roll_mean_lagMinus1_spill)

    def value(self, timestep, scenario_index):
        """
        value_lag = lag * rollmean_lag - (lag - 1) * rollmean_lagMinus1
        
        Args:
            timestep (int): current timestep
            scenario_index (int): current scenario index
            
        Returns:
            float: release value for N timesteps ago
        """
        if self.lag == 1:
            value = self.roll_mean_lag_outflow.get_value(scenario_index) + \
                    self.roll_mean_lag_spill.get_value(scenario_index)
        else:
            value = self.lag * (self.roll_mean_lag_outflow.get_value(scenario_index) + \
                                self.roll_mean_lag_spill.get_value(scenario_index)) - \
                    (self.lag - 1) * (self.roll_mean_lagMinus1_outflow.get_value(scenario_index) + \
                                      self.roll_mean_lagMinus1_spill.get_value(scenario_index))
        return max(value, 0.)


    @classmethod
    def load(cls, model, data):
        """Setup the parameter."""
        lag = data.pop('lag')
        node = data.pop('node')

        roll_mean_lag_outflow = load_parameter(model, f'outflow_{node}_rollmean{lag}')
        roll_mean_lag_spill = load_parameter(model, f'spill_{node}_rollmean{lag}')

        if lag > 1:
            roll_mean_lagMinus1_outflow = load_parameter(model, f'outflow_{node}_rollmean{lag - 1}')
            roll_mean_lagMinus1_spill = load_parameter(model, f'spill_{node}_rollmean{lag-1}')
        else:
            roll_mean_lagMinus1_outflow = None
            roll_mean_lagMinus1_spill = None

        return cls(model, lag, roll_mean_lag_outflow, roll_mean_lagMinus1_outflow, roll_mean_lag_spill,
                   roll_mean_lagMinus1_spill, **data)


# register the custom parameter so Pywr recognizes it
LaggedReservoirRelease.register()
