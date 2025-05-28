"""
Contains the custom parameter classes which use the LSTM model from Zwart et al. (2023)
to predict mean water temperature at Lordville each timestep.

Overview
--------
The temperature model is developed based on Zwart et al. (2023). In order to fit to the 
control purpose, we construct LSTM1 to predict the Cannonsville downstream gauge temperature (T_C) 
and LSTM2 to predict the East Branch flow and the natural flow to Lordville (T_i).
The final water temperature at Lordville (T_L) is calculated by mapping the average temperature (Tavg)
to the maximum temperature (T_L) using a random forest model.

PywrDRB_ML plugin: github.com/philip928lin/PywrDRB-ML

LSTM model reference:
Zwart, J. A., Oliver, S. K., Watkins, W. D., Sadler, J. M., Appling, A. P., Corson‐Dosch,
H. R., ... & Read, J. S. (2023). Near‐term forecasts of stream temperature using deep learning
and data assimilation in support of management decisions.
JAWRA Journal of the American Water Resources Association, 59(2), 317-337.

Change Log
----------
Chung-Yi Lin, 2025-05-25, Create the script.
"""
# Necessary evil for lstm to find files
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from tqdm import tqdm
from pywr.parameters import Parameter, load_parameter

from pywrdrb.path_manager import get_pn_object
# Directories (PathNavigator)
# https://github.com/philip928lin/PathNavigator
global pn
pn = get_pn_object()

class TemperatureModel(Parameter):
    def __init__(self, model, start_date, activate_thermal_control, Q_C_lstm_var_name, Q_i_lstm_var_name, PywrDRB_ML_plugin_path, disable_tqdm, **kwargs):
        super().__init__(model, **kwargs)
        """
        A custom parameter class to predict daily maximum water temperature at Lordville using LSTM models.
        
        Parameters
        ----------
        model : pywr.core.Model
            The Pywr model object.
        start_date : str
            The start date for the model in "YYYY-MM-DD" format. If None, uses the model's start date.
        activate_thermal_control : bool
            If True, activates the thermal control mechanism.
        Q_C_lstm_var_name : str
            The variable name for the Cannonsville reservoir downstream flow in the LSTM model.
        Q_i_lstm_var_name : str
            The variable name for the East Branch flow and the natural flow to Lordville in the LSTM model.
        PywrDRB_ML_plugin_path : str
            The path to the PywrDRB_ML plugin directory containing the LSTM model configuration.
        disable_tqdm : bool
            If True, disables the tqdm progress bar during model initialization.
        **kwargs : dict
            Additional keyword arguments for the Parameter class.
        """
        
        # import plugin 
        PywrDRB_ML_plugin_path = Path(PywrDRB_ML_plugin_path)
        sys.path.insert(1, PywrDRB_ML_plugin_path) 
        from src.torch_bmi import bmi_lstm
        
        # Final water temperature at Lordville
        self.mu, self.sd = np.nan, np.nan
        
        # Forecasted water temperature at Lordville
        self.forecasted_mu, self.forecasted_sd = np.nan, np.nan
        
        # Indicate whether to activate thermal control
        self.activate_thermal_control = activate_thermal_control
        self.Q_C_lstm_var_name = Q_C_lstm_var_name
        self.Q_i_lstm_var_name = Q_i_lstm_var_name
        
        # Predict the Cannonsville reservoir release temperature (T_C)
        lstm1 = bmi_lstm()
        lstm1.initialize(config_file=PywrDRB_ML_plugin_path / "models" / "TempLSTM1.yml", train=False, root_dir=PywrDRB_ML_plugin_path)
        self.lstm1 = lstm1
        
        # Predict the water temperature for east branch (T_i)
        lstm2 = bmi_lstm()
        lstm2.initialize(config_file=PywrDRB_ML_plugin_path / "models" / "TempLSTM2.yml", train=False, root_dir=PywrDRB_ML_plugin_path)
        self.lstm2 = lstm2
        
        # Map Tavg to Tmax (T_L) at Lordville
        rf_model = joblib.load(PywrDRB_ML_plugin_path / "models" / "rf_model.gz")
        self.rf_model = rf_model
        
        # Get the start date, which the latest among LSTM1, LSTM2, and pywrdrb start dates
        if start_date is not None:
            dt = max(model.timestepper.start, datetime.strptime(start_date, "%Y-%m-%d"))
        else:
            dt = model.timestepper.start
        
        # Identify the start date of the LSTM models
        dt1 = pd.to_datetime(lstm1.get_current_date())
        dt2 = pd.to_datetime(lstm2.get_current_date())
        self.start_date = min(max(dt1, dt2, dt), dt)
        length1 = max((self.start_date - dt1).days, 0)
        length2 = max((self.start_date - dt2).days, 0)
        if length1 == 0 and length2 == 0:
            self.start_date = max(dt1, dt2)
        elif length1 == 0 and length2 > 0:
            self.start_date = dt1
        elif length1 > 0 and length2 == 0:
            self.start_date = dt2
        length1 = max((self.start_date - dt1).days, 0)
        length2 = max((self.start_date - dt2).days, 0)
        
        # Advance the LSTM models to the start date 
        def update_until(lstm, length):
            # Get unscaled lstm input data
            unscaled_data = lstm.get_unscaled_values(lead_time=length)
            for _ in tqdm(range(length), disable=disable_tqdm):
                for vi, var in enumerate(unscaled_data):
                    if var in lstm.x_vars:
                        lstm.set_value(var, unscaled_data.loc[int(lstm.t), var])
                lstm.update()
                
        if disable_tqdm is False: # For debugging
            print(f"Advancing TempLSTM models to the {self.start_date} ...")
            
        update_until(lstm=self.lstm1, length=length1)
        update_until(lstm=self.lstm2, length=length2)
        
        # Safenet to ensure the LSTM is only update once per timestep
        self.current_date = self.start_date 
        
        # Initialize thermal mitigation bank size (MGD)
        self.thermal_mitigation_bank_size = 1620
        self.remained_bank_amount = 1620

    def make_control_release(self, Q_C, Q_i, timestep):
        # activate if self.activate_thermal_control is True
        # Here is the place to plugin control algorithm
        
        pass
        return np.nan
    
    def update(self, Q_C, Q_i, current_date):
        """
        Forward the LSTM models to one step.
        
        Parameters
        ----------
        Q_C : float
            The Cannonsville reservoir downstream flow (01425000).
        Q_i : float
            The East Branch downstream flow (01417000) and natural inflow to Lordville.
        current_date : pywr.core.CurrentDate
            The current date in the model, used to determine if the LSTM models need to be updated.
        """
        previous_date = current_date.datetime - timedelta(days=1) # as we are using the previous day flow to update the LSTM
        if previous_date < self.current_date:
            return None
        
        elif previous_date == self.current_date:
            
            lstm1 = self.lstm1
            lstm2 = self.lstm2
            Q_C_lstm_var_name = self.Q_C_lstm_var_name
            Q_i_lstm_var_name = self.Q_i_lstm_var_name
            
            # Update the LSTM1 models with the current flow values
            lstm1.set_value(Q_C_lstm_var_name, Q_C)
            lstm1.update()
            
            # Update the LSTM2 models with the current flow values
            lstm2.set_value(Q_i_lstm_var_name, Q_i)
            if Q_C_lstm_var_name in lstm2.cfg_bmi["x_vars"]:
                lstm2.set_value(Q_C_lstm_var_name, Q_C) # connected model
            lstm2.update()
            
            # T_C
            T_C_mu = np.zeros(1)
            T_C_sd = np.zeros(1)
            lstm1.get_value("channel_water_surface_water__mu_max_of_temperature", T_C_mu)
            lstm1.get_value("channel_water_surface_water__sd_max_of_temperature", T_C_sd)
            T_C_mu, T_C_sd = T_C_mu[0], T_C_sd[0]
            
            # T_i
            T_i_mu = np.zeros(1)
            T_i_sd = np.zeros(1)
            lstm2.get_value("channel_water_surface_water__mu_max_of_temperature", T_i_mu)
            lstm2.get_value("channel_water_surface_water__sd_max_of_temperature", T_i_sd)
            T_i_mu, T_i_sd = T_i_mu[0], T_i_sd[0]
            
            # Tavg
            Tavg_mu = (T_C_mu*Q_C + T_i_mu*Q_i)/(Q_C + Q_i)
            # Assuming T_i and T_C are independent
            Tavg_sd = np.sqrt((T_C_sd**2 * Q_C**2 + T_i_sd**2 * Q_i**2) / (Q_C + Q_i)**2)
            
            # T_L (Tmax at Lordville) Using a random forest model to map Tavg to T_L
            rf_model = self.rf_model
            T_L_mu = rf_model.predict([[Tavg_mu]])[0]
            T_L_sd = Tavg_sd # assuming a constant sd for T_L
            self.mu, self.sd = T_L_mu, T_L_sd
            
            print(f"T_C={T_C_mu}, T_i={T_i_mu}, Tavg={Tavg_mu}, T_L={self.mu} on {current_date.datetime.strftime('%Y-%m-%d')}")
            
            self.current_date += timedelta(days=1) # Avoid updating the LSTM models multiple times in a single timestep
            return None
    
    def value(self, timestep, scenario_index):
        # The values are retrieved through other parameters like ForecastedTemperatureBeforeThermalRelease and TemperatureAfterThermalRelease
        pass
        return np.nan

    @classmethod
    def load(cls, model, data):
        start_date = data.pop("start_date")
        activate_thermal_control = data.pop("activate_thermal_control")
        Q_C_lstm_var_name = data.pop("Q_C_lstm_var_name")
        Q_i_lstm_var_name = data.pop("Q_i_lstm_var_name")
        PywrDRB_ML_plugin_path = data.pop("PywrDRB_ML_plugin_path")
        disable_tqdm = data.pop("disable_tqdm", False)
        return cls(model, start_date, activate_thermal_control, Q_C_lstm_var_name, Q_i_lstm_var_name, PywrDRB_ML_plugin_path, disable_tqdm, **data)
TemperatureModel.register()
# temperature_model

class Estimated_Q_C(Parameter):
    # Cannonsville reservoir release => downstream gauge (01425000) => Lordville
    def __init__(self, model, downstream_release_target_cannonsville, 
                 flow_01425000, max_flow_catchmentConsumption_01425000, 
                 flow_cannonsville, max_flow_catchmentConsumption_cannonsville,
                 max_flow_delivery_nyc,
                 **kwargs):
        super().__init__(model, **kwargs)
        self.downstream_release_target_cannonsville = downstream_release_target_cannonsville
        self.flow_01425000 = flow_01425000
        self.max_flow_catchmentConsumption_01425000 = max_flow_catchmentConsumption_01425000
        self.flow_cannonsville = flow_cannonsville
        self.max_flow_catchmentConsumption_cannonsville = max_flow_catchmentConsumption_cannonsville
        self.max_flow_delivery_nyc = max_flow_delivery_nyc

        self.children.add(downstream_release_target_cannonsville)
        self.children.add(flow_01425000)
        self.children.add(max_flow_catchmentConsumption_01425000)
        self.children.add(flow_cannonsville)
        self.children.add(max_flow_catchmentConsumption_cannonsville)
        self.children.add(max_flow_delivery_nyc)
        
    def setup(self):
        super().setup()
        self.reservoir_cannonsville = self.model.nodes["reservoir_cannonsville"] 
        
    def value(self, timestep, scenario_index):
        max_flow_delivery_nyc = self.max_flow_delivery_nyc.get_value(scenario_index)
        # = min("demand_nyc", "max_flow_drought_delivery_nyc", "max_flow_ffmp_delivery_nyc")
        
        # Currently, the delivery to NYC is allocated to three NYC reservoirs through VolBalanceNYCDemand.
        # I don't want to repeat the logic here, we approximate the allocation by the reservoir volumes.
        max_volume_cannonsville = 95700 # MG (We manually input here to avoid complexity)
        max_volume_nyc = 270800 # MG (We manually input here to avoid complexity)
        max_flow_delivery_nyc_cannonsville = max_flow_delivery_nyc * max_volume_cannonsville / max_volume_nyc
        
        available_connonsville_volume = self.reservoir_cannonsville.volume[0] \
            + self.flow_cannonsville.get_value(scenario_index) \
            - self.max_flow_catchmentConsumption_cannonsville.get_value(scenario_index) \
            
        # outflow = downstream_release_target_cannonsville if the reservoir is not empty
        target_outflow = self.downstream_release_target_cannonsville.get_value(scenario_index)
            
        # Assuming max_flow_delivery_nyc_cannonsville is not the piority during the drought
        outflow = min(available_connonsville_volume, target_outflow)
        
        # For spill situation
        available_connonsville_volume = available_connonsville_volume - max_flow_delivery_nyc_cannonsville
        spill = max((available_connonsville_volume-target_outflow) - max_volume_cannonsville, 0)
        
        reservoir_release = outflow + spill 
        
        Q_C = reservoir_release + self.flow_01425000.get_value(scenario_index) \
            - self.max_flow_catchmentConsumption_01425000.get_value(scenario_index)
        return Q_C

    @classmethod
    def load(cls, model, data):
        downstream_release_target_cannonsville = load_parameter(model, "downstream_release_target_cannonsville")
        flow_01425000 = load_parameter(model, "flow_01425000") # catchment_01425000
        max_flow_catchmentConsumption_01425000 = load_parameter(model, "max_flow_catchmentConsumption_01425000")
        
        flow_cannonsville = load_parameter(model, "flow_cannonsville") # catchment_cannonsville
        max_flow_catchmentConsumption_cannonsville = load_parameter(model, "max_flow_catchmentConsumption_cannonsville")
        
        max_flow_delivery_nyc = load_parameter(model, "max_flow_delivery_nyc") # aggregated parameter
        return cls(model, downstream_release_target_cannonsville, 
                   flow_01425000, max_flow_catchmentConsumption_01425000, 
                   flow_cannonsville, max_flow_catchmentConsumption_cannonsville,
                   max_flow_delivery_nyc, **data)
Estimated_Q_C.register()
# estimated_Q_C

class Estimated_Q_i(Parameter):
    # Pepacton reservoir release => downstream gauge (01417000) => Lordville
    def __init__(self, model, downstream_release_target_pepacton, flow_01417000, 
                 max_flow_catchmentConsumption_01417000, flow_delLordville, 
                 max_flow_catchmentConsumption_delLordville, 
                 flow_pepacton, max_flow_catchmentConsumption_pepacton,
                 max_flow_delivery_nyc,
                 **kwargs):
        super().__init__(model, **kwargs)
        self.downstream_release_target_pepacton = downstream_release_target_pepacton
        self.flow_01417000 = flow_01417000
        self.max_flow_catchmentConsumption_01417000 = max_flow_catchmentConsumption_01417000
        self.flow_delLordville = flow_delLordville
        self.max_flow_catchmentConsumption_delLordville = max_flow_catchmentConsumption_delLordville
        self.flow_pepacton = flow_pepacton
        self.max_flow_catchmentConsumption_pepacton = max_flow_catchmentConsumption_pepacton
        self.max_flow_delivery_nyc = max_flow_delivery_nyc

        self.children.add(downstream_release_target_pepacton)
        self.children.add(flow_01417000)
        self.children.add(max_flow_catchmentConsumption_01417000)
        self.children.add(flow_delLordville)
        self.children.add(max_flow_catchmentConsumption_delLordville)
        self.children.add(flow_pepacton)
        self.children.add(max_flow_catchmentConsumption_pepacton)
        self.children.add(max_flow_delivery_nyc)
    
    def setup(self):
        super().setup()
        self.reservoir_pepacton = self.model.nodes["reservoir_pepacton"] 
        
    def value(self, timestep, scenario_index):
        max_flow_delivery_nyc = self.max_flow_delivery_nyc.get_value(scenario_index)
        # = min("demand_nyc", "max_flow_drought_delivery_nyc", "max_flow_ffmp_delivery_nyc")
        
        # Currently, the delivery to NYC is allocated to three NYC reservoirs through VolBalanceNYCDemand.
        # I don't want to repeat the logic here, we approximate the allocation by the reservoir volumes.
        max_volume_pepacton = 140200 # MG (We manually input here to avoid complexity)
        max_volume_nyc = 270800 # MG (We manually input here to avoid complexity)
        max_flow_delivery_nyc_pepacton = max_flow_delivery_nyc * max_volume_pepacton / max_volume_nyc
        
        available_pepacton_volume = self.reservoir_pepacton.volume[0] \
            + self.flow_pepacton.get_value(scenario_index) \
            - self.max_flow_catchmentConsumption_pepacton.get_value(scenario_index) \
            
        # outflow = downstream_release_target_pepacton if the reservoir is not empty
        target_outflow = self.downstream_release_target_pepacton.get_value(scenario_index)
            
        # Assuming max_flow_delivery_nyc_pepacton is not the piority during the drought
        outflow = min(available_pepacton_volume, target_outflow)
        
        # For spill situation
        available_pepacton_volume = available_pepacton_volume - max_flow_delivery_nyc_pepacton
        spill = max((available_pepacton_volume - target_outflow) - max_volume_pepacton, 0)
        
        reservoir_release = outflow + spill 

        # Q_i The East Branch downstream flow (01417000) and natural inflow to Lordville.
        Q_i = reservoir_release \
            + self.flow_01417000.get_value(scenario_index) \
            - self.max_flow_catchmentConsumption_01417000.get_value(scenario_index) \
            + self.flow_delLordville.get_value(scenario_index) \
            - self.max_flow_catchmentConsumption_delLordville.get_value(scenario_index) 
        return Q_i

    @classmethod
    def load(cls, model, data):
        # We can not directly call link_01417000 as its value require the release from Pepacton which is not available at this point
        # link_01417000 = 0 = outflow_pepacton + spill_pepacton + catchment_01417000 - catchmentWithdrawal_01417000 - link_delLordville
        # Uncosummed withdrawal will be return to the river
        # Q_i = load_parameter(model, "link_01417000")
        
        #catchment_01417000 - catchmentWithdrawal_01417000
        downstream_release_target_pepacton = load_parameter(model, "downstream_release_target_pepacton")
        flow_01417000 = load_parameter(model, "flow_01417000") # catchment_01417000
        max_flow_catchmentConsumption_01417000 = load_parameter(model, "max_flow_catchmentConsumption_01417000")
        flow_delLordville = load_parameter(model, "flow_delLordville") # catchment_delLordville
        max_flow_catchmentConsumption_delLordville = load_parameter(model, "max_flow_catchmentConsumption_delLordville")
        flow_pepacton = load_parameter(model, "flow_pepacton") # catchment_pepacton
        max_flow_catchmentConsumption_pepacton = load_parameter(model, "max_flow_catchmentConsumption_pepacton")
        max_flow_delivery_nyc = load_parameter(model, "max_flow_delivery_nyc") # aggregated parameter
        
        return cls(model, downstream_release_target_pepacton, 
                   flow_01417000, max_flow_catchmentConsumption_01417000, 
                   flow_delLordville, max_flow_catchmentConsumption_delLordville, 
                   flow_pepacton, max_flow_catchmentConsumption_pepacton,
                   max_flow_delivery_nyc,
                   **data)
Estimated_Q_i.register()
# estimated_Q_i

# Calculate the total thermal release requirement at Lordville    
class ThermalReleaseRequirement(Parameter):
    def __init__(self, model, temperature_model, update_temperature_at_lordville, Q_C, Q_i, **kwargs):
        super().__init__(model, **kwargs)
        self.Q_C = Q_C
        self.Q_i = Q_i
        self.thermal_release = 0.0
        
        # To ensure cannonsville_release & pepacton_release are updated before this parameter
        self.children.add(Q_C)
        self.children.add(Q_i) 
        self.children.add(update_temperature_at_lordville)
        self.temperature_model = temperature_model
        self.activate_thermal_control = temperature_model.activate_thermal_control
        self.mu, self.sd = 0, 0
    
    def value(self, timestep, scenario_index):
        temperature_model = self.temperature_model
        # Check if thermal control is activated
        if temperature_model.activate_thermal_control is False:
            return 0.0 # No thermal release
        else:
            thermal_release = temperature_model.make_control_release(
                Q_C = self.Q_C.get_value(scenario_index), 
                Q_i = self.Q_i.get_value(scenario_index),
                timestep = timestep
            )
            self.thermal_release = thermal_release
            return thermal_release
    
    @classmethod
    def load(cls, model, data):
        Q_C = load_parameter(model, "estimated_Q_C")
        Q_i = load_parameter(model, "estimated_Q_i")      
        temperature_model = load_parameter(model, "temperature_model")
        update_temperature_at_lordville = load_parameter(model, "update_temperature_at_lordville")
        return cls(model, temperature_model, update_temperature_at_lordville, Q_C, Q_i, **data) 
ThermalReleaseRequirement.register() 
# thermal_release_requirement

class ForecastedTemperatureBeforeThermalRelease(Parameter):
    def __init__(self, model, temperature_model, thermal_release_requirement, variable, **kwargs):
        super().__init__(model, **kwargs)
        self.temperature_model = temperature_model
        self.variable = variable

        # To ensure thermal_release_requirement is run before this parameter.
        self.children.add(thermal_release_requirement)

    def value(self, timestep, scenario_index):
        # The forecasted temperature should be populated when making the control release decision.
        # If activate_thermal_control is False, the forecasted temperature will be None. 
        if self.variable == "mu":
            return self.temperature_model.forecasted_mu
        elif self.variable == "sd":
            return self.temperature_model.forecasted_sd
        else:
            raise ValueError("Invalid variable. Must be 'mu' or 'sd'.")
        
    @classmethod
    def load(cls, model, data):
        assert "variable" in data.keys()
        temperature_model = load_parameter(model, "temperature_model")
        thermal_release_requirement = load_parameter(model, "thermal_release_requirement")
        variable = data.pop("variable")
        return cls(model, temperature_model, thermal_release_requirement, variable, **data)
ForecastedTemperatureBeforeThermalRelease.register()
# forecasted_temperature_before_thermal_release_mu
# forecasted_temperature_before_thermal_release_sd

class UpdateTemperatureAtLordville(Parameter):
    def __init__(self, model, temperature_model, flow_delLordville, max_flow_catchmentConsumption_delLordville, **kwargs):
        super().__init__(model, **kwargs)
        self.temperature_model = temperature_model
        self.flow_delLordville = flow_delLordville
        self.max_flow_catchmentConsumption_delLordville = max_flow_catchmentConsumption_delLordville
        
        # To ensure downstream_add_thermal_release_to_target_cannonsville & pepacton are updated before this parameter
        # This will also ensure forecast is run before predict
        # To ensure thermal_release_requirement is run before this parameter.
        #self.children.add(thermal_release_requirement)
        self.children.add(flow_delLordville)
        self.children.add(max_flow_catchmentConsumption_delLordville)
    
    def setup(self):
        super().setup()  # CRITICAL
        self.link_01425000 = self.model.nodes["link_01425000"] # Cannonsville reservoir downstream flow (01425000)
        self.link_01417000 = self.model.nodes["link_01417000"] # East Branch downstream flow (01417000) and natural inflow to Lordville
        self.children.add(self.link_01425000)
        self.children.add(self.link_01417000)
    
    # Need to use prev flow_delLordville and max_flow_catchmentConsumption_delLordville
    # Or get prev_flow from Lordeville node and infer Q_i = Q_L - Q_C
    def value(self, timestep, scenario_index):
        temperature_model = self.temperature_model
        # Cannonsville reservoir downstream flow (01425000)
        Q_C = self.link_01425000.prev_flow[0]  
        # East Branch downstream flow (01417000) and natural inflow to Lordville
        Q_i = self.link_01417000.prev_flow[0] \
            + self.flow_delLordville.get_value(scenario_index) \
            - self.max_flow_catchmentConsumption_delLordville.get_value(scenario_index)
        #print(f"Updating temperature at Lordville with Q_C: {Q_C}, Q_i: {Q_i} at timestep {timestep}")
        temperature_model.update(Q_C, Q_i, timestep)
        return Q_i #np.nan
    
    @classmethod
    def load(cls, model, data):
        temperature_model = load_parameter(model, "temperature_model")
        flow_delLordville = load_parameter(model, "flow_delLordville") # catchment_delLordville
        max_flow_catchmentConsumption_delLordville = load_parameter(model, "max_flow_catchmentConsumption_delLordville")
        return cls(model, temperature_model, flow_delLordville, max_flow_catchmentConsumption_delLordville, **data) 
UpdateTemperatureAtLordville.register()
# update_temperature_at_lordville

class TemperatureAfterThermalRelease(Parameter):
    def __init__(self, model, temperature_model, thermal_release_requirement, variable, **kwargs):
        super().__init__(model, **kwargs)
        self.temperature_model = temperature_model
        self.variable = variable

        # To ensure update_temperature_at_lordville is run before this parameter.
        self.children.add(thermal_release_requirement)

    def value(self, timestep, scenario_index):
        # The forecasted temperature should be populated when making the control release decision.
        # If activate_thermal_control is False, the forecasted temperature will be None. 
        if self.variable == "mu":
            return self.temperature_model.mu
        elif self.variable == "sd":
            return self.temperature_model.sd
        else:
            raise ValueError("Invalid variable. Must be 'mu' or 'sd'.")
        
    @classmethod
    def load(cls, model, data):
        assert "variable" in data.keys()
        temperature_model = load_parameter(model, "temperature_model")
        thermal_release_requirement = load_parameter(model, "thermal_release_requirement")
        variable = data.pop("variable")
        return cls(model, temperature_model, thermal_release_requirement, variable, **data)
TemperatureAfterThermalRelease.register()
# temperature_after_thermal_release_mu
# temperature_after_thermal_release_sd



