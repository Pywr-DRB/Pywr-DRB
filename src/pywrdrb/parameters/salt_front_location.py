"""
Contains the custom parameter classes which use the Salinity LSTM model to predict the 
salt front location in the Delaware River Basin (DRB).

Overview
--------
The Salinity LSTM model is developed based on Gorski et al. (2024). We rebuild the model using the LSTM and BMI sturcture derived from Zwart et al. (2023).
from Zwart et al. (2023)
to predict mean water temperature at Lordville each timestep.

PywrDRB_ML plugin: github.com/philip928lin/PywrDRB-ML

Gorski, G., Cook, S., Snyder, A., Appling, A. P., Thompson, T., Smith, J. D., 
Warner, J. C., & Topp, S. N. (2024). Deep learning of estuary salinity dynamics is 
physically accurate at a fraction of hydrodynamic model computational cost. Limnology 
and Oceanography, 69(5), 1070–1085. https://doi.org/10.1002/lno.12549

Zwart, J. A., Oliver, S. K., Watkins, W. D., Sadler, J. M., Appling, A. P., Corson‐Dosch,
H. R., ... & Read, J. S. (2023). Near‐term forecasts of stream temperature using deep learning
and data assimilation in support of management decisions.
JAWRA Journal of the American Water Resources Association, 59(2), 317-337.

To do
------
- We have not yet add salt front to the policy. Likely, we will call 
  UpdateSaltFrontLocation as a childern and access salinity_model to get mu 
  (previous day salt front location) to update the Trenton/Montague flow target policy 
  during the emergent drought.
- Currently, the sd of the salt front is super large and not usable. We will need to further
  investigate the model and the data to improve the sd prediction.

Change Log
----------
Chung-Yi Lin, 2025-05-25, Create the script.
Chung-Yi Lin, 2025-05-28, Fixed logical bugs and verify the correctness of the output.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from pywr.parameters import Parameter, load_parameter
from pywr.recorders import Recorder

from pywrdrb.path_manager import get_pn_object
# Directories (PathNavigator)
# https://github.com/philip928lin/PathNavigator
global pn
pn = get_pn_object()

class SalinityModel(Parameter):
    def __init__(self, model, 
                 start_date, 
                 Q_Trenton_lstm_var_name, 
                 Q_Schuylkill_lstm_var_name, 
                 PywrDRB_ML_plugin_path, 
                 disable_tqdm, debug, **kwargs):
        super().__init__(model, **kwargs)
        """
        Initialize the SalinityModel parameter.
        
        Parameters
        ----------
        model : pywr.core.Model
            The Pywr model object.
        start_date : str
            The start date for the model in "YYYY-MM-DD" format.
        Q_Trenton_lstm_var_name : str
            The variable name for the Trenton flow in the LSTM model.
        Q_Schuylkill_lstm_var_name : str
            The variable name for the Schuylkill flow in the LSTM model.
        PywrDRB_ML_plugin_path : str
            The path to the PywrDRB_ML plugin directory containing the LSTM model configuration.
        disable_tqdm : bool
            If True, disables the tqdm progress bar during model initialization.
        **kwargs : dict
            Additional keyword arguments for the Parameter class.
        """
        
        self.debug = debug
        
        # Add the plug-in directory to the system path and then import plugin 
        PywrDRB_ML_plugin_path = Path(PywrDRB_ML_plugin_path)
        sys.path.insert(1, PywrDRB_ML_plugin_path) 
        from src.torch_bmi import bmi_lstm # BMI wrapper for the LSTM model
        
        # Final salt front location (river mile)
        self.mu, self.sd = np.nan, np.nan
        self.delTrenton_lstm_var_name = Q_Trenton_lstm_var_name
        self.outletSchuylkill_lstm_var_name = Q_Schuylkill_lstm_var_name
        
        #!! For debugging purposes
        if debug is False:
            self.records = {
                "date": [],
                "mu": [],
                "sd": [],
                "Q_Trenton": [],
                "Q_Schuylkill": [],
            }
        
        # Initialize the LSTM model for salt front prediction.
        # Default model path is "models/SalinityLSTM.yml" in the PywrDRB_ML_plugin_path.
        lstm = bmi_lstm()
        lstm.initialize(config_file=PywrDRB_ML_plugin_path / "models" / "SalinityLSTM.yml", train=False, root_dir=PywrDRB_ML_plugin_path)
        self.lstm = lstm
     
        # Get the LSTM start date. 
        if start_date is not None:
            dt = max(model.timestepper.start, datetime.strptime(start_date, "%Y-%m-%d"))
        else:
            dt = model.timestepper.start
        dt1 = pd.to_datetime(lstm.get_current_date())
        self.start_date = min(max(dt1, dt), dt)
        length=max((self.start_date - dt1).days, 0)
        if length == 0:
            self.start_date = dt1
            
        if disable_tqdm is False:
            print(f"Advancing the SalinityLSTM model to the start date: {self.start_date} (length={length} days)")
        # Advance the LSTM models to the start date 
        # For debugging
        def update_until(lstm, length):
            # Get unscaled lstm input data
            unscaled_data = lstm.get_unscaled_values(lead_time=length)
            for _ in tqdm(range(length), disable=disable_tqdm):
                
                for vi, var in enumerate(unscaled_data):
                    if var in lstm.x_vars:
                        lstm.set_value(var, unscaled_data.loc[int(lstm.t), var])
                    
                lstm.update()
        
        update_until(lstm=lstm, length=length)

        self.current_date = self.start_date # safenet to ensure the LSTM is only update once per timestep
    
    def update(self, Q_Trenton, Q_Schuylkill, current_date):
        previous_date = current_date.datetime - timedelta(days=1) # as we are using the previous day flow to update the LSTM
        if previous_date < self.current_date:
            return None
        
        elif previous_date == self.current_date:
            
            lstm = self.lstm
            Q_Trenton_lstm_var_name = self.delTrenton_lstm_var_name
            Q_Schuylkill_lstm_var_name = self.outletSchuylkill_lstm_var_name
            
            unscaled_data = lstm.get_unscaled_values(lead_time=0) # Retrieve unscaled data for the current date
            for var in lstm.x_vars:
                if var == Q_Trenton_lstm_var_name:
                    lstm.set_value(var, Q_Trenton)
                elif var == Q_Schuylkill_lstm_var_name:
                    lstm.set_value(var, Q_Schuylkill)
                else:
                    lstm.set_value(var, unscaled_data.loc[0, var]) 
            lstm.update()
            
            # salt_front (We use Jake's bmi so we still we temperature variable name internally in the bmi)
            salt_front_mu = np.zeros(1)
            salt_front_sd = np.zeros(1)
            lstm.get_value("channel_water_surface_water__mu_max_of_temperature", salt_front_mu)
            lstm.get_value("channel_water_surface_water__sd_max_of_temperature", salt_front_sd)
            salt_front_mu, salt_front_sd = salt_front_mu[0], salt_front_sd[0]
            
            self.mu, self.sd = salt_front_mu, salt_front_sd
            
            #!! For debugging purposes
            if self.debug:
                records = self.records
                records["date"].append(previous_date)
                records["mu"].append(salt_front_mu)
                records["sd"].append(salt_front_sd)
                records["Q_Trenton"].append(Q_Trenton)
                records["Q_Schuylkill"].append(Q_Schuylkill)
            
            self.current_date += timedelta(days=1)
            return None
    
    def value(self, timestep, scenario_index):
        pass
        return np.nan

    @classmethod
    def load(cls, model, data):
        start_date = data.pop("start_date")
        Q_Trenton_lstm_var_name = data.pop("Q_Trenton_lstm_var_name")
        Q_Schuylkill_lstm_var_name = data.pop("Q_Schuylkill_lstm_var_name")
        PywrDRB_ML_plugin_path = data.pop("PywrDRB_ML_plugin_path")
        disable_tqdm = data.pop("disable_tqdm", True)
        debug = data.pop("debug", False)
        return cls(model, start_date, Q_Trenton_lstm_var_name, Q_Schuylkill_lstm_var_name, PywrDRB_ML_plugin_path, disable_tqdm, debug, **data)
SalinityModel.register()
# salinity_model

class UpdateSaltFrontLocation(Parameter):
    def __init__(self, model, salinity_model, **kwargs):
        super().__init__(model, **kwargs)
        """
        Update the salt front location based on the salinity model predictions.
        
        parameters
        ----------
        model : pywr.core.Model
            The Pywr model object.
        salinity_model : SalinityModel
            The SalinityModel parameter object that provides the salt front predictions.
        **kwargs : dict
            Additional keyword arguments for the Parameter class.
        """
        self.salinity_model = salinity_model
        
        # To ensure downstream_add_thermal_release_to_target_cannonsville & pepacton are updated before this parameter
        # This will also ensure forecast is run before predict
        # To ensure thermal_release_requirement is run before this parameter.
        self.children.add(salinity_model)
        
        self.link_delTrenton = self.model.nodes["link_delTrenton"]
        self.link_outletSchuylkill = self.model.nodes["link_outletSchuylkill"]
        self.children.add(self.link_delTrenton)
        self.children.add(self.link_outletSchuylkill)
    
    def setup(self):
        super().setup()  # CRITICAL
        
        pass
    
    def value(self, timestep, scenario_index):
        salinity_model = self.salinity_model
        Q_Trenton = self.link_delTrenton.prev_flow[0]
        Q_Schuylkill = self.link_outletSchuylkill.prev_flow[0]
        salinity_model.update(Q_Trenton, Q_Schuylkill, timestep)
        #print(f"Update salt front location: {timestep.datetime} | Q_Trenton: {Q_Trenton}, Q_Schuylkill: {Q_Schuylkill}")
        return Q_Trenton

    @classmethod
    def load(cls, model, data):
        salinity_model = load_parameter(model, "salinity_model")
        return cls(model, salinity_model, **data) 
UpdateSaltFrontLocation.register()
# update_salt_front_location

class SaltFrontLocation(Parameter):
    def __init__(self, model, salinity_model, update_salt_front_location, variable, **kwargs):
        super().__init__(model, **kwargs)
        """
        A parameter to access the salt front location (mu or sd) from the salinity model.
        
        Parameters
        ----------
        model : pywr.core.Model
            The Pywr model object.
        salinity_model : SalinityModel
            The SalinityModel parameter object that provides the salt front predictions.
        update_salt_front_location : UpdateSaltFrontLocation
            The UpdateSaltFrontLocation parameter that updates the salt front location.
        variable : str
            The variable to access from the salinity model, either "mu" for the mean salt front location or "sd" for the standard deviation.
        **kwargs : dict
            Additional keyword arguments for the Parameter class.
        """
        self.salinity_model = salinity_model
        self.variable = variable

        # To ensure update_salt_front_location is run before this parameter.
        self.children.add(update_salt_front_location)

    def value(self, timestep, scenario_index):
        # The forecasted temperature should be populated when making the control release decision.
        # If activate_thermal_control is False, the forecasted temperature will be None. 
        if self.variable == "mu":
            return self.salinity_model.mu
        elif self.variable == "sd":
            return self.salinity_model.sd
        else:
            raise ValueError("Invalid variable. Must be 'mu' or 'sd'.")
        
    @classmethod
    def load(cls, model, data):
        assert "variable" in data.keys()
        salinity_model = load_parameter(model, "salinity_model")
        update_salt_front_location = load_parameter(model, "update_salt_front_location")
        variable = data.pop("variable")
        return cls(model, salinity_model, update_salt_front_location, variable, **data)
SaltFrontLocation.register()
# salt_front_location_mu
# salt_front_location_sd
