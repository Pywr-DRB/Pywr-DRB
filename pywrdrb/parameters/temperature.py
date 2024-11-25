"""
Contains the custom parameter classes which use the LSTM model from Zwart et al. (2023)
to predict mean water temperature at Lordville each timestep.

Classes:
- TemperaturePrediction

LSTM model reference:
Zwart, J. A., Oliver, S. K., Watkins, W. D., Sadler, J. M., Appling, A. P., Corson‐Dosch,
H. R., ... & Read, J. S. (2023). Near‐term forecasts of stream temperature using deep learning
and data assimilation in support of management decisions.
JAWRA Journal of the American Water Resources Association, 59(2), 317-337.
"""
# Necessary evil for lstm to find files
import os
import numpy as np
import pandas as pd
from pywr.parameters import Parameter, load_parameter

from pywrdrb.utils.constants import cms_to_mgd
from pywrdrb.utils.dates import temp_pred_date_range
from pywrdrb.utils.directories import ROOT_DIR

# Adding BMI LSTM model dir to the path
import sys

bmi_temp_model_path = f"{ROOT_DIR}/../../bmi-stream-temp"

sys.path.insert(1, f"{bmi_temp_model_path}/3_model_train/src")
sys.path.insert(1, f"{bmi_temp_model_path}/3_model_train/in")
sys.path.insert(1, f"{bmi_temp_model_path}/4_model_forecast/src")

os.chdir(bmi_temp_model_path)
# BMI class for running the LSTM model
import torch
import torch_bmi

class TemperatureLSTM():
    """
    External class wrapper to operate the LSTM model from Zwart et al. (2023) for temperature prediction.
    """
    def __init__(self, start_date, torch_seed):
        """
        Initialize the LSTM model and advance it to the start date of the pywrdrb simulation

        Parameters
        ----------
        start_date: datetime object
            The start date of the pywrdrb simulation. The LSTM model will be advanced to this date.
        """
        # LSTM valid prediction date range
        self.lstm_date_range = pd.date_range(
            start=temp_pred_date_range[0], end=temp_pred_date_range[1]
        )

        # location of the BMI configuration file 
        bmi_cfg_file = f"{bmi_temp_model_path}/model_config.yml"
        self.bmi_cfg_file = bmi_cfg_file

        # creating an instance of an LSTM model
        #print("Creating an instance of an BMI_LSTM model object")
        self.lstm = torch_bmi.bmi_lstm()

        # Initializing the BMI for LSTM
        #print("Initializing the temperature prediction LSTM\n")
        self.lstm.initialize(bmi_cfg_file=bmi_cfg_file, torch_seed=torch_seed)

        ### Manully advance the LSTM to the pywrdrb simulation start
        # LSTM is set up to start on 1982-04-03
        simulation_start = start_date #model.timestepper.start
        days_to_advance = (simulation_start - self.lstm_date_range[0]).days

        # Advance the LSTM model to the simulation start date
        for ti in range(days_to_advance):
            unscaled_data = (
                self.lstm.x[
                    0,
                    int(self.lstm.t),
                ]
                * (self.lstm.input_std + 1e-10)
                + self.lstm.input_mean
            )
            for i in range(len(unscaled_data)):
                self.lstm.set_value(self.lstm.x_vars[i], unscaled_data[i])
            self.lstm.update()
        #print(
        #    f"Advanced LSTM temperature prediction model {days_to_advance} to start of pywrdrb simulation."
        #)
        print(f"LSTM model is now at date {self.lstm.dates[int(self.lstm.t)]}.")

    def forecast(self, n, total_reservoir_release_t, timestep=None):     
        """
        Forecast the mean max temperature at Lordville for the next n timesteps.
        
        Parameters
        ----------
        n: int
            Number of timesteps to forecast.
        total_reservoir_release_t: float
            Total reservoir release at the current timestep.
        timestep: int
            The current timestep of the pywrdrb simulation. Used for debugging.
        
        Returns
        -------
        mu_pred: list
            List of forecasted mean max temperature at Lordville.
        sd_pred: list
            List of forecasted standard deviation of max temperature at Lordville.
        """
        #print(f"[{timestep}] forecast: {total_reservoir_release_t}")
        # Save current hist states before modifying
        saved_ht = self.lstm.h_t.clone()
        saved_ct = self.lstm.c_t.clone()
        saved_t = self.lstm.t

        # Save the current random state in torch
        saved_rng_state = torch.get_rng_state()

        # convert from MGD to m3 s-1
        total_reservoir_release_t_cms = total_reservoir_release_t * (1.0 / cms_to_mgd)

        mu_pred = []
        sd_pred = []
        for i in range(int(saved_t), int(saved_t) + n + 1):
            unscaled_data = (
                self.lstm.x[
                    0,
                    int(self.lstm.t),
                ]
                * (self.lstm.input_std + 1e-10)
                + self.lstm.input_mean
            )
            ### set input values 
            for k in range(len(unscaled_data)):
                var_name = self.lstm.x_vars[k]
                if i == int(saved_t): # only set the first time step used drb model output
                    if var_name == "reservoir_release":
                        self.lstm.set_value(var_name, total_reservoir_release_t_cms)
                    else:
                        self.lstm.set_value(var_name, unscaled_data[k])
                else:
                    #raise
                    self.lstm.set_value(self.lstm.x_vars[k], unscaled_data[k])

            # make prediction LSTM
            self.lstm.update()

            # get predicted mean max temp
            dest_array = np.zeros(1) 
            self.lstm.get_value("channel_water_surface_water__mu_max_of_temperature", dest_array)
            mu_pred.append(dest_array[0])

            dest_array = np.zeros(1) 
            self.lstm.get_value("channel_water_surface_water__sd_max_of_temperature", dest_array)
            sd_pred.append(dest_array[0])

        # Reset to original state
        self.lstm.h_t = saved_ht
        self.lstm.c_t = saved_ct
        self.lstm.t = saved_t
        # Restore the saved random state
        torch.set_rng_state(saved_rng_state)
        # return the entire forecasted time series
        return mu_pred, sd_pred
    
    def predict(self, total_reservoir_release, timestep=None): 
        """
        Predict the mean max temperature at Lordville for the current timestep and forward the LSTM model by one timestep.

        Parameters
        ----------
        total_reservoir_release: float
            Total reservoir release at the current timestep.
        timestep: int
            The current timestep of the pywrdrb simulation. Used for debugging.
        
        Returns
        -------
        mu_pred: float
            Predicted mean max temperature at Lordville.
        sd_pred: float
            Predicted standard deviation of max temperature at Lordville.
        """
        #print(f"[{timestep}] predict")

        # convert from MGD to m3 s-1
        total_reservoir_release_cms = total_reservoir_release * (1.0 / cms_to_mgd)

        # Unscaling the driver data stored in x for the current time step
        # data are already scaled so need to unscale prior to putting in the model
        unscaled_data = (
            self.lstm.x[
                0,
                int(self.lstm.t),
            ]
            * (self.lstm.input_std + 1e-10)
            + self.lstm.input_mean
        )

        # Setting the unscaled data values for the current time step in the BMI model
        for i in range(len(unscaled_data)):
            var_name = self.lstm.x_vars[i]
            if var_name == "reservoir_release":
                self.lstm.set_value(var_name, total_reservoir_release_cms)
            else:
                self.lstm.set_value(var_name, unscaled_data[i])

        # run the BMI model with the update() function
        self.lstm.update()

        # retrieving the main prediction output from the BMI LSTM model
        # the predicted mean of max water temperature is stored in CSDMS naming convention
        mu_pred = []
        sd_pred = []
        self.lstm.get_value("channel_water_surface_water__mu_max_of_temperature", mu_pred)
        self.lstm.get_value("channel_water_surface_water__sd_max_of_temperature", sd_pred)
        err_msg = ("LSTM model should only return one value for the predicted temperature")
        err_msg += f" but returned {len(mu_pred)} values."
        assert len(mu_pred) == 1, err_msg
        return mu_pred[0], sd_pred[0]


class TemperatureModel(Parameter):
    def __init__(self, model, torch_seed, **kwargs):
        super().__init__(model, **kwargs)
        #self.mu, self.sd = 0, 0
        #self.mu_no_control, self.sd_no_control = 0, 0
        self.torch_seed = torch_seed
        self.temp_model = TemperatureLSTM(start_date=model.timestepper.start, torch_seed=torch_seed)
        self.timestep = None # safenet to ensure the LSTM is only update once per timestep

    def value(self, timestep, scenario_index):
        pass
        return -99

    @classmethod
    def load(cls, model, data):
        torch_seed = data.pop("torch_seed")
        return cls(model, torch_seed, **data)
TemperatureModel.register()

# Calculate the total thermal release requirement at Lordville    
class TotalThermalReleaseRequirement(Parameter):
    def __init__(self, model, temperature_model, cannonsville_release, pepacton_release, **kwargs):
        super().__init__(model, **kwargs)
        self.cannonsville_release = cannonsville_release
        self.pepacton_release = pepacton_release
        self._total_thermal_release = 0.0
        self.temperature_threshold = 23.89 # C = 75F

        # To ensure cannonsville_release & pepacton_release are updated before this parameter
        self.children.add(cannonsville_release)
        self.children.add(pepacton_release)

        # Initialize the LSTM model for estimating temperature without thermal release
        self.temperature_lstm = temperature_model
        self.mu, self.sd = 0, 0
        self.above_threshold = False
    
    def value(self, timestep, scenario_index):
        # Total release from both reservoirs
        total_reservoir_release = (
            self.cannonsville_release.get_value(scenario_index) 
            + self.pepacton_release.get_value(scenario_index)
        )

        # Estimate the mean max temperature at Lordville without thermal release
        mu_list, sd_list = self.temperature_lstm.temp_model.forecast(
            n=0,
            total_reservoir_release_t=total_reservoir_release,
            timestep=timestep
            )
        self.mu, self.sd = mu_list[-1], sd_list[-1]

        # Calculate the total thermal release requirement
        if self.mu > self.temperature_threshold:
            # self._total_thermal_release = mgd
            self._total_thermal_release = 64.63 # mgd = 100 cfs
            self.above_threshold = True
            #print(f"[{timestep}] forecast: {total_reservoir_release}")
        else:
            self.above_threshold = False
            self._total_thermal_release = 0

        return self._total_thermal_release
    
    @classmethod
    def load(cls, model, data):
        cannonsville_release = load_parameter(model, "downstream_release_target_cannonsville")
        pepacton_release = load_parameter(model, "downstream_release_target_pepacton")      
        temperature_model = load_parameter(model, "temperature_model")
        return cls(model, temperature_model, cannonsville_release, pepacton_release, **data) 
TotalThermalReleaseRequirement.register()

class GetTemperatureLSTMValueWithoutThermalRelease(Parameter):
    def __init__(self, model, variable, total_thermal_release_requirement, **kwargs):
        super().__init__(model, **kwargs)
        self.variable = variable
        self.total_thermal_release_requirement = total_thermal_release_requirement

        # To ensure total_thermal_release_requirement is updated before this parameter.
        self.children.add(total_thermal_release_requirement)

    def value(self, timestep, scenario_index):
        if self.variable == "mu":
            return self.total_thermal_release_requirement.mu
        elif self.variable == "sd":
            return self.total_thermal_release_requirement.sd
        else:
            raise ValueError("Invalid variable. Must be 'mu' or 'sd'.")
        
    @classmethod
    def load(cls, model, data):
        assert "variable" in data.keys()
        variable = data.pop("variable")
        total_thermal_release_requirement = load_parameter(model, "total_thermal_release_requirement")
        return cls(model, variable, total_thermal_release_requirement, **data)
GetTemperatureLSTMValueWithoutThermalRelease.register()

class AllocateThermalReleaseRequirement(Parameter):
    def __init__(self, model, reservoir, total_thermal_release_requirement, **kwargs):
        super().__init__(model, **kwargs)
        self.reservoir = reservoir
        self.total_thermal_release_requirement = total_thermal_release_requirement

        # To ensure total_thermal_release_requirement is updated before this parameter.
        self.children.add(total_thermal_release_requirement)

        # hardcoded allocation factor for now
        self.allocation_factor = {"cannonsville": 0.5, "pepacton": 0.5}

    def value(self, timestep, scenario_index):
        # Total release from both reservoirs
        total_thermal_release_requirement = self.total_thermal_release_requirement.get_value(scenario_index)
        thermal_release = self.allocation_factor[self.reservoir] * total_thermal_release_requirement

        #!!!! Will need to consider the extreme case where no additional water to release
        # track the annual thermal release  

        return thermal_release
    
    @classmethod
    def load(cls, model, data):
        assert "reservoir" in data.keys()
        assert data["reservoir"] in ["cannonsville", "pepacton"]
        reservoir = data.pop("reservoir")

        total_thermal_release_requirement = load_parameter(model, "total_thermal_release_requirement")

        return cls(model, reservoir, total_thermal_release_requirement, **data)    
AllocateThermalReleaseRequirement.register()

# run predict
class PredictedMaxTemperatureAtLordville(Parameter):
    def __init__(self, model, total_thermal_release_requirement, temperature_model, downstream_add_thermal_release_to_target_cannonsville, downstream_add_thermal_release_to_target_pepacton, **kwargs):
        super().__init__(model, **kwargs)
        self.downstream_add_thermal_release_to_target_cannonsville = downstream_add_thermal_release_to_target_cannonsville
        self.downstream_add_thermal_release_to_target_pepacton = downstream_add_thermal_release_to_target_pepacton

        # To ensure downstream_add_thermal_release_to_target_cannonsville & pepacton are updated before this parameter
        # This will also ensure forecast is run before predict
        self.children.add(downstream_add_thermal_release_to_target_cannonsville)
        self.children.add(downstream_add_thermal_release_to_target_pepacton)

        self.temperature_lstm = temperature_model
        self.total_thermal_release_requirement = total_thermal_release_requirement
        self.children.add(total_thermal_release_requirement)

        self.mu, self.sd = 0, 0
    
    def value(self, timestep, scenario_index):
        # Total release from both reservoirs
        total_reservoir_release = (
            self.downstream_add_thermal_release_to_target_cannonsville.get_value(scenario_index) 
            + self.downstream_add_thermal_release_to_target_pepacton.get_value(scenario_index)
        )

        #if self.total_thermal_release_requirement.above_threshold:
            #print(f"[{timestep}] predict: {total_reservoir_release}")

        self.mu, self.sd = self.temperature_lstm.temp_model.predict(
            total_reservoir_release=total_reservoir_release,
            timestep=timestep
            )
        # GetTemperatureLSTMValue is designed to get the mu and sig values. The return value here is not used/important.
        return self.mu
    
    @classmethod
    def load(cls, model, data):
        downstream_add_thermal_release_to_target_cannonsville = load_parameter(
            model, "downstream_add_thermal_release_to_target_cannonsville"
        )
        downstream_add_thermal_release_to_target_pepacton = load_parameter(
            model, "downstream_add_thermal_release_to_target_pepacton"
            )      

        temperature_model = load_parameter(model, "temperature_model")
        # Debugging
        total_thermal_release_requirement = load_parameter(model, "total_thermal_release_requirement")
        return cls(model, total_thermal_release_requirement, temperature_model, downstream_add_thermal_release_to_target_cannonsville, downstream_add_thermal_release_to_target_pepacton, **data) 
PredictedMaxTemperatureAtLordville.register()

class GetTemperatureLSTMValue(Parameter):
    def __init__(self, model, variable, predicted_max_temperature_at_lordville_run_lstm, **kwargs):
        super().__init__(model, **kwargs)
        self.variable = variable
        self.predicted_max_temperature_at_lordville_run_lstm = predicted_max_temperature_at_lordville_run_lstm

        # To ensure predicted_max_temperature_at_lordville_run_lstm is updated before this parameter.
        self.children.add(predicted_max_temperature_at_lordville_run_lstm)

    def value(self, timestep, scenario_index):
        redundant = self.predicted_max_temperature_at_lordville_run_lstm.get_value(scenario_index)
        if self.variable == "mu":
            return self.predicted_max_temperature_at_lordville_run_lstm.mu
        elif self.variable == "sd":
            return self.predicted_max_temperature_at_lordville_run_lstm.sd
        else:
            raise ValueError("Invalid variable. Must be 'mu' or 'sd'.")
        
    @classmethod
    def load(cls, model, data):
        assert "variable" in data.keys()
        variable = data.pop("variable")
        predicted_max_temperature_at_lordville_run_lstm = load_parameter(model, "predicted_max_temperature_at_lordville_run_lstm")
        return cls(model, variable, predicted_max_temperature_at_lordville_run_lstm, **data)
GetTemperatureLSTMValue.register()