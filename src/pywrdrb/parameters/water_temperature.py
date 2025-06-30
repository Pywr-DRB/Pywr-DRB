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

To do
------
- Currently, we did not dynamically update the lag-1 temperature at Lordville inputs,
  which we assume lag-1 information is available in the real-world.
- Will add the thermal control algorithm to the TemperatureModel class and enable
  forecast functionality.
- Thermal bank is an attribute of the TemperatureModel class, which is used to store the
  thermal mitigation bank size.
  Ideally, all mitigation banks should be stored as a dedicated parameter class.
- We use simplfied demand allocation logic to estimate the Cannonsville and Pepacton
  reservoir diversion, which works fine. Chung-Yi recommends not to complicate the logic
  and calculation here.

Change Log
----------
Chung-Yi Lin, 2025-05-25, Create the script.
Chung-Yi Lin, 2025-05-28, Fixed logical bugs and verify the correctness of the output.
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

class TemperatureModelLSTM(Parameter):
    def __init__(self, model,
                 model1, model2, Tavg2Tmax_coefs,
                 start_date, end_date, activate_thermal_control,
                 Q_C_lstm_var_name, Q_i_lstm_var_name, cannonsville_storage_pct_lstm_var_name,
                 PywrDRB_ML_plugin_path,
                 thermal_mitigation_bank_size,
                 asycronized_update,
                 debug,
                 **kwargs):
        super().__init__(model, **kwargs)
        """

        """
        self.debug = debug

         # import plugin
        PywrDRB_ML_plugin_path = Path(PywrDRB_ML_plugin_path)
        sys.path.insert(1, PywrDRB_ML_plugin_path)
        from src.lstm_model import WaterTempLSTMModel

        db_TempLSTM = pd.read_csv(PywrDRB_ML_plugin_path / "data/database/TempLSTM_database.csv", index_col=0, parse_dates=True)
        database = db_TempLSTM[start_date: '2023-12-31'] #'1979-01-01'
        self.asycronized_update = asycronized_update
        self.activate_thermal_control = activate_thermal_control

        ml_model = WaterTempLSTMModel(
            model1=model1,
            model2=model2,
            Tavg2Tmax_coefs=Tavg2Tmax_coefs,
            start_date=start_date, end_date=end_date,
            Q_C_lstm_var_name=Q_C_lstm_var_name,
            Q_i_lstm_var_name=Q_i_lstm_var_name,
            cannonsville_storage_pct_lstm_var_name=cannonsville_storage_pct_lstm_var_name,
            thermal_mitigation_bank_size=thermal_mitigation_bank_size,  # mgd
            debug=debug
            )
        ml_model.load_data(database)

        self.ml_model = ml_model

        self.control_algorithm = None  # Placeholder for the control algorithm function

    def set_control_algorithm(self, control_algorithm):
        """
        Set the control algorithm function for the thermal control release decision.

        Parameters
        ----------
        control_algorithm : callable
            A function that takes the LSTM model and other parameters to make the thermal control release decision.
        """
        if callable(control_algorithm):
            self.control_algorithm = control_algorithm
        else:
            raise ValueError("The control_algorithm must be a callable function.")

    def make_control_release(self, Q_C, Q_i, cannonsville_storage_pct, current_date):
        """
        Make the thermal control release decision based on the LSTM model predictions.

        Parameters
        ----------
        Q_C : float
            The Cannonsville reservoir downstream flow (01425000).
        Q_i : float
            The East Branch downstream flow (01417000) and natural inflow to Lordville.
        cannonsville_storage_pct : float
            The percentage of the Cannonsville reservoir storage.
        current_date : pywr.core.CurrentDate
            The current date in the model, used to determine if the LSTM models need to be updated.

        Returns
        -------
        float
            The thermal control release amount in million gallons per day (MGD).
        """
        # activate if self.activate_thermal_control is True
        # Here is the place to plugin control algorithm

        control_algorithm = self.control_algorithm

        thermal_release = control_algorithm(
            ml_model=self.ml_model,
            Q_C=Q_C,
            Q_i=Q_i,
            cannonsville_storage_pct=cannonsville_storage_pct,
            current_date=current_date,
            )
        return thermal_release

    def update(self, Q_C, Q_i, cannonsville_storage_pct, current_date):

        ml_model = self.ml_model

        previous_date = current_date.datetime - timedelta(days=1) # as we are using the previous day flow to update the LSTM
        if previous_date < ml_model.current_date:
            return None

        asycronized_update = self.asycronized_update

        _ = ml_model.update(t=ml_model.t,
                        Q_C=Q_C, Q_i=Q_i, cannonsville_storage_pct=cannonsville_storage_pct,
                        asycronized_update=asycronized_update)
        return None

    def value(self, timestep, scenario_index):
        # The values are retrieved through other parameters like
        # ForecastedTemperatureBeforeThermalRelease and TemperatureAfterThermalRelease
        pass
        return np.nan

    @classmethod
    def load(cls, model, data):
        model1 = data.pop("model1")
        model2 = data.pop("model2")
        Tavg2Tmax_coefs = data.pop("Tavg2Tmax_coefs")
        start_date = data.pop("start_date")
        end_date = data.pop("end_date")
        activate_thermal_control = data.pop("activate_thermal_control", False)
        Q_C_lstm_var_name = data.pop("Q_C_lstm_var_name")
        Q_i_lstm_var_name = data.pop("Q_i_lstm_var_name")
        cannonsville_storage_pct_lstm_var_name = data.pop("cannonsville_storage_pct_lstm_var_name")
        PywrDRB_ML_plugin_path = data.pop("PywrDRB_ML_plugin_path")
        thermal_mitigation_bank_size = data.pop("thermal_mitigation_bank_size")  # mgd
        asycronized_update = data.pop("asycronized_update", False)
        debug = data.pop("debug", False)
        return cls(model, model1, model2, Tavg2Tmax_coefs, start_date, end_date, activate_thermal_control,
                   Q_C_lstm_var_name, Q_i_lstm_var_name, cannonsville_storage_pct_lstm_var_name,
                   PywrDRB_ML_plugin_path, thermal_mitigation_bank_size, asycronized_update, debug, **data)
TemperatureModelLSTM.register()
# temperature_model

class TemperatureModelRF(Parameter):
    def __init__(self, model, start_date, activate_thermal_control, quantile,
                 PywrDRB_ML_plugin_path, asycronized_update, debug, **kwargs):
        super().__init__(model, **kwargs)
        """
        A custom parameter class to predict daily maximum water temperature at Lordville using LSTM models.

        Parameters
        ----------
        model : pywr.core.Model
            The Pywr model object.
        start_date : str
            The start date for the model in "YYYY-MM-DD" format. If None, uses the model's start date.
        PywrDRB_ML_plugin_path : str
            The path to the PywrDRB_ML plugin directory containing the LSTM model configuration.
        debug : bool
            If True, enables debugging mode, which records intermediate values for inspection.
        **kwargs : dict
            Additional keyword arguments for the Parameter class.
        """
        self.debug = debug


        # import plugin
        PywrDRB_ML_plugin_path = Path(PywrDRB_ML_plugin_path)
        sys.path.insert(1, PywrDRB_ML_plugin_path)
        from src.rf_model import WaterTempRandomForestUncertaintyModel

        db_TempLSTM = pd.read_csv(PywrDRB_ML_plugin_path / "data/database/TempLSTM_database.csv", index_col=0, parse_dates=True)
        database = db_TempLSTM[start_date: '2023-12-31'] #'1979-01-01'
        self.asycronized_update = asycronized_update
        self.quantile = quantile
        self.activate_thermal_control = activate_thermal_control

        folder = "RFModels"

        ml_model = WaterTempRandomForestUncertaintyModel(
        rf_model1=PywrDRB_ML_plugin_path / f"models/{folder}/rf_model1.gz",
        rf_model2=PywrDRB_ML_plugin_path / f"models/{folder}/rf_model2.gz",
        rf_model_map=PywrDRB_ML_plugin_path / f"models/{folder}/rf_model_map.gz",
        debug=debug
        )
        ml_model.load_data(database)
        self.ml_model = ml_model

    def make_control_release(self, Q_C, Q_i, cannonsville_storage_pct, current_date):
        """
        Make the thermal control release decision based on the LSTM model predictions.

        Parameters
        ----------
        Q_C : float
            The Cannonsville reservoir downstream flow (01425000).
        Q_i : float
            The East Branch downstream flow (01417000) and natural inflow to Lordville.
        cannonsville_storage_pct : float
            The percentage of the Cannonsville reservoir storage.
        current_date : pywr.core.CurrentDate
            The current date in the model, used to determine if the LSTM models need to be updated.

        Returns
        -------
        float
            The thermal control release amount in million gallons per day (MGD).
        """
        # activate if self.activate_thermal_control is True
        # Here is the place to plugin control algorithm

        control_algorithm = self.control_algorithm
        if callable(control_algorithm) is False:
            raise ValueError("The control_algorithm must be a callable function.")

        thermal_release = control_algorithm(
            model=self,
            Q_C=Q_C,
            Q_i=Q_i,
            cannonsville_storage_pct=cannonsville_storage_pct,
            current_date=current_date.datetime,
            )
        return thermal_release

    def update(self, Q_C, Q_i, cannonsville_storage_pct, current_date):
        """
        Forward the LSTM models to one step.

        Parameters
        ----------
        Q_C : float
            The Cannonsville reservoir downstream flow (01425000).
        Q_i : float
            The East Branch downstream flow (01417000) and natural inflow to Lordville.
        cannonsville_storage_pct : float
            The percentage of the Cannonsville reservoir storage.
        current_date : pywr.core.CurrentDate
            The current date in the model, used to determine if the LSTM models need to be updated.
        """
        debug = self.debug
        ml_model = self.ml_model
        previous_date = current_date.datetime - timedelta(days=1) # as we are using the previous day flow to update the LSTM
        if previous_date < ml_model.current_date:
            return None

        # Update input data
        t = ml_model.t
        ml_model.Q_C[t] = Q_C
        try:
            ml_model.X_1[t, ml_model.rf_model1.x_vars.index("QbcTavg_Q_C")] = Q_C
        except ValueError:
            if debug: print("Warning: 'QbcTavg_Q_C' not found in rf_model1.x_vars. Skipping update.")
        try:
            ml_model.X_2[t, ml_model.rf_model2.x_vars.index("QbcTavg_Q_C")] = Q_C
        except ValueError:
            if debug: print("Warning: 'QbcTavg_Q_C' not found in rf_model2.x_vars. Skipping update.")

        ml_model.Q_i[t] = Q_i
        try:
            ml_model.X_2[t, ml_model.rf_model2.x_vars.index("QbcTavg_Q_i")] = Q_i
        except ValueError:
            if debug: print("Warning: 'QbcTavg_Q_i' not found in rf_model2.x_vars. Skipping update.")

        try:
            ml_model.X_1[t, ml_model.rf_model1.x_vars.index("bc_cannonsville_storage_pct")] = cannonsville_storage_pct
        except ValueError:
            if debug: print("Warning: 'bc_cannonsville_storage_pct' not found in rf_model1.x_vars. Skipping update.")

        if self.asycronized_update is False:
            if previous_date == ml_model.current_date: # avoid double update
                ml_model.update(t=ml_model.t, quantile=self.quantile) # outputing quantile will be very slow
            return None
        else:
            # User can calulate the water temperature after the simulation, which avoids for loop that make the simulation much faster!
            # We will dynamically update the pywrdrb variables dynamically here to the ml_model object.
            # In the control algorithm, user can safely use the update or update until with the internal data (updated) if needed.
            return None

    def value(self, timestep, scenario_index):
        # The values are retrieved through other parameters like
        # ForecastedTemperatureBeforeThermalRelease and TemperatureAfterThermalRelease
        pass
        return np.nan

    @classmethod
    def load(cls, model, data):
        start_date = data.pop("start_date", None)
        quantile = data.pop("quantile", None)
        activate_thermal_control = data.pop("activate_thermal_control", False)
        PywrDRB_ML_plugin_path = data.pop("PywrDRB_ML_plugin_path")
        asycronized_update = data.pop("asycronized_update", False)
        debug = data.pop("debug", False)
        return cls(model, start_date, activate_thermal_control, quantile,
                     PywrDRB_ML_plugin_path, asycronized_update, debug, **data)
TemperatureModelRF.register()
# temperature_model


# Update the TempLSTMs using the flows at previous timestep as the class is called before LP.
class UpdateTemperatureAtLordville(Parameter):
    def __init__(self, model, temperature_model, **kwargs):
        """
        A custom parameter class to update the temperature model at Lordville using the previous flow values.

        Parameters
        ----------
        model : pywr.core.Model
            The Pywr model object.
        temperature_model : TemperatureModel
            The TemperatureModel instance to be updated.
        **kwargs : dict
            Additional keyword arguments for the Parameter class.
        """
        super().__init__(model, **kwargs)
        self.temperature_model = temperature_model
        self.children.add(temperature_model)

    def setup(self):
        super().setup()
        self.link_01425000 = self.model.nodes["link_01425000"] # Cannonsville reservoir downstream flow (01425000)
        self.link_delLordville = self.model.nodes["link_delLordville"] # flow at delLordville
        self.reservoir_cannonsville = self.model.nodes["reservoir_cannonsville"]

        self.children.add(self.link_01425000)
        self.children.add(self.link_delLordville)
        self.children.add(self.reservoir_cannonsville)

    # Need to use prev flow_delLordville and max_flow_catchmentConsumption_delLordville
    # Or get prev_flow from Lordeville node and infer Q_i = Q_L - Q_C
    def value(self, timestep, scenario_index):
        temperature_model = self.temperature_model
        # Cannonsville reservoir downstream flow (01425000)
        Q_C = self.link_01425000.prev_flow[0]
        # East Branch downstream flow (01417000) and natural inflow to Lordville
        Q_i = self.link_delLordville.prev_flow[0] - Q_C
        cannonsville_storage_pct = self.reservoir_cannonsville.volume[0] / 95700 * 100
        temperature_model.update(Q_C, Q_i, cannonsville_storage_pct, timestep)
        return np.nan

    @classmethod
    def load(cls, model, data):
        temperature_model = load_parameter(model, "temperature_model")
        return cls(model, temperature_model, **data)
UpdateTemperatureAtLordville.register()
# update_temperature_at_lordville

class TemperatureAfterThermalRelease(Parameter):
    def __init__(self, model, temperature_model, update_temperature_at_lordville, variable, ml_model_type, **kwargs):
        super().__init__(model, **kwargs)
        """
        A custom parameter class to retrieve the temperature after thermal release at Lordville.

        Parameters
        ----------
        model : pywr.core.Model
            The Pywr model object.
        temperature_model : TemperatureModel
            The TemperatureModel instance to retrieve the temperature from.
        update_temperature_at_lordville : UpdateTemperatureAtLordville
            The UpdateTemperatureAtLordville instance to ensure the temperature model is updated before this parameter.
        variable : str
            The variable to retrieve from the temperature model, either "mu" or "sd".
        **kwargs : dict
            Additional keyword arguments for the Parameter class.
        """
        self.temperature_model = temperature_model
        self.variable = variable
        self.ml_model_type = ml_model_type

        # To ensure update_temperature_at_lordville is run before this parameter.
        self.children.add(update_temperature_at_lordville)

    def value(self, timestep, scenario_index):
        # The forecasted temperature should be populated when making the control release decision.
        # If activate_thermal_control is False, the forecasted temperature will be None.
        if self.ml_model_type == "lstm":
            if self.variable == "mu":
                return self.temperature_model.ml_model.T_L_mu
            elif self.variable == "sd":
                return self.temperature_model.ml_model.T_L_sd
            else:
                raise ValueError("Invalid variable. Must be 'mu' or 'sd'.")
        elif self.ml_model_type == "rf":
            if self.variable == "mu":
                return self.temperature_model.ml_model.T_L
            elif self.variable == "lb":
                return self.temperature_model.ml_model.T_L_lb
            elif self.variable == "ub":
                return self.temperature_model.ml_model.T_L_ub
            else:
                raise ValueError("Invalid variable. Must be 'mu', 'lb', or 'ub.")

    @classmethod
    def load(cls, model, data):
        assert "variable" in data.keys()
        temperature_model = load_parameter(model, "temperature_model")
        update_temperature_at_lordville = load_parameter(model, "update_temperature_at_lordville")
        ml_model_type = data.pop("ml_model_type", "lstm")  # Default to LSTM if not specified
        variable = data.pop("variable")
        return cls(model, temperature_model, update_temperature_at_lordville, variable, ml_model_type, **data)
TemperatureAfterThermalRelease.register()
# temperature_after_thermal_release_mu
# temperature_after_thermal_release_sd (turning off the sd for now)

# Estimated Q is for forecasting purposes (thremal control)
class Estimated_Q_C(Parameter):
    # Cannonsville reservoir release => downstream gauge (01425000) => Lordville
    def __init__(self, model, downstream_release_target_cannonsville,
                 flow_01425000, max_flow_catchmentConsumption_01425000,
                 flow_cannonsville, max_flow_catchmentConsumption_cannonsville,
                 max_flow_delivery_nyc,
                 **kwargs):
        super().__init__(model, **kwargs)
        """
        A custom parameter class to estimate the Cannonsville reservoir downstream flow (Q_C)
        based on the downstream release target and the water balance at the Cannonsville reservoir.

        Parameters
        ----------
        model : pywr.core.Model
            The Pywr model object.
        downstream_release_target_cannonsville : Parameter
            The target downstream release from the Cannonsville reservoir.
        flow_01425000 : Parameter
            The inflow to the downstream gauge (01425000) representing the Cannonsville reservoir downstream flow.
        max_flow_catchmentConsumption_01425000 : Parameter
            The catchment consumption at the downstream gauge (01425000).
        flow_cannonsville : Parameter
            The inflow to the Cannonsville reservoir.
        max_flow_catchmentConsumption_cannonsville : Parameter
            The catchment consumption at the Cannonsville reservoir.
        max_flow_delivery_nyc : Parameter
            The maximum delivery to New York City.
        **kwargs : dict
            Additional keyword arguments for the Parameter class.
        """
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
        self.reservoir_cannonsville = self.model.nodes["reservoir_cannonsville"] # will retrieve the reservoir volume at the previous timestep

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
        """
        A custom parameter class to estimate the East Branch downstream flow + natural
        inflow to Lordville (Q_i).

        parameters
        ----------
        model : pywr.core.Model
            The Pywr model object.
        downstream_release_target_pepacton : Parameter
            The target downstream release from the Pepacton reservoir.
        flow_01417000 : Parameter
            The inflow to the downstream gauge (01417000).
        max_flow_catchmentConsumption_01417000 : Parameter
            The catchment consumption at the downstream gauge (01417000).
        flow_delLordville : Parameter
            The inflow to the Lordville gauge.
        max_flow_catchmentConsumption_delLordville : Parameter
            The catchment consumption at the Lordville gauge.
        flow_pepacton : Parameter
            The inflow to the Pepacton reservoir.
        max_flow_catchmentConsumption_pepacton : Parameter
            The catchment consumption at the Pepacton reservoir.
        max_flow_delivery_nyc : Parameter
            The maximum delivery to New York City.
        **kwargs : dict
            Additional keyword arguments for the Parameter class.
        """
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
        """
        A custom parameter class to calculate the thermal release requirement at Lordville.

        Parameters
        ----------
        model : pywr.core.Model
            The Pywr model object.
        temperature_model : TemperatureModel
            The TemperatureModel instance to retrieve the temperature from.
        update_temperature_at_lordville : UpdateTemperatureAtLordville
            The UpdateTemperatureAtLordville instance to ensure the temperature model is updated before this parameter.
        Q_C : Estimated_Q_C
            The estimated Cannonsville reservoir downstream flow (Q_C).
        Q_i : Estimated_Q_i
            The estimated East Branch downstream flow + natural inflow to Lordville (Q_i).
        **kwargs : dict
            Additional keyword arguments for the Parameter class.
        """
        self.Q_C = Q_C
        self.Q_i = Q_i
        self.thermal_release = 0.0

        # To ensure cannonsville_release & pepacton_release are updated before this parameter
        self.children.add(Q_C)
        self.children.add(Q_i)
        self.children.add(update_temperature_at_lordville) # make sure the temperature model is updated before this parameter using the previous flow values
        self.temperature_model = temperature_model
        self.activate_thermal_control = temperature_model.activate_thermal_control

    def setup(self):
        super().setup()
        self.reservoir_cannonsville = self.model.nodes["reservoir_cannonsville"] # will retrieve the reservoir volume at the previous timestep

    def value(self, timestep, scenario_index):
        temperature_model = self.temperature_model
        # Check if thermal control is activated
        if temperature_model.activate_thermal_control is False:
            return 0.0 # No thermal release
        else:
            thermal_release = temperature_model.make_control_release(
                Q_C=self.Q_C.get_value(scenario_index),
                Q_i=self.Q_i.get_value(scenario_index),
                cannonsville_storage_pct=self.reservoir_cannonsville.volume[0] / 95700 * 100,
                current_date=timestep
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
    def __init__(self, model, temperature_model, thermal_release_requirement, variable, ml_model_type, **kwargs):
        super().__init__(model, **kwargs)
        """
        A custom parameter class to retrieve the forecasted temperature before thermal release at Lordville.

        Parameters
        ----------
        model : pywr.core.Model
            The Pywr model object.
        temperature_model : TemperatureModel
            The TemperatureModel instance to retrieve the forecasted temperature from.
        thermal_release_requirement : ThermalReleaseRequirement
            The ThermalReleaseRequirement instance to ensure the thermal release requirement is calculated before this parameter.
        variable : str
            The variable to retrieve from the temperature model, either "mu" or "sd".
        **kwargs : dict
            Additional keyword arguments for the Parameter class.
        """
        self.temperature_model = temperature_model
        self.variable = variable
        self.ml_model_type = ml_model_type

        # To ensure thermal_release_requirement is run before this parameter.
        self.children.add(thermal_release_requirement)

    def value(self, timestep, scenario_index):
        # The forecasted temperature should be populated when making the control release decision.
        # If activate_thermal_control is False, the forecasted temperature will be None.

        if self.ml_model_type == "lstm":
            if self.variable == "mu":
                forecast_mu = self.temperature_model.ml_model.forecast_T_L_mu_arr
                if isinstance(forecast_mu, float):
                    return forecast_mu
                else:
                    return float(forecast_mu[0]) # Only return nowcast value as value method can only return one value
            elif self.variable == "sd":
                forecast_sd = self.temperature_model.ml_model.forecast_T_L_sd_arr
                if isinstance(forecast_sd, float):
                    return forecast_sd
                else:
                    return float(forecast_sd[0]) # Only return nowcast value as value method can only return one value
            else:
                raise ValueError("Invalid variable. Must be 'mu' or 'sd'.")
        elif self.ml_model_type == "rf":
            if self.variable == "mu":
                return float(self.temperature_model.ml_model.forecast_T_L_arr[0])
            elif self.variable == "lb":
                return float(self.temperature_model.ml_model.forecast_T_L_lb_arr[0])
            elif self.variable == "ub":
                return float(self.temperature_model.ml_model.forecast_T_L_ub_arr[0])
            else:
                raise ValueError("Invalid variable. Must be 'mu', 'lb', or 'ub'.")

    @classmethod
    def load(cls, model, data):
        assert "variable" in data.keys()
        temperature_model = load_parameter(model, "temperature_model")
        thermal_release_requirement = load_parameter(model, "thermal_release_requirement")
        variable = data.pop("variable")
        ml_model_type = data.pop("ml_model_type", "lstm")
        return cls(model, temperature_model, thermal_release_requirement, variable, ml_model_type, **data)
ForecastedTemperatureBeforeThermalRelease.register()
# forecasted_temperature_before_thermal_release_mu
# forecasted_temperature_before_thermal_release_sd (turning off the sd for now)





