"""
Contains the custom parameter classes which use the Salinity LSTM model to predict the
salt front location in the Delaware River Basin (DRB).

Overview
--------
The Salinity LSTM model is developed based on Gorski et al. (2024). We rebuild the model
using the LSTM and BMI sturcture derived from Zwart et al. (2023) to predict 7-day
averaged salt front location in river mile at each timestep.

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
from pywr.parameters import Parameter, load_parameter

from pywrdrb.path_manager import get_pn_object

# Directories (PathNavigator)
# https://github.com/philip928lin/PathNavigator
global pn
pn = get_pn_object()


class SalinityModelLSTM(Parameter):
    def __init__(
        self,
        model,
        model_salinity,
        start_date,
        end_date,
        Q_Trenton_lstm_var_name,
        Q_Schuylkill_lstm_var_name,
        PywrDRB_ML_plugin_path,
        asycronized_update,
        debug,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        """

        """
        self.debug = debug

        # import plugin
        PywrDRB_ML_plugin_path = Path(PywrDRB_ML_plugin_path)
        sys.path.insert(1, PywrDRB_ML_plugin_path)
        from src.lstm_model import SalinityLSTMModel

        # db_SalinityLSTM = pd.read_csv(PywrDRB_ML_plugin_path / "data/database/SalinityLSTM_database.csv", index_col=0, parse_dates=True)
        # database = db_SalinityLSTM[start_date: '2023-12-31'] #'1979-01-01'
        self.asycronized_update = asycronized_update

        ml_model = SalinityLSTMModel(
            model_salinity=model_salinity,
            start_date=start_date,
            end_date=end_date,
            Q_Trenton_lstm_var_name=Q_Trenton_lstm_var_name,
            Q_Schuylkill_lstm_var_name=Q_Schuylkill_lstm_var_name,
            debug=debug,
            disable_tqdm=True,
        )
        ml_model.load_data()

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

    def update(self, Q_Trenton, Q_Schuylkill, current_date):
        ml_model = self.ml_model

        previous_date = current_date.datetime - timedelta(
            days=1
        )  # as we are using the previous day flow to update the LSTM
        if previous_date < ml_model.current_date:
            return None

        asycronized_update = self.asycronized_update

        _ = ml_model.update(
            t=ml_model.t,
            Q_Trenton=Q_Trenton,
            Q_Schuylkill=Q_Schuylkill,
            asycronized_update=asycronized_update,
        )
        return None

    def value(self, timestep, scenario_index):
        # The values are retrieved through other parameters like
        # ForecastedTemperatureBeforeThermalRelease and TemperatureAfterThermalRelease
        pass
        return np.nan

    @classmethod
    def load(cls, model, data):
        model_salinity = data.pop("model_salinity")
        start_date = data.pop("start_date")
        end_date = data.pop("end_date")
        Q_Trenton_lstm_var_name = data.pop("Q_Trenton_lstm_var_name")
        Q_Schuylkill_lstm_var_name = data.pop("Q_Schuylkill_lstm_var_name")
        PywrDRB_ML_plugin_path = data.pop("PywrDRB_ML_plugin_path")
        asycronized_update = data.pop("asycronized_update", False)
        debug = data.pop("debug", False)
        return cls(
            model,
            model_salinity,
            start_date,
            end_date,
            Q_Trenton_lstm_var_name,
            Q_Schuylkill_lstm_var_name,
            PywrDRB_ML_plugin_path,
            asycronized_update,
            debug,
            **data,
        )


SalinityModelLSTM.register()
# salinity_model


class SalinityModelRF(Parameter):
    def __init__(
        self,
        model,
        start_date,
        quantile,
        PywrDRB_ML_plugin_path,
        asycronized_update,
        debug,
        **kwargs,
    ):
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
        from src.rf_model import SaltfrontRandomForestUncertaintyModel

        db_SalinityLSTM = pd.read_csv(
            PywrDRB_ML_plugin_path / "data/database/SalinityLSTM_database.csv",
            index_col=0,
            parse_dates=True,
        )
        database = db_SalinityLSTM[start_date:"2023-12-31"]  #'1979-01-01'
        self.asycronized_update = asycronized_update
        self.quantile = quantile

        folder = "RFModels"

        ml_model = SaltfrontRandomForestUncertaintyModel(
            rf_model_saltfront=PywrDRB_ML_plugin_path
            / f"models/{folder}/rf_model_saltfront.gz",
            debug=debug,
        )
        ml_model.load_data(database)
        self.ml_model = ml_model

    def update(self, Q_Trenton, Q_Schuylkill, current_date):
        ml_model = self.ml_model
        previous_date = current_date.datetime - timedelta(
            days=1
        )  # as we are using the previous day flow to update the LSTM
        if previous_date < ml_model.current_date:
            return None

        # Update input data
        t = ml_model.t

        ml_model.Q_Trenton[t] = Q_Trenton
        try:
            ml_model.X[
                t, ml_model.rf_model_saltfront.x_vars.index("Q_Trenton_bc")
            ] = Q_Trenton
        except ValueError:
            print(
                "Warning: 'Q_Trenton_bc' not found in rf_model_saltfront.x_vars. Skipping update."
            )

        ml_model.Q_Schuylkill[t] = Q_Schuylkill
        try:
            ml_model.X[
                t, ml_model.rf_model_saltfront.x_vars.index("Q_Schuylkill_bc")
            ] = Q_Schuylkill
        except ValueError:
            print(
                "Warning: 'Q_Schuylkill_bc' not found in rf_model_saltfront.x_vars. Skipping update."
            )

        ml_model.Q_Trenton_7darr.append(ml_model.Q_Trenton[t])
        ml_model.Q_Schuylkill_7darr.append(ml_model.Q_Schuylkill[t])
        ml_model.Q_Trenton_7d_avg[t] = np.mean(ml_model.Q_Trenton_7darr)
        ml_model.Q_Schuylkill_7d_avg[t] = np.mean(ml_model.Q_Schuylkill_7darr)

        try:
            ml_model.X[
                t, ml_model.rf_model_saltfront.x_vars.index("Q_Trenton_bc_7d_avg")
            ] = ml_model.Q_Trenton_7d_avg[t]
        except ValueError:
            print(
                "Warning: 'Q_Trenton_bc_7d_avg' not found in rf_model_saltfront.x_vars. Skipping update."
            )
        try:
            ml_model.X[
                t, ml_model.rf_model_saltfront.x_vars.index("Q_Schuylkill_bc_7d_avg")
            ] = ml_model.Q_Schuylkill_7d_avg[t]
        except ValueError:
            print(
                "Warning: 'Q_Schuylkill_bc_7d_avg' not found in rf_model_saltfront.x_vars. Skipping update."
            )

        if self.asycronized_update is False:
            if previous_date == ml_model.current_date:  # avoid double update
                ml_model.update(
                    t=ml_model.t, quantile=self.quantile
                )  # outputing quantile will be very slow
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
        PywrDRB_ML_plugin_path = data.pop("PywrDRB_ML_plugin_path")
        asycronized_update = data.pop("asycronized_update", False)
        debug = data.pop("debug", False)
        return cls(
            model,
            start_date,
            quantile,
            PywrDRB_ML_plugin_path,
            asycronized_update,
            debug,
            **data,
        )


SalinityModelRF.register()
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

        self.num_scenarios = 1

    def setup(self):
        super().setup()  # CRITICAL
        # Discover the number of scenarios pywr will iterate and propagate
        # to the underlying LSTM so its hidden/cell state is sized correctly.
        self.num_scenarios = len(self.model.scenarios.combinations)
        ml_model = self.salinity_model.ml_model
        # ml_model exposes set_n_scenarios on its bmi_lstm; we also resize
        # the wrapper-level state arrays.
        ml_model.lstm.set_n_scenarios(self.num_scenarios)
        ml_model.n_scenarios = self.num_scenarios
        # Re-allocate per-scenario state and debug records to match.
        ml_model.sf_mu = np.full(self.num_scenarios, np.nan)
        ml_model.sf_sd = np.full(self.num_scenarios, np.nan)
        ml_model._Q_Trenton_7d_prev = np.full(self.num_scenarios, np.nan)
        ml_model._Q_Schuylkill_7d_prev = np.full(self.num_scenarios, np.nan)
        ml_model._scenario_X_overrides = {}
        if getattr(ml_model, "debug", False):
            length = ml_model.length
            ml_model.records = {
                "sf_mu": np.full((length, self.num_scenarios), np.nan),
                "sf_sd": np.full((length, self.num_scenarios), np.nan),
                "adj_ratio_Trenton": np.full((length, self.num_scenarios), np.nan),
                "adj_ratio_Montague": np.full((length, self.num_scenarios), np.nan),
                "drought_idx": np.full((length, self.num_scenarios), np.nan),
            }
            ml_model.forecast_records = {
                "sf_mu": np.full((length, self.num_scenarios), np.nan),
                "sf_sd": np.full((length, self.num_scenarios), np.nan),
            }

    def value(self, timestep, scenario_index):
        salinity_model = self.salinity_model
        # Each scenario's per-scenario flow value is what we report back as
        # this parameter's value. The actual LSTM forward pass fires once
        # per timestep, gated on global_id == 0, gathering all scenarios.
        gid = scenario_index.global_id
        Q_Trenton = self.link_delTrenton.prev_flow[gid]
        Q_Schuylkill = self.link_outletSchuylkill.prev_flow[gid]

        if gid == 0:
            if self.num_scenarios == 1:
                # Preserve scalar update path for backward-compatible numerics.
                salinity_model.update(Q_Trenton, Q_Schuylkill, timestep)
            else:
                Q_Trenton_all = np.asarray(
                    self.link_delTrenton.prev_flow, dtype=np.float64
                )
                Q_Schuylkill_all = np.asarray(
                    self.link_outletSchuylkill.prev_flow, dtype=np.float64
                )
                salinity_model.update(Q_Trenton_all, Q_Schuylkill_all, timestep)
        # Cast to plain Python float so pywr's cython-side scalar conversion
        # never sees a numpy 0-d / 1-d array (deprecated since numpy 1.25).
        return float(Q_Trenton)

    @classmethod
    def load(cls, model, data):
        salinity_model = load_parameter(model, "salinity_model")
        return cls(model, salinity_model, **data)


UpdateSaltFrontLocation.register()
# update_salt_front_location


class SaltFrontLocation(Parameter):
    def __init__(
        self,
        model,
        salinity_model,
        update_salt_front_location,
        variable,
        ml_model_type,
        **kwargs,
    ):
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
        self.ml_model_type = ml_model_type

        # To ensure update_salt_front_location is run before this parameter.
        self.children.add(update_salt_front_location)

    def value(self, timestep, scenario_index):
        # The forecasted temperature should be populated when making the control release decision.
        # If activate_thermal_control is False, the forecasted temperature will be None.
        gid = scenario_index.global_id
        if self.ml_model_type == "lstm":
            ml = self.salinity_model.ml_model
            if self.variable == "mu":
                return float(np.asarray(ml.sf_mu).reshape(-1)[gid])
            elif self.variable == "sd":
                return float(np.asarray(ml.sf_sd).reshape(-1)[gid])
            else:
                raise ValueError("Invalid variable. Must be 'mu' or 'sd'.")
        elif self.ml_model_type == "rf":
            if self.variable == "mu":
                return self.salinity_model.ml_model.saltfront
            elif self.variable == "lb":
                return self.salinity_model.ml_model.saltfront_lb
            elif self.variable == "ub":
                return self.salinity_model.ml_model.saltfront_ub
            else:
                raise ValueError("Invalid variable. Must be 'mu', 'lb' or 'ub'.")

    @classmethod
    def load(cls, model, data):
        assert "variable" in data.keys()
        salinity_model = load_parameter(model, "salinity_model")
        update_salt_front_location = load_parameter(model, "update_salt_front_location")
        variable = data.pop("variable")
        ml_model_type = data.pop("ml_model_type", "lstm")
        return cls(
            model,
            salinity_model,
            update_salt_front_location,
            variable,
            ml_model_type,
            **data,
        )


SaltFrontLocation.register()
# salt_front_location_mu
# salt_front_location_sd


class FlowTargetSaltFrontAdjustmentRatio(Parameter):
    def __init__(
        self,
        model,
        salinity_model,
        update_salt_front_location,
        ml_model_type,
        drought_level_agg_nyc,
        flow_target,
        nyc_drought_emergency_level: int = 6,
        **kwargs,
    ):
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
        nyc_drought_emergency_level : int, optional
            ControlCurveIndex value corresponding to NYC Drought Emergency (L5
            in FFMP). Default 6 matches the stock FFMP 6-curve scheme. For
            N-zone configurations this should equal ``n_drought_levels - 1``.
        **kwargs : dict
            Additional keyword arguments for the Parameter class.
        """
        self.salinity_model = salinity_model
        self.ml_model_type = ml_model_type
        self.drought_level_agg_nyc = drought_level_agg_nyc
        self.flow_target = flow_target
        self.nyc_drought_emergency_level = int(nyc_drought_emergency_level)

        # To ensure update_salt_front_location is run before this parameter.
        self.children.add(update_salt_front_location)
        self.children.add(drought_level_agg_nyc)

    def value(self, timestep, scenario_index):
        # The forecasted temperature should be populated when making the control release decision.
        # If activate_thermal_control is False, the forecasted temperature will be None.
        gid = scenario_index.global_id
        drought_level_agg_nyc_idx = self.drought_level_agg_nyc.get_value(scenario_index)
        ml_model = self.salinity_model.ml_model
        ml_model.records["drought_idx"][ml_model.t, gid] = drought_level_agg_nyc_idx

        # BUGFIX (2026-04-20): previously compared to literal 5, which under
        # pywr's ControlCurveIndex semantics corresponds to the L4 band
        # (Drought Warning). Per FFMP (Appendix A, Table 5 heading and Table 2),
        # salt-front-based adjustment of Montague/Trenton flow objectives
        # applies only during Drought Emergency (L5 = index 6 for the default
        # 6-curve scheme), matching the emergency gate in lower_basin_ffmp.py.
        if int(round(drought_level_agg_nyc_idx)) != self.nyc_drought_emergency_level:
            return 1.0  # No adjustment needed if not in drought emergency

        if self.ml_model_type == "lstm":
            sf_mu = float(np.asarray(ml_model.sf_mu).reshape(-1)[gid])
        elif self.ml_model_type == "rf":
            sf_mu = ml_model.saltfront

        month = timestep.month

        # Default in pywrdrb is "Between 87 and 92.5 RM"
        trenton = {
            (12, 1, 2, 3, 4): [1, 1, 0.925925926, 0.925925926],
            (5, 6, 7, 8, 9, 10, 11): [1.074074074, 1, 0.925925926, 0.925925926],
        }

        montague = {
            (12, 1, 2, 3, 4): [1.185185185, 1, 1, 0.814814815],
            (5, 6, 7, 8): [1.03125, 1, 1, 0.6875],
            (9, 10, 11): [1.1, 1, 1, 0.733333333],
        }

        if sf_mu > 92.5:
            idx = 0
        elif sf_mu > 87.0:
            idx = 1
        elif sf_mu > 82.9:
            idx = 2
        else:
            idx = 3

        flow_target = self.flow_target
        if flow_target == "delTrenton":
            for (
                m,
                v,
            ) in trenton.items():
                if month in m:
                    ratio = v[idx]
                    ml_model.records["adj_ratio_Trenton"][ml_model.t, gid] = ratio
                    return ratio
        elif flow_target == "delMontague":
            for (
                m,
                v,
            ) in montague.items():
                if month in m:
                    ratio = v[idx]
                    ml_model.records["adj_ratio_Montague"][ml_model.t, gid] = ratio
                    return ratio
        else:
            raise ValueError("Invalid flow target. Must be 'trenton' or 'montague'.")

    @classmethod
    def load(cls, model, data):
        flow_target = data.pop("flow_target", None)
        ml_model_type = data.pop("ml_model_type", "lstm")
        salinity_model = load_parameter(model, "salinity_model")
        update_salt_front_location = load_parameter(model, "update_salt_front_location")
        drought_level_agg_nyc = load_parameter(model, "drought_level_agg_nyc")
        return cls(
            model,
            salinity_model,
            update_salt_front_location,
            ml_model_type,
            drought_level_agg_nyc,
            flow_target,
            **data,
        )


FlowTargetSaltFrontAdjustmentRatio.register()
# flow_target_salt_front_adjustment_ratio_delTrenton
# flow_target_salt_front_adjustment_ratio_delMontague
