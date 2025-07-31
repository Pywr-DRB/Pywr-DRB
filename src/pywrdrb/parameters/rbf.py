"""
Radial Basis Function custom Pywr parameters for reservoir release.

Overview
--------
Custom Pywr parameter that computes reservoir releases based on a surrogate operating policy.

- Uses weighted combinations of radial basis functions centered on state space points to approximate the release function.
- Designed for the Delaware River Basin (DRB) Pywr models.
- Supports scenario-based sensitivity analysis using parameters from CSV or HDF5.
- Enforces policy constraints (e.g., R_min, R_max) and physical bounds.

Inputs:
- Normalized storage (S/S_cap)
- Standardized inflow ((I - I_bar) / I_bar)
- Seasonal indicator (STARFIT: day of year, RBF: normalized week, PWL: storage/inflow breakpoints)


Key Steps
---------
1. **Parameter Setup**: Initializes the parameter with scenario-aware storage for releases.
2. **Policy Parameter Assignment**: Loads and assigns RBF policy parameters.
3. **Policy Evaluation**: Evaluates the release based on current storage, inflow,
    and policy parameters, ensuring compliance with maximum and minimum release limits.
4. **Value Calculation**: Computes the release value for a given timestep and scenario,
    considering inflow and storage conditions.
5. **Parameter Loading**: Supports loading parameters from sensitivity analysis scenarios
    or default values, ensuring flexibility in policy evaluation.

Technical Notes
---------------
This parameter is designed to be used in conjunction with the Pywr framework, specifically
for the Delaware River Basin (DRB) models. It integrates with the STARFIT framework
for evaluating reservoir releases based on radial basis function policies. The parameter
supports scenario-based sensitivity analysis, allowing for dynamic policy evaluation
based on different operational conditions.


Links
-----


Change Log
----------
Marilyn Smith, [Date], Initial implementation of RBFReservoirRelease parameter.
"""

import numpy as np
import pandas as pd
from pywr.parameters import Parameter, load_parameter
from pywrdrb.path_manager import get_pn_object
from pywrdrb.parameters.lower_basin_ffmp import conservation_releases, max_discharges

pn = get_pn_object()

class RBFReservoirRelease(Parameter):
    """
    RBF-based reservoir release parameter for DRB Pywr models.

    Implements scenario-aware parameter loading and evaluation consistent with STARFIT integration.
    Includes inflow standardization by long-term mean inflow (I_bar).
    """
    _default_params_cache = None

    def __init__(
        self,
        model,
        reservoir_name,
        storage_node,
        flow_parameter,
        run_sensitivity_analysis,
        sensitivity_analysis_scenarios,
        policy_id,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.node = storage_node
        self.reservoir_name = reservoir_name
        self.inflow = flow_parameter
        self.children.add(flow_parameter)

        self.run_sensitivity_analysis = run_sensitivity_analysis
        self.sensitivity_analysis_scenarios = sensitivity_analysis_scenarios
        self.policy_id = policy_id
        self.parameters_loaded = False

        # Modifications to
        self.remove_R_max = False

    @classmethod
    def load_default_params(cls):
        """
        Load default piecewise linear parameters from a CSV file.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by reservoir name containing PWL calibration parameters.

        """
        if cls._default_params_cache is None:
            cls._default_params_cache = pd.read_csv(
                pn.operational_constants.get_str("rbf.csv"), 
                sep=",", 
                index_col=["reservoir", "policy_id"]
            )
        return cls._default_params_cache
    
    def assign_policy_params(self, rbf_params):
        """
        Parse and store RBF policy parameters.
        """
        key = (self.reservoir_name, self.policy_id)

        if key not in rbf_params.index:
            fallback_key = (self.reservoir_name, "default")
            if fallback_key in rbf_params.index:
                print(f"Warning: policy_id '{self.policy_id}' not found. Falling back to 'default'.")
                key = fallback_key
            else:
                raise KeyError(f"RBF parameters not found for '{self.reservoir_name}' with policy_id '{self.policy_id}' or 'default'.")

        print(f"[RBF] Loaded parameters for {self.reservoir_name}, policy_id = {self.policy_id}")

        policy_params = rbf_params.loc[key]

        # Extract and assemble RBF data
        centers = []
        widths = []
        weights = []

        for i in range(1, 10):  # support up to 9 RBFs (adjust as needed)
            try:
                c = [
                    policy_params[f"rbf{i}_center_storage"],
                    policy_params[f"rbf{i}_center_inflow"],
                    policy_params[f"rbf{i}_center_doy"]
                ]
                s = [
                    policy_params[f"rbf{i}_scale_storage"],
                    policy_params[f"rbf{i}_scale_inflow"],
                    policy_params[f"rbf{i}_scale_doy"]
                ]
                w = policy_params[f"rbf{i}_weight"]

                centers.append(c)
                widths.append(s)
                weights.append(w)
            except KeyError:
                break  # Stop at first missing group

        self.centers = np.array(centers)
        self.widths = np.maximum(np.array(widths), 1e-6)
        self.weights = np.array(weights) / np.sum(weights)

        # Capacity and inflow mean
        self.S_cap = policy_params["Adjusted_CAP_MG"]
        self.I_bar = policy_params["Adjusted_MEANFLOW_MGD"]

        # Release limits
        # Override RBF max releases at DRBC lower reservoirs
        if self.reservoir_name in max_discharges:
            self.R_max = max_discharges[self.reservoir_name]
        else:
            self.R_max = (
                999999
                if self.remove_R_max
                else (policy_params["Release_max"] + 1) * self.I_bar
            )

        # Override STARFIT min releases at DRBC lower reservoirs
        if self.reservoir_name in conservation_releases:
            self.R_min = conservation_releases[self.reservoir_name]
        else:
            self.R_min = (policy_params["Release_min"] + 1) * self.I_bar

    def evaluate_policy(self, S_norm, I_std, W_norm):
        """
        Evaluate the RBF function using normalized scalar inputs for a single scenario.

        Parameters
        ----------
        S_norm : float
            Normalized storage.
        I_std : float
            Standardized inflow.
        W_norm : float
            Normalized day-of-year indicator (0–1).

        Returns
        -------
        float
            Release fraction (between 0 and 1).

        """
        X = np.array([S_norm, I_std, W_norm])
        z = 0.0

        for i in range(len(self.weights)):
            sq_term = np.sum(((X - self.centers[i]) / self.widths[i]) ** 2)
            z += self.weights[i] * np.exp(-sq_term)
        
        return np.clip(z, 0.0, 1.0)

    def value(self, timestep, scenario_index):
        if not self.parameters_loaded:
            if self.run_sensitivity_analysis:
                self.sample_scenario_index = self.sensitivity_analysis_scenarios[
                    scenario_index.indices[0]
                ]
                all_params = self.load_sensitivity_samples(
                self.sample_scenario_index, "rbf"
            )
                self.policy_params = all_params.loc[self.policy_id]
            else:
                self.rbf_params = self.load_default_params()

            self.assign_policy_params(self.rbf_params)
            self.parameters_loaded = True

        # Get scalar inputs
        S_t = float(self.node.volume[scenario_index.indices])
        I_t = float(self.inflow.get_value(scenario_index))
        week_of_year_norm = timestep.dayofyear / 365.0

        # Normalize storage
        S_norm = S_t / self.S_cap
        # Standardize inflow using I_bar
        I_std = (I_t - self.I_bar) / self.I_bar

        # Evaluate
        release_fraction = self.evaluate_policy(S_norm, I_std, week_of_year_norm)
        raw_release = release_fraction * self.R_max

        # Physical constraints
        available_water = I_t + S_t
        min_required = available_water - self.S_cap
        release_t = max(min(raw_release, available_water), min_required)

        return max(self.R_min, release_t)

    @classmethod
    def load(cls, model, data):
        reservoir_name = data.pop("node")
        storage_node = model.nodes[f"reservoir_{reservoir_name}"]
        flow_parameter = load_parameter(model, f"flow_{reservoir_name}")
        run_sensitivity_analysis = data.pop("run_sensitivity_analysis")
        sensitivity_analysis_scenarios = data.pop("sensitivity_analysis_scenarios")
        policy_id = data.pop("policy_id")
        return cls(
            model,
            reservoir_name,
            storage_node,
            flow_parameter,
            run_sensitivity_analysis,
            sensitivity_analysis_scenarios,
            policy_id,
            **data,
        )

# Register with Pywr
RBFReservoirRelease.register()
