"""
Piecewise Linear custom Pywr parameters for reservoir release.

Overview
--------
Custom Pywr parameter that computes reservoir releases based on a surrogate operating policy.

- Breaks the state space into linear segments, specifying reservoir release as a function of storage or inflow using connected line segments.
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
2. **Policy Parameter Assignment**: Loads and assigns piecewise linear policy parameters.
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
for evaluating reservoir releases based on piecewise linear policies. The parameter
supports scenario-based sensitivity analysis, allowing for dynamic policy evaluation
based on different operational conditions.


Links
-----


Change Log
----------
Marilyn Smith, [Date], Initial implementation of PWLReservoirRelease parameter.
"""
import numpy as np
import pandas as pd
from pywr.parameters import Parameter, load_parameter

from pywrdrb.parameters.lower_basin_ffmp import conservation_releases, max_discharges
from pywrdrb.path_manager import get_pn_object

pn = get_pn_object()

class PWLReservoirRelease(Parameter):
    """
    Piecewise Linear reservoir release parameter for DRB Pywr models.

    Implements scenario-aware parameter loading and piecewise linear policy 
    evaluation for release decisions, consistent with STARFIT integration.

    Supports:
    - Storage-normalized input
    - Inflow-standardized input (by mean inflow)
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
        policy_id="default",
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
                pn.operational_constants.get_str("pwl.csv"), 
                sep=",", 
                index_col=["reservoir", "policy_id"]
            )
        return cls._default_params_cache

    @staticmethod
    def _reconstruct_intercepts(breakpoints, slopes):
        intercepts = [0.0]  # first intercept = 0
        for i in range(1, len(slopes)):
            dx = breakpoints[i] - breakpoints[i - 1]
            prev_y = slopes[i - 1] * dx + intercepts[i - 1]
            intercepts.append(prev_y)
        return intercepts

    def assign_policy_params(self, pwl_params):
        """
        Parse and store PWL policy parameters.
        """
        key = (self.reservoir_name, self.policy_id)

        if key not in pwl_params.index:
            fallback_key = (self.reservoir_name, "default")
            if fallback_key in pwl_params.index:
                print(f"Warning: policy_id '{self.policy_id}' not found. Falling back to 'default'.")
                key = fallback_key
            else:
                raise KeyError(f"PWL parameters not found for '{self.reservoir_name}' with policy_id '{self.policy_id}' or 'default'.")

        print(f"[PWL] Loaded parameters for {self.reservoir_name}, policy_id = {self.policy_id}")

        policy_params = pwl_params.loc[key]

        # === Reconstruct storage breakpoints and slopes ===
        self.storage_breakpoints = [0.0, policy_params["storage_x1"], policy_params["storage_x2"], 1.0]
        self.storage_slopes = [policy_params["storage_theta1"], policy_params["storage_theta2"], policy_params["storage_theta3"]]
        self.storage_intercepts = self._reconstruct_intercepts(self.storage_breakpoints, self.storage_slopes)

        # === Reconstruct inflow breakpoints and slopes ===
        self.inflow_breakpoints = [0.0, policy_params["inflow_x1"], policy_params["inflow_x2"], 1.0]
        self.inflow_slopes = [policy_params["inflow_theta1"], policy_params["inflow_theta2"], policy_params["inflow_theta3"]]
        self.inflow_intercepts = self._reconstruct_intercepts(self.inflow_breakpoints, self.inflow_slopes)

        # Capacity and inflow mean
        self.S_cap = policy_params["Adjusted_CAP_MG"]
        self.I_bar = policy_params["Adjusted_MEANFLOW_MGD"]

        # Release limits
        # Override STARFIT max releases at DRBC lower reservoirs
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

    def _eval_piecewise(self, x, breakpoints, slopes, intercepts):
        """
        Evaluate a single piecewise-linear curve.
        """
        for i in range(len(breakpoints) - 1):
            if breakpoints[i] <= x < breakpoints[i + 1]:
                dx = x - breakpoints[i]
                return slopes[i] * dx + intercepts[i]
        return intercepts[-1]

    def evaluate_policy(self, S_t, I_t):
        """
        Evaluate combined PWL release given storage and inflow.
        """
        # Normalize inputs
        S_norm = S_t / self.S_cap
        I_std = (I_t - self.I_bar) / self.I_bar

        # Evaluate both components
        zS = self._eval_piecewise(S_norm, self.storage_breakpoints, self.storage_slopes, self.storage_intercepts)
        zI = self._eval_piecewise(I_std, self.inflow_breakpoints, self.inflow_slopes, self.inflow_intercepts)

        # Combine (simple average like controls project)
        z = (zS + zI) / 2.0

        # Enforce [0,1] bounds
        z = max(0.0, min(1.0, z))
        return z

    def value(self, timestep, scenario_index):
        if not self.parameters_loaded:
            if self.run_sensitivity_analysis:
                self.sample_scenario_index = self.sensitivity_analysis_scenarios[
                    scenario_index.indices[0]
                ]
                all_params = self.load_sensitivity_samples(
                    self.sample_scenario_index, "pwl"
                )
                self.policy_params = all_params.loc[self.policy_id]
            else:
                self.pwl_params = self.load_default_params()
            
            self.assign_policy_params(self.pwl_params)
            self.parameters_loaded = True

        S_t = self.node.volume[scenario_index.indices]
        I_t = self.inflow.get_value(scenario_index)

        # Evaluate normalized policy
        release_fraction = self.evaluate_policy(S_t, I_t)
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
PWLReservoirRelease.register()
