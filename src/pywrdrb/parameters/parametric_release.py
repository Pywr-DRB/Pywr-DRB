"""
Parametric reservoir release for different release policies.

Overview
--------
This module defines a Pywr Parameter that computes reservoir releases using
pluggable, parameterized policies (PWL, RBF, STARFIT). Each policy maps
normalized states [S, I, D] -> z in [0,1]; scaling & constraints are applied
via a shared AbstractPolicy base (set_context + enforce_constraints).

Key Steps
---------
1. Initialize the chosen policy (PWL/RBF/STARFIT) and set its context
   (release_min/max, capacity, input scaling).
2. At each timestep, read storage/inflow, compute day-of-year, and ask the
   policy for a feasible release via `get_release(...)`.
3. Support scenario/sensitivity workflows by (re)assigning parameters and/or
   context at runtime.

Technical Notes
---------------
- Uses `get_policy_context(...)` to build consistent context (limits + min/max
  for normalization) from reservoir metadata and optional overrides.
- Policies must implement the AbstractPolicy contract in your codebase.
- This class avoids duplicating constraint logic; it defers to the policy’s
  `enforce_constraints` using available=S+I.

Change Log
----------
Marilyn Smith, [date], initial integration across PWL/RBF/STARFIT with shared context.
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from pywr.parameters import Parameter, load_parameter

from pywrdrb.path_manager import get_pn_object
from pywrdrb.release_policies.config import get_policy_context
from pywrdrb.release_policies import RBF, PWL, STARFIT  

pn = get_pn_object()


class ParametricRelease(Parameter):
    """
    Parametric reservoir release `Parameter` that delegates to a selected policy.

    Attributes
    ----------
    model : pywr.Model
        The Pywr model instance.
    reservoir_name : str
        Name used to fetch metadata/context.
    node : pywr.nodes.Storage
        The storage node (provides current volume).
    inflow : pywr.Parameter
        Upstream inflow parameter for the reservoir.
    policy_type : str
        'RBF', 'PWL', or 'STARFIT'.
    policy : AbstractPolicy
        The instantiated policy object.
    S_cap, I_min, I_max, R_min, R_max : float or None
        Optional overrides for the operating envelope passed to the policy context.
    """

    # Class-level cache for defaults (reserved)
    _default_params_cache: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        model,
        reservoir_name: str,
        storage_node,
        flow_parameter,
        run_sensitivity_analysis: bool,
        sensitivity_analysis_scenarios,
        policy_type: str,
        policy_params=None,
        policy_id: str = "default",
        # Optional context overrides (can be None)
        R_min: Optional[float] = None,
        R_max: Optional[float] = None,
        S_cap: Optional[float] = None,
        I_min: Optional[float] = None,
        I_max: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(model, **kwargs)

        self.node = storage_node
        self.inflow = flow_parameter
        self.children.add(flow_parameter)

        self.reservoir_name = reservoir_name
        self.policy_id = policy_id
        self.policy_type = policy_type                 # <-- set it!

        # Optional context overrides; if None, get_policy_context will fill from metadata
        self.R_min = R_min
        self.R_max = R_max
        self.S_cap = S_cap
        self.I_min = I_min
        self.I_max = I_max

        self.run_sensitivity_analysis = run_sensitivity_analysis
        self.sensitivity_analysis_scenarios = sensitivity_analysis_scenarios
        self.parameters_loaded = False
        self.sample_scenario_index = None

        # Initialize the policy (instance) and set its operating context
        self._init_policy(policy_params)

    # ---------- Policy/bootstrap ----------
    def _init_policy(self, policy_params):
        """Instantiate the chosen policy and set context from model metadata & overrides."""
        if self.policy_type == "PWL":
            policy = PWL(policy_params=policy_params)
        elif self.policy_type == "RBF":
            policy = RBF(policy_params=policy_params)
        elif self.policy_type == "STARFIT":
            policy = STARFIT(policy_params=policy_params)
        else:
            raise ValueError(f"Invalid policy type: {self.policy_type}")

        # Build context (min/max release; capacity; input scaling) from reservoir metadata + overrides
        ctx = get_policy_context(
            self.reservoir_name,
            release_min_override=self.R_min,
            release_max_override=self.R_max,
            capacity_override=self.S_cap,
            inflow_bounds_override=(self.I_min, self.I_max) if (self.I_min is not None and self.I_max is not None) else None,
        )
        policy.set_context(**ctx)
        # keep resolved values locally for convenience
        self.R_min = policy.release_min
        self.R_max = policy.release_max
        self.S_cap = policy.storage_capacity
        self.I_min, self.I_max = float(policy.x_min[1]), float(policy.x_max[1])

        # Validate optimizer-vector params if present
        if policy.policy_params is not None:
            policy.validate_policy_params()

        self.policy = policy

    # ---------- Optional defaults ----------
    @classmethod
    def default_parameters(cls):
        """Return default parameters for the parametric release model (placeholder)."""
        # Hook for caching CSV/HDF5 defaults keyed by (reservoir, policy_type, policy_id)
        return {}

    # ---------- Re-assignment / sensitivity ----------
    def assign_param_values(self, param_bundle: Dict[str, Any]):
        """
        Update context and/or policy parameters.

        Parameters
        ----------
        param_bundle : dict
            {
              "S_cap": float, "I_min": float, "I_max": float,
              "R_min": float, "R_max": float,
              "policy_params": np.ndarray or list
            }
        """
        # Update overrides (if keys exist)
        for k in ("S_cap", "I_min", "I_max", "R_min", "R_max"):
            if k in param_bundle and param_bundle[k] is not None:
                setattr(self, k, float(param_bundle[k]))

        # If policy params are provided, rebuild the policy with new vector
        new_vec = param_bundle.get("policy_params", None)
        if new_vec is not None:
            # Replace the policy instance entirely so it can re-parse
            if self.policy_type == "PWL":
                self.policy = PWL(policy_params=new_vec)
            elif self.policy_type == "RBF":
                self.policy = RBF(policy_params=new_vec)
            elif self.policy_type == "STARFIT":
                self.policy = STARFIT(policy_params=new_vec)
            else:
                raise ValueError(f"Invalid policy type: {self.policy_type}")

        # Re-apply context (this also handles normalization ranges)
        ctx = get_policy_context(
            self.reservoir_name,
            release_min_override=self.R_min,
            release_max_override=self.R_max,
            capacity_override=self.S_cap,
            inflow_bounds_override=(self.I_min, self.I_max) if (self.I_min is not None and self.I_max is not None) else None,
        )
        self.policy.set_context(**ctx)
        self.parameters_loaded = True

    # ---------- Pywr components ----------
    def setup(self):
        """
        Prepare arrays for scenario results (if needed). Called once per Pywr run.
        """
        super().setup()
        self.N_SCENARIOS = len(self.model.scenarios.combinations) if hasattr(self.model, "scenarios") else 1
        self.releases = np.empty([self.N_SCENARIOS], np.float64)

    # ---------- Core: value ----------
    def value(self, timestep, scenario_index):
        """
        Compute the policy release for a given timestep and scenario.

        Parameters
        ----------
        timestep : pd.Timestamp
            Current timestep.
        scenario_index : int
            Scenario index (Pywr passes this in multi-scenario runs).

        Returns
        -------
        float
            Feasible release for this timestep.
        """
        # 1) Raw states
        S_t = float(self.node.volume)  # MG
        # Pywr Parameters usually expose .value(timestep, scenario_index); if your inflow parameter
        # already gives the scalar via a convenience method, adapt here.
        I_t = float(self.inflow.value(timestep, scenario_index)) if hasattr(self.inflow, "value") \
              else float(self.inflow.get_value(scenario_index))
        D_t = float(timestep.dayofyear)  # 1..366

        # 2) Ask the policy for a release (handles normalization + constraints + S+I availability)
        release = self.policy.get_release(S_t, I_t, D_t)

        # 3) (Optional) Any additional system-wide rules can be applied here if required

        return float(release)

    # ---------- YAML loader ----------
    @classmethod
    def load(cls, model, data):
        """
        Construct from YAML.

        Expected keys in `data`:
          node: <reservoir_name>
          run_sensitivity_analysis: bool
          sensitivity_analysis_scenarios: list
          policy_type: 'RBF'|'PWL'|'STARFIT'
          policy_params: optional list/array
          policy_id: optional str (default 'default')
          R_min, R_max, S_cap, I_min, I_max: optional overrides
        """
        reservoir_name = data.pop("node")
        storage_node = model.nodes[f"reservoir_{reservoir_name}"]
        flow_param_name = data.pop("flow_parameter_name", f"flow_{reservoir_name}")
        flow_parameter = load_parameter(model, flow_param_name)
        run_sensitivity_analysis = data.pop("run_sensitivity_analysis")
        sensitivity_analysis_scenarios = data.pop("sensitivity_analysis_scenarios")
        policy_id = data.pop("policy_id", "default")

        return cls(
            model,
            reservoir_name,
            storage_node,
            flow_parameter,
            run_sensitivity_analysis,
            sensitivity_analysis_scenarios,
            policy_id=policy_id,
            **data, 
        )


# Register the parameter for use with Pywr
ParametricRelease.register()
