"""
Parametric reservoir release for different release policies.

Overview
--------
This module defines a parameter class for parametric reservoir releases
based on different release policies, including STARFIT, RBF, and Piecewise Linear (PWL).
The class integrates with the Pywr framework and allows for flexible
configuration of reservoir release strategies.

Key Steps
---------
1. Define the `ParametricRelease` class, inheriting from `pywr.Parameter`.
2. Implement methods to initialize, load, and assign parameter values.
3. Integrate with Pywr's scenario management for sensitivity analysis.
4. Implement release calculation logic based on the selected policy type.
5. Ensure compliance with reservoir constraints and DRBC rules.
6. Provide default parameters and validation mechanisms.
7. Include detailed documentation and usage examples.
8. Facilitate easy extension for additional release policies in the future.

Technical Notes
---------------


Links
-----

Change Log
----------
Marilyn Smith,[date], [description of changes]

"""

import numpy as np
import pandas as pd

from pywr.parameters import Parameter, load_parameter

from pywrdrb.utils.lists import modified_starfit_reservoir_list
from pywrdrb.parameters.lower_basin_ffmp import conservation_releases, max_discharges
from pywrdrb.path_manager import get_pn_object

from pywrdrb.release_policies import RBF, PWL, STARFIT

pn = get_pn_object()

class ParametricRelease(Parameter):
    """
    Class to define parametric reservoir release parameters for different policies.

    Attributes
    ----------
    model : pywr.Model
        The Pywr model instance.
    reservoir_name : str
        Reservoir identifier used to access STARFIT parameters.
    node : pywr.nodes.Storage
        The Pywr storage node for the reservoir.
    policy_type : str
        The type of release policy (e.g., 'RBF', 'PWL', 'STARFIT').
    reservoir_name : str
        The name of the reservoir.
    storage_node : pywr.Node
        The storage node associated with the reservoir.
    inflow : pywr.Parameter
        The inflow parameter for the reservoir.
    run_sensitivity_analysis : bool
        Flag to indicate if sensitivity analysis is to be run.
    sensitivity_analysis_scenarios : list
        List of scenarios for sensitivity analysis.
    policy_id : str
        Identifier for the specific policy configuration.
    policy : AbstractPolicy
        The instantiated policy object based on the selected policy type.
    #TODO: Check if these are needed as attributes
    R_max : float
        Maximum allowable release (MGD).
    R_min : float
        Minimum allowable release (MGD).
    S_cap : float
        Reservoir storage capacity (MG).
    I_bar : float
        Long-term mean inflow (MGD).

    Methods
    -------
    #TODO: Verify the inputs to these methods 
    default_parameters()
        Returns default parameters for the parametric release model.
    assign_param_values(params)
        Assigns parameter values to the reservoir.
    setup()
        Initializes runtime arrays and pre-computes seasonal lookup tables.
    calculate_target_release(S_hat, I)
        Calculates the target release based on current storage and inflow.
    value(timestep, scenario_index)
        Evaluates the release at a given timestep and scenario.
    """

    def __init__(
        self,
        model, 
        reservoir_name,
        storage_node,
        flow_parameter,
        run_sensitivity_analysis,
        sensitivity_analysis_scenarios,
        policy_type 
        policy_id="default",
        **data
    ):
        super().__init__(model, **data)

        # policy type can be RBF/PWL/STARFIT
        self.policy_type = policy_type
        
        if policy_type == 'RBF':
            policy = RBF()
        elif policy_type == 'PWL':
            policy = PWL()
        elif policy_type == 'STARFIT':
            policy = STARFIT()
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

        self.policy = policy
        self.policy_id = policy_id

        self.node = storage_node
        self.inflow = flow_parameter
        self.reservoir_name = reservoir_name
        # Add children
        self.children.add(flow_parameter)

        # Check if parameters have been loaded
        self.parameters_loaded = False
        # Load the sample scenario IDs
        self.sample_scenario_index = None
        self.run_sensitivity_analysis = run_sensitivity_analysis
        self.sensitivity_analysis_scenarios = sensitivity_analysis_scenarios

        # Modifications to
        self.remove_R_max = False
        self.linear_below_NOR = False
        self.use_adjusted_storage = True
        self.WATER_YEAR_OFFSET = 0

        # Placeholders filled by assign_policy_params
        self.S_cap = None
        self.I_bar = None
        self.I_min = None
        self.I_max = None
        self.R_min = None
        self.R_max = None

    @classmethod
    def default_parameters(cls):
        """Return default parameters for the parametric release model."""
        #TODO: load default parameters from csv with caching, indexed by policy type and policy_id
        return 
    
    def assign_param_values(self, params):
        """Assign parameter values to the reservoir

        Parameters
        ----------
        params : dict
            Dictionary of parameter values.

        Notes
        -----
        
        """

    def setup(self):
        """
        Initialize runtime arrays for simulation and pre-compute seasonal lookup tables.

        Notes
        -----
        Called once per Pywr run. Allocates array for storing scenario-specific results
        and pre-computes seasonal values for all days of year.
        """
        super().setup()
        self.N_SCENARIOS = len(self.model.scenarios.combinations)
        self.releases = np.empty([self.N_SCENARIOS], np.float64)


    def _normalize(self, X):
        # clamp into [0,1]
        return np.clip((X - self.x_min) / (self.x_max - self.x_min), 0.0, 1.0)

    def value(self, timestep, scenario_index):
        # 1) get raw states
        S_t = float(self.node.volume)            # MG (storage node)
        I_t = float(self.inflow.get_value(scenario_index))  # MGD (or consistent)

        D_t = float((timestep.dayofyear))  # 1..366

        # 2) normalize
        X = np.array([S_t, I_t, D_t], dtype=float)
        X_norm = self._normalize(X)

        # 3) policy z in [0,1] → preliminary target
        z = self.policy.evaluate(X_norm)
        target = z * self.R_max  # scale by cap (or use different scaling rule if desired)

        # 4) physical feasibility with capacity/mass-balance envelope
        available = I_t + S_t
        min_required = max(0.0, available - self.S_cap)

        release = max(self.R_min, min(target, self.R_max))
        release = min(release, available)
        release = max(release, min_required)
        return max(0.0, release)

    @classmethod
    def load(cls, model, data):
        """Set up the parameter."""
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
            policy_id=policy_id,
            **data,
        )
    
 # Register the parameter for use with Pywr
ParametricRelease.register()