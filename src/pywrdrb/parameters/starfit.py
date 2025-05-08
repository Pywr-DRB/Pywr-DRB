"""
STARFIT-based custom Pywr parameters for reservoir release.

Overview
--------
This module defines a custom Pywr parameter class `STARFITReservoirRelease` that implements 
the STARFIT reservoir operating policy for non-NYC reservoirs, based on the empirical model 
proposed by Turner et al. (2021). The parameter translates seasonal, inflow, and storage 
conditions into daily release decisions, constrained by physical and policy limits.

The STARFIT policy is parameterized using either default calibration values or scenario-based 
samples for sensitivity analysis. It is used to evaluate alternative operating rules and their 
effects on reservoir behavior in the DRB system.

Key Steps
---------
1. Load STARFIT parameters (default or sampled) for each reservoir.
2. Calculate seasonal and hydrologically informed target release values.
3. Enforce physical (capacity) and regulatory (R_min, R_max) constraints on final release.

Technical Notes
---------------
- STARFIT parameter values are loaded from CSV or HDF5 (for sampled scenarios).
- Supports scenario-aware Pywr modeling and runtime parameter loading.
- Designed for integration into PywrDRB models using YAML/JSON configuration.
- Used primarily for DRBC policy exploration and robustness testing.
- Relies on external operational constants, inflow time series, and reservoir capacity.

Links
-----
- https://github.com/Pywr-DRB/Pywr-DRB
- Turner, S.W.D., Steyaert, J.C., Condon, L., & Voisin, N. (2021). 
  Water storage and release policies for all large reservoirs of conterminous United States. 
  Environmental Modelling & Software, 145, 105201. https://doi.org/10.1016/j.envsoft.2021.105201

Change Log
----------
Marilyn Smith, 2025-05-07, Added documentation and cleaned to DRB documentation standard.
"""

import numpy as np
import pandas as pd
import math

from pywr.parameters import Parameter, load_parameter

from pywrdrb.utils.lists import modified_starfit_reservoir_list
from pywrdrb.parameters.lower_basin_ffmp import conservation_releases, max_discharges
from pywrdrb.path_manager import get_pn_object

pn = get_pn_object()

class STARFITReservoirRelease(Parameter):
    """
    STARFIT reservoir release parameter for non-NYC reservoirs.

    Implements the STARFIT rule-based reservoir operation policy described in Turner et al. (2021).
    STARFIT determines seasonal releases using a combination of harmonic (seasonal), storage, and 
    inflow-based terms. Parameters can be either default values or loaded dynamically from 
    scenario samples for sensitivity analysis.

    Attributes
    ----------
    reservoir_name : str
        Reservoir identifier used to access STARFIT parameters.
    node : pywr.nodes.Storage
        The Pywr storage node for the reservoir.
    inflow : Parameter
        Parameter representing catchment inflow to the reservoir.
    run_sensitivity_analysis : bool
        Flag indicating whether to use scenario-based STARFIT parameters.
    sensitivity_analysis_scenarios : list
        Mapping of Pywr scenario index to STARFIT sample scenario ID.
    parameters_loaded : bool
        Tracks whether STARFIT parameters have been initialized.
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
    value(timestep, scenario_index)
        Compute the STARFIT release for a given timestep and scenario.
    load_starfit_sensitivity_samples(sample_scenario_id)
        Load STARFIT samples for a given scenario from HDF5.
    load_default_starfit_params()
        Load default STARFIT parameters from CSV.
    assign_starfit_param_values(starfit_params)
        Parse and assign STARFIT parameters to internal attributes.
    standardize_inflow(inflow)
        Normalize inflow by long-term average.
    calculate_percent_storage(storage)
        Compute percent of reservoir storage capacity.
    get_NORhi(timestep)
        Calculate the upper bound of normal operating range (NOR) for the given day.
    get_NORlo(timestep)
        Calculate the lower bound of normal operating range (NOR) for the given day.
    get_harmonic_release(timestep)
        Compute seasonal release component using harmonic terms.
    calculate_release_adjustment(S_hat, I_hat, NORhi_t, NORlo_t)
        Compute adjustment to seasonal release based on storage and inflow.
    calculate_target_release(harmonic_release, epsilon, NORhi, NORlo, S_hat, I)
        Compute unbounded target release based on policy logic.
    setup()
        Initialize runtime arrays for release values.
    load(model, data)
        Load the parameter in Pywr configuration via YAML.
    """

    def __init__(
        self,
        model,
        reservoir_name,
        storage_node,
        flow_parameter,
        run_starfit_sensitivity_analysis,
        sensitivity_analysis_scenarios,
        **kwargs,
    ):
        super().__init__(model, **kwargs)

        self.node = storage_node
        self.reservoir_name = reservoir_name
        self.inflow = flow_parameter

        # Add children
        self.children.add(flow_parameter)

        # Check if parameters have been loaded
        self.parameters_loaded = False
        # Load the sample scenario IDs
        self.sample_scenario_index = None
        self.run_sensitivity_analysis = run_starfit_sensitivity_analysis
        self.sensitivity_analysis_scenarios = sensitivity_analysis_scenarios

        # Modifications to
        self.remove_R_max = False
        self.linear_below_NOR = False
        use_adjusted_storage = True
        self.WATER_YEAR_OFFSET = 0

    def load_default_starfit_params(self):
        """
        Load default STARFIT parameters from `istarf_conus.csv`.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by reservoir name containing STARFIT calibration parameters.
        """

        return pd.read_csv(
            pn.operational_constants.get_str("istarf_conus.csv"), sep=",", index_col=0
        )

    def load_starfit_sensitivity_samples(self, sample_scenario_id):
        """
        Load STARFIT sensitivity samples from a scenario-specific group in an HDF5 file.

        Parameters
        ----------
        sample_scenario_id : int
            Index of the sensitivity analysis scenario to load.

        Returns
        -------
        pd.DataFrame
            STARFIT parameters for the given scenario ID, indexed by reservoir name.
        """
        samples = f"/starfit/scenario_{sample_scenario_id}"

        # Load the data from the HDF5 file using pandas
        df = pd.read_hdf(
            pn.operational_constants.get_str("scenarios_data.h5"),
            key=samples
            )
        df.set_index("reservoir", inplace=True)

        return df

    def assign_starfit_param_values(self, starfit_params):
        """
        Assign STARFIT parameter values to the reservoir.

        Parameters
        ----------
        starfit_params : pd.DataFrame
            DataFrame containing STARFIT policy parameters. Expected to include either
            Adjusted_* or GRanD_* capacity/flow values, as well as seasonal and operational terms.

        Notes
        -----
        Modifies internal class attributes for release logic and enforces overrides for R_min/R_max
        where DRBC rules apply.
        """
        ## Load STARFIT parameters

        use_adjusted_storage = True

        # Use modified storage parameters for DRBC relevant reservoirs
        if self.name in modified_starfit_reservoir_list:
            self.starfit_name = "modified_" + self.name
        else:
            self.starfit_name = self.reservoir_name

        # Check if parameters are available
        if self.starfit_name not in starfit_params.index:
            print(f"Warning: No STARFIT parameters found for '{self.starfit_name}'.")
            return

        # Pull data from node
        if use_adjusted_storage:
            self.S_cap = starfit_params.loc[self.starfit_name, "Adjusted_CAP_MG"]
            self.I_bar = starfit_params.loc[self.starfit_name, "Adjusted_MEANFLOW_MGD"]

        else:
            self.S_cap = starfit_params.loc[self.starfit_name, "GRanD_CAP_MG"]
            self.I_bar = starfit_params.loc[self.starfit_name, "GRanD_MEANFLOW_MGD"]

        # Store STARFIT parameters
        self.NORhi_mu = starfit_params.loc[self.starfit_name, "NORhi_mu"]
        self.NORhi_min = starfit_params.loc[self.starfit_name, "NORhi_min"]
        self.NORhi_max = starfit_params.loc[self.starfit_name, "NORhi_max"]
        self.NORhi_alpha = starfit_params.loc[self.starfit_name, "NORhi_alpha"]
        self.NORhi_beta = starfit_params.loc[self.starfit_name, "NORhi_beta"]

        self.NORlo_mu = starfit_params.loc[self.starfit_name, "NORlo_mu"]
        self.NORlo_min = starfit_params.loc[self.starfit_name, "NORlo_min"]
        self.NORlo_max = starfit_params.loc[self.starfit_name, "NORlo_max"]
        self.NORlo_alpha = starfit_params.loc[self.starfit_name, "NORlo_alpha"]
        self.NORlo_beta = starfit_params.loc[self.starfit_name, "NORlo_beta"]

        self.Release_alpha1 = starfit_params.loc[self.starfit_name, "Release_alpha1"]
        self.Release_alpha2 = starfit_params.loc[self.starfit_name, "Release_alpha2"]
        self.Release_beta1 = starfit_params.loc[self.starfit_name, "Release_beta1"]
        self.Release_beta2 = starfit_params.loc[self.starfit_name, "Release_beta2"]

        self.Release_c = starfit_params.loc[self.starfit_name, "Release_c"]
        self.Release_p1 = starfit_params.loc[self.starfit_name, "Release_p1"]
        self.Release_p2 = starfit_params.loc[self.starfit_name, "Release_p2"]

        # Override STARFIT max releases at DRBC lower reservoirs
        if self.reservoir_name in list(max_discharges.keys()):
            self.R_max = max_discharges[self.reservoir_name]

        else:
            self.R_max = (
                999999
                if self.remove_R_max
                else (
                    (starfit_params.loc[self.starfit_name, "Release_max"] + 1)
                    * self.I_bar
                )
            )

        # Override STARFIT min releases at DRBC lower reservoirs
        if self.reservoir_name in list(conservation_releases.keys()):
            self.R_min = conservation_releases[self.reservoir_name]
        else:
            self.R_min = (
                starfit_params.loc[self.starfit_name, "Release_min"] + 1
            ) * self.I_bar

    def setup(self):
        """
        Initialize runtime arrays for simulation.

        Notes
        -----
        Called once per Pywr run. Allocates array for storing scenario-specific results.
        """

        super().setup()
        self.N_SCENARIOS = len(self.model.scenarios.combinations)
        self.releases = np.empty([self.N_SCENARIOS], np.float64)

    def standardize_inflow(self, inflow):
        """
        Normalize inflow using long-term mean flow for the reservoir.

        Parameters
        ----------
        inflow : float
            Instantaneous inflow at current timestep (MGD).

        Returns
        -------
        float
            Standardized inflow (unitless).
        """
        return (inflow - self.I_bar) / self.I_bar

    def calculate_percent_storage(self, storage):
        """
        Compute fraction of current storage relative to reservoir capacity.

        Parameters
        ----------
        storage : float
            Reservoir storage at current timestep (MG).

        Returns
        -------
        float
            Percent of storage capacity (0–1).
        """
        return storage / self.S_cap

    def get_NORhi(self, timestep):
        """
        Compute upper bound of the Normal Operating Range (NOR) using harmonic seasonal terms.

        Parameters
        ----------
        timestep : datetime-like
            Current model timestep.

        Returns
        -------
        float
            NORhi value (normalized; 0–1).
        """
        c = math.pi * (timestep.dayofyear + self.WATER_YEAR_OFFSET) / 365
        NORhi = (
            self.NORhi_mu
            + self.NORhi_alpha * math.sin(2 * c)
            + self.NORhi_beta * math.cos(2 * c)
        )
        if (NORhi <= self.NORhi_max) and (NORhi >= self.NORhi_min):
            return NORhi / 100
        elif NORhi > self.NORhi_max:
            return self.NORhi_max / 100
        else:
            return self.NORhi_min / 100

    def get_NORlo(self, timestep):
        """
        Compute lower bound of the Normal Operating Range (NOR) using harmonic seasonal terms.

        Parameters
        ----------
        timestep : datetime-like
            Current model timestep.

        Returns
        -------
        float
            NORlo value (normalized; 0–1).
        """
        c = math.pi * (timestep.dayofyear + self.WATER_YEAR_OFFSET) / 365
        NORlo = (
            self.NORlo_mu
            + self.NORlo_alpha * math.sin(2 * c)
            + self.NORlo_beta * math.cos(2 * c)
        )
        if (NORlo <= self.NORlo_max) and (NORlo >= self.NORlo_min):
            return NORlo / 100
        elif NORlo > self.NORlo_max:
            return self.NORlo_max / 100
        else:
            return self.NORlo_min / 100

    def get_harmonic_release(self, timestep):
        """
        Compute seasonal base release using harmonic Fourier terms.

        Parameters
        ----------
        timestep : datetime-like
            Current model timestep.

        Returns
        -------
        float
            Harmonic component of release (MGD).
        """

        c = math.pi * (timestep.dayofyear + self.WATER_YEAR_OFFSET) / 365
        R_avg_t = (
            self.Release_alpha1 * math.sin(2 * c)
            + self.Release_alpha2 * math.sin(4 * c)
            + self.Release_beta1 * math.cos(2 * c)
            + self.Release_beta2 * math.cos(4 * c)
        )
        return R_avg_t

    def calculate_release_adjustment(self, S_hat, I_hat, NORhi_t, NORlo_t):
        """
        Adjust release based on current standardized storage and inflow.

        Parameters
        ----------
        S_hat : float
            Normalized storage (0–1).
        I_hat : float
            Standardized inflow (unitless).
        NORhi_t : float
            Current upper NOR bound (normalized).
        NORlo_t : float
            Current lower NOR bound (normalized).

        Returns
        -------
        float
            Adjustment factor for release (unitless).
        """
        # Calculate normalized value within NOR
        A_t = (S_hat - NORlo_t) / (NORhi_t)
        epsilon_t = self.Release_c + self.Release_p1 * A_t + self.Release_p2 * I_hat
        return epsilon_t

    def calculate_target_release(
        self, harmonic_release, epsilon, NORhi, NORlo, S_hat, I
    ):
        """
        Calculate target release based on STARFIT logic.

        Parameters
        ----------
        harmonic_release : float
            Base seasonal release from harmonic terms (MGD).
        epsilon : float
            Release adjustment factor from current state (unitless).
        NORhi : float
            Upper bound of NOR (normalized).
        NORlo : float
            Lower bound of NOR (normalized).
        S_hat : float
            Normalized storage (0–1).
        I : float
            Current inflow (MGD).

        Returns
        -------
        float
            Target release before enforcing physical constraints (MGD).
        """

        if (S_hat <= NORhi) and (S_hat >= NORlo):
            target = min(
                (self.I_bar * (harmonic_release + epsilon) + self.I_bar), self.R_max
            )
        elif S_hat > NORhi:
            target = min((self.S_cap * (S_hat - NORhi) + I * 7) / 7, self.R_max)
        else:
            if self.linear_below_NOR:
                target = (self.I_bar * (harmonic_release + epsilon) + self.I_bar) * (
                    S_hat / NORlo
                )  # (1 - (NORlo - S_hat)/NORlo)
                target = max(target, self.R_min)
            else:
                target = self.R_min
        return target

    def value(self, timestep, scenario_index):
        """
        Evaluate STARFIT release at a given timestep and scenario.

        Parameters
        ----------
        timestep : pd.Timestamp
            Model timestep.
        scenario_index : pywr.ScenarioIndex
            Pywr scenario index (including IDs for sensitivity scenarios if enabled).

        Returns
        -------
        float
            Final constrained release (MGD).
        """

        # Check if parameters have been loaded
        if not self.parameters_loaded and self.run_sensitivity_analysis:
            self.pywr_scenario_index = scenario_index
            self.sample_scenario_index = self.sensitivity_analysis_scenarios[
                self.pywr_scenario_index.indices[0]
            ]

            # load values from file
            self.starfit_params = self.load_starfit_sensitivity_samples(
                self.sample_scenario_index
            )

            print(f"Loading STARFIT parameters for {self.reservoir_name}")

            self.assign_starfit_param_values(self.starfit_params)
            # change bool to prevent re-loading
            self.parameters_loaded = True

        elif not self.parameters_loaded and not self.run_sensitivity_analysis:
            self.starfit_params = self.load_default_starfit_params()
            # print(f"Assigning STARFIT parameters for {self.reservoir_name}")
            self.assign_starfit_param_values(self.starfit_params)
            self.parameters_loaded = True

        # Get current storage and inflow conditions
        I_t = self.inflow.get_value(scenario_index)
        S_t = self.node.volume[scenario_index.indices]

        I_hat_t = self.standardize_inflow(I_t)
        S_hat_t = self.calculate_percent_storage(S_t)

        NORhi_t = self.get_NORhi(timestep)
        NORlo_t = self.get_NORlo(timestep)

        seasonal_release_t = self.get_harmonic_release(timestep)

        # Get adjustment from seasonal release
        epsilon_t = self.calculate_release_adjustment(
            S_hat_t, I_hat_t, NORhi_t, NORlo_t
        )

        # Get target release
        target_release = self.calculate_target_release(
            S_hat=S_hat_t,
            I=I_t,
            NORhi=NORhi_t,
            NORlo=NORlo_t,
            epsilon=epsilon_t,
            harmonic_release=seasonal_release_t,
        )

        # Ensure release does not exceed available water and does not exceed storage capacity over time
        release_t = max(min(target_release, I_t + S_t), (I_t + S_t - self.S_cap))

        return max(0, release_t)

    @classmethod
    def load(cls, model, data):
        """Set up the parameter."""
        reservoir_name = data.pop("node")
        storage_node = model.nodes[f"reservoir_{reservoir_name}"]
        flow_parameter = load_parameter(model, f"flow_{reservoir_name}")
        run_starfit_sensitivity_analysis = data.pop("run_starfit_sensitivity_analysis")
        sensitivity_analysis_scenarios = data.pop("sensitivity_analysis_scenarios")
        return cls(
            model,
            reservoir_name,
            storage_node,
            flow_parameter,
            run_starfit_sensitivity_analysis,
            sensitivity_analysis_scenarios,
            **data,
        )


# Register the parameter for use with Pywr
STARFITReservoirRelease.register()
