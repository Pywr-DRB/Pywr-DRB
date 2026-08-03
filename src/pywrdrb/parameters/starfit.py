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
- Turner, S.W.D., Steyaert, J.C., Condon, L., & Voisin, N. (2021). 
  Water storage and release policies for all large reservoirs of conterminous United States. 
  Environmental Modelling & Software, 145, 105201. https://doi.org/10.1016/j.envsoft.2021.105201

Change Log
----------
Marilyn Smith, 2025-05-07, Added documentation and cleaned to DRB documentation standard.
TJA, 2025-06-11, Performance optimizations while maintaining identical functionality.
TJA, 2026-07-10, Added starfit_params_filename option for custom parameter CSVs.
TJA, 2026-07-22, Enforce R_min in all storage conditions (was below-NOR only).
TJA, 2026-08, Added compute_starfit_release_step canonical helper; inlined value() hot path.
"""

import os
import numpy as np
import pandas as pd
import math
from functools import lru_cache

from pywr.parameters import Parameter, load_parameter

from pywrdrb.utils.lists import modified_starfit_reservoir_list
from pywrdrb.parameters.lower_basin_ffmp import conservation_releases, max_discharges
from pywrdrb.path_manager import get_pn_object

pn = get_pn_object()


def compute_starfit_release_step(
    *,
    I_t: float,
    S_t: float,
    S_cap: float,
    I_bar: float,
    R_min: float,
    R_max: float,
    Release_c: float,
    Release_p1: float,
    Release_p2: float,
    harmonic_release: float,
    NORhi: float,
    NORlo: float,
    inv_S_cap: float,
    inv_I_bar: float,
    linear_below_NOR: bool = False,
) -> float:
    """
    One STARFIT release step — the canonical scalar release equation.

    This pure function is the source of truth for the STARFIT release math
    used by the online ``STARFITReservoirRelease.value`` parameter. It
    mirrors Turner et al. (2021) with the DRB modification that ``R_min``
    is enforced in all storage conditions (not only below the NOR).

    The online parameter inlines this body (rather than calling the helper)
    because ``value`` is invoked once per (timestep × scenario × reservoir)
    and the Python call overhead is non-trivial; the helper exists so there
    is exactly one canonical implementation that downstream code and the
    test suite can verify against.

    Parameters
    ----------
    I_t : float
        Gross inflow at this timestep (MGD).
    S_t : float
        Reservoir storage at the start of this timestep (MG).
    S_cap : float
        Reservoir storage capacity (MG).
    I_bar : float
        Long-term mean inflow (MGD), used to standardize ``I_t``.
    R_min, R_max : float
        Minimum and maximum daily release (MGD).
    Release_c, Release_p1, Release_p2 : float
        STARFIT release-adjustment coefficients (Turner Eq. 5).
    harmonic_release : float
        Day-of-year harmonic release component (precomputed lookup).
    NORhi, NORlo : float
        Upper and lower bounds of the Normal Operating Range (NOR) for this
        day-of-year, expressed as a fraction of capacity.
    inv_S_cap, inv_I_bar : float
        Precomputed reciprocals (1 / S_cap, 1 / I_bar). Passed in rather
        than computed here because both call sites cache them.
    linear_below_NOR : bool, default False
        If True, the below-NOR target ramps linearly from R_min toward the
        in-NOR target. If False, releases below NOR are clamped to R_min.

    Returns
    -------
    float
        Final constrained release (MGD), guaranteed non-negative and
        bounded by available water and remaining storage capacity.
    """
    # Standardize inflow and percent storage.
    I_hat = (I_t - I_bar) * inv_I_bar
    S_hat = S_t * inv_S_cap

    # Release adjustment.
    A = (S_hat - NORlo) / NORhi
    epsilon = Release_c + Release_p1 * A + Release_p2 * I_hat

    # Target release.
    if NORlo <= S_hat <= NORhi:
        target = min(I_bar * (harmonic_release + epsilon + 1), R_max)
    elif S_hat > NORhi:
        target = min((S_cap * (S_hat - NORhi) + I_t * 7) / 7, R_max)
    else:
        if linear_below_NOR:
            target = (I_bar * (harmonic_release + epsilon + 1)) * (S_hat / NORlo)
        else:
            target = R_min

    # Enforce the minimum release in all storage conditions (previously
    # only applied below the NOR, allowing the in-NOR release to dip
    # below R_min on very dry days). Physical availability is still
    # enforced by the bounds below.
    target = max(target, R_min)

    # Constrain by available water + capacity.
    available_water = I_t + S_t
    min_required = available_water - S_cap
    release = max(min(target, available_water), min_required)
    return max(0.0, release)


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

    # Class-level cache for default parameters (shared across instances)
    _default_params_cache = None
    # Class-level cache for custom parameter files, keyed by (abspath, mtime)
    # so overwriting a file in a live session invalidates its cached entry
    _custom_params_cache = {}

    def __init__(
        self,
        model,
        reservoir_name,
        storage_node,
        flow_parameter,
        run_starfit_sensitivity_analysis,
        sensitivity_analysis_scenarios,
        starfit_params_filename=None,
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
        # Optional alternative parameter CSV (same format as istarf_conus.csv)
        self.starfit_params_filename = starfit_params_filename

        # Modifications to
        self.remove_R_max = False
        self.linear_below_NOR = False
        self.use_adjusted_storage = True
        self.WATER_YEAR_OFFSET = 0

        # Pre-computed seasonal lookup tables (initialized in setup)
        self._seasonal_lookup = None
        self._nor_hi_lookup = None
        self._nor_lo_lookup = None

        # Pre-computed constants
        self._pi_over_365 = math.pi / 365
        self._two_pi_over_365 = 2 * math.pi / 365
        self._four_pi_over_365 = 4 * math.pi / 365

    @classmethod
    def load_default_starfit_params(cls):
        """
        Load default STARFIT parameters from `istarf_conus.csv` with caching.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by reservoir name containing STARFIT calibration parameters.
        """
        if cls._default_params_cache is None:
            cls._default_params_cache = pd.read_csv(
                pn.operational_constants.get_str("istarf_conus.csv"),
                sep=",",
                index_col=0
            )
        return cls._default_params_cache

    @classmethod
    def load_custom_starfit_params(cls, filename):
        """
        Load STARFIT parameters from a custom CSV file with caching.

        The CSV must follow the same format as `istarf_conus.csv` (indexed by
        reservoir name, containing all rows needed by the model). The cache is
        keyed by (absolute path, modification time) so overwriting the file
        invalidates the cached entry.

        Parameters
        ----------
        filename : str
            Path to the custom STARFIT parameter CSV.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by reservoir name containing STARFIT calibration parameters.
        """
        key = (os.path.abspath(filename), os.path.getmtime(filename))
        if key not in cls._custom_params_cache:
            cls._custom_params_cache[key] = pd.read_csv(
                filename,
                sep=",",
                index_col=0,
            )
        return cls._custom_params_cache[key]

    @classmethod
    def clear_starfit_cache(cls):
        """Clear cached default and custom STARFIT parameter tables."""
        cls._default_params_cache = None
        cls._custom_params_cache = {}

    @lru_cache(maxsize=128)
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
        # Use modified storage parameters for DRBC relevant reservoirs
        self.starfit_name = (
            "modified_" + self.reservoir_name 
            if self.reservoir_name in modified_starfit_reservoir_list 
            else self.reservoir_name
        )

        # Check if parameters are available
        if self.starfit_name not in starfit_params.index:
            raise ValueError(
                f"No STARFIT parameters found for '{self.starfit_name}' "
                f"(reservoir: {self.reservoir_name}). "
                "Check that the parameter CSV contains this row."
            )

        params = starfit_params.loc[self.starfit_name]

        # Pull data from node
        if self.use_adjusted_storage:
            self.S_cap = params["Adjusted_CAP_MG"]
            self.I_bar = params["Adjusted_MEANFLOW_MGD"]
        else:
            self.S_cap = params["GRanD_CAP_MG"]
            self.I_bar = params["GRanD_MEANFLOW_MGD"]

        # Pre-compute derived constants
        self._inv_S_cap = 1.0 / self.S_cap
        self._inv_I_bar = 1.0 / self.I_bar

        # Store STARFIT parameters with vectorized access
        self.NORhi_mu = params["NORhi_mu"]
        self.NORhi_min = params["NORhi_min"] / 100
        self.NORhi_max = params["NORhi_max"] / 100
        self.NORhi_alpha = params["NORhi_alpha"]
        self.NORhi_beta = params["NORhi_beta"]

        self.NORlo_mu = params["NORlo_mu"]
        self.NORlo_min = params["NORlo_min"] / 100
        self.NORlo_max = params["NORlo_max"] / 100
        self.NORlo_alpha = params["NORlo_alpha"]
        self.NORlo_beta = params["NORlo_beta"]

        self.Release_alpha1 = params["Release_alpha1"]
        self.Release_alpha2 = params["Release_alpha2"]
        self.Release_beta1 = params["Release_beta1"]
        self.Release_beta2 = params["Release_beta2"]

        self.Release_c = params["Release_c"]
        self.Release_p1 = params["Release_p1"]
        self.Release_p2 = params["Release_p2"]

        # Override STARFIT max releases at DRBC lower reservoirs
        if self.reservoir_name in max_discharges:
            self.R_max = max_discharges[self.reservoir_name]
        else:
            self.R_max = (
                999999
                if self.remove_R_max
                else (params["Release_max"] + 1) * self.I_bar
            )

        # Override STARFIT min releases at DRBC lower reservoirs
        if self.reservoir_name in conservation_releases:
            self.R_min = conservation_releases[self.reservoir_name]
        else:
            self.R_min = (params["Release_min"] + 1) * self.I_bar

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

    def _precompute_seasonal_lookups(self):
        """Pre-compute seasonal values for all days of year to avoid repeated calculations."""
        if self._seasonal_lookup is not None:
            return  # Already computed

        days = np.arange(1, 367)  # Day of year 1-366
        c_values = self._pi_over_365 * (days + self.WATER_YEAR_OFFSET)
        
        # Pre-compute trig values
        sin_2c = np.sin(2 * c_values)
        sin_4c = np.sin(4 * c_values)
        cos_2c = np.cos(2 * c_values)
        cos_4c = np.cos(4 * c_values)
        
        # Harmonic release lookup
        self._seasonal_lookup = (
            self.Release_alpha1 * sin_2c +
            self.Release_alpha2 * sin_4c +
            self.Release_beta1 * cos_2c +
            self.Release_beta2 * cos_4c
        )
        
        # NOR bounds lookup
        nor_hi_raw = (
            self.NORhi_mu +
            self.NORhi_alpha * sin_2c +
            self.NORhi_beta * cos_2c
        )
        self._nor_hi_lookup = np.clip(nor_hi_raw, self.NORhi_min * 100, self.NORhi_max * 100) / 100
        
        nor_lo_raw = (
            self.NORlo_mu +
            self.NORlo_alpha * sin_2c +
            self.NORlo_beta * cos_2c
        )
        self._nor_lo_lookup = np.clip(nor_lo_raw, self.NORlo_min * 100, self.NORlo_max * 100) / 100

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
        return (inflow - self.I_bar) * self._inv_I_bar

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
        return storage * self._inv_S_cap

    def get_NORhi(self, timestep):
        """
        Compute upper bound of the Normal Operating Range (NOR) using pre-computed lookup.

        Parameters
        ----------
        timestep : datetime-like
            Current model timestep.

        Returns
        -------
        float
            NORhi value (normalized; 0–1).
        """
        if self._nor_hi_lookup is None:
            self._precompute_seasonal_lookups()
        return self._nor_hi_lookup[timestep.dayofyear - 1]

    def get_NORlo(self, timestep):
        """
        Compute lower bound of the Normal Operating Range (NOR) using pre-computed lookup.

        Parameters
        ----------
        timestep : datetime-like
            Current model timestep.

        Returns
        -------
        float
            NORlo value (normalized; 0–1).
        """
        if self._nor_lo_lookup is None:
            self._precompute_seasonal_lookups()
        return self._nor_lo_lookup[timestep.dayofyear - 1]

    def get_harmonic_release(self, timestep):
        """
        Compute seasonal base release using pre-computed lookup table.

        Parameters
        ----------
        timestep : datetime-like
            Current model timestep.

        Returns
        -------
        float
            Harmonic component of release (MGD).
        """
        if self._seasonal_lookup is None:
            self._precompute_seasonal_lookups()
        return self._seasonal_lookup[timestep.dayofyear - 1]

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
        A_t = (S_hat - NORlo_t) / NORhi_t
        return self.Release_c + self.Release_p1 * A_t + self.Release_p2 * I_hat

    def calculate_target_release(self, harmonic_release, epsilon, NORhi, NORlo, S_hat, I):
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
        if NORlo <= S_hat <= NORhi:
            target = min(
                self.I_bar * (harmonic_release + epsilon + 1),
                self.R_max
            )
        elif S_hat > NORhi:
            target = min((self.S_cap * (S_hat - NORhi) + I * 7) / 7, self.R_max)
        else:
            if self.linear_below_NOR:
                target = (self.I_bar * (harmonic_release + epsilon + 1)) * (S_hat / NORlo)
            else:
                target = self.R_min
        # Enforce the minimum release in all storage conditions (previously
        # only applied below the NOR, allowing the in-NOR release to dip
        # below R_min on very dry days). Physical availability is still
        # enforced in value().
        return max(target, self.R_min)

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

        Notes
        -----
        This method is performance-critical and called thousands of times per simulation.
        Helper methods are inlined to reduce Python function call overhead (~17% speedup).
        """
        # Lazy parameter loading with caching (only runs once per simulation)
        if not self.parameters_loaded:
            if self.run_sensitivity_analysis:
                self.pywr_scenario_index = scenario_index
                self.sample_scenario_index = self.sensitivity_analysis_scenarios[
                    self.pywr_scenario_index.indices[0]
                ]
                self.starfit_params = self.load_starfit_sensitivity_samples(
                    self.sample_scenario_index
                )
                print(f"Loading STARFIT parameters for {self.reservoir_name}")
            elif self.starfit_params_filename is not None:
                self.starfit_params = self.load_custom_starfit_params(
                    self.starfit_params_filename
                )
            else:
                self.starfit_params = self.load_default_starfit_params()

            self.assign_starfit_param_values(self.starfit_params)
            self.parameters_loaded = True

            # Ensure seasonal lookups are pre-computed
            if self._seasonal_lookup is None:
                self._precompute_seasonal_lookups()

        # Get current storage and inflow conditions
        I_t = self.inflow.get_value(scenario_index)
        S_t = self.node.volume[scenario_index.global_id]

        # NOTE: The body below is the inlined form of
        # ``pywrdrb.parameters.starfit.compute_starfit_release_step``. The
        # helper is the canonical spec; this hot path inlines it for the
        # ~17% per-call speedup. Any change to the math must be applied to
        # both — the test suite asserts equivalence.
        # Inlined: standardize_inflow and calculate_percent_storage
        I_hat_t = (I_t - self.I_bar) * self._inv_I_bar
        S_hat_t = S_t * self._inv_S_cap

        # Inlined: get_NORhi, get_NORlo, get_harmonic_release using lookup tables
        day_idx = timestep.dayofyear - 1
        NORhi_t = self._nor_hi_lookup[day_idx]
        NORlo_t = self._nor_lo_lookup[day_idx]
        seasonal_release_t = self._seasonal_lookup[day_idx]

        # Inlined: calculate_release_adjustment
        A_t = (S_hat_t - NORlo_t) / NORhi_t
        epsilon_t = self.Release_c + self.Release_p1 * A_t + self.Release_p2 * I_hat_t

        # Inlined: calculate_target_release
        if NORlo_t <= S_hat_t <= NORhi_t:
            target_release = min(
                self.I_bar * (seasonal_release_t + epsilon_t + 1),
                self.R_max
            )
        elif S_hat_t > NORhi_t:
            target_release = min((self.S_cap * (S_hat_t - NORhi_t) + I_t * 7) / 7, self.R_max)
        else:
            if self.linear_below_NOR:
                target_release = (self.I_bar * (seasonal_release_t + epsilon_t + 1)) * (S_hat_t / NORlo_t)
            else:
                target_release = self.R_min

        # Enforce the minimum release in all storage conditions (matches
        # calculate_target_release and compute_starfit_release_step; was
        # previously only applied below the NOR).
        target_release = max(target_release, self.R_min)

        # Ensure release does not exceed available water and capacity constraints
        available_water = I_t + S_t
        min_required = available_water - self.S_cap
        release_t = max(min(target_release, available_water), min_required)

        return max(0.0, release_t)

    @classmethod
    def load(cls, model, data):
        """Set up the parameter."""
        reservoir_name = data.pop("node")
        storage_node = model.nodes[f"reservoir_{reservoir_name}"]
        flow_parameter = load_parameter(model, f"flow_{reservoir_name}")
        run_starfit_sensitivity_analysis = data.pop("run_starfit_sensitivity_analysis")
        sensitivity_analysis_scenarios = data.pop("sensitivity_analysis_scenarios")
        starfit_params_filename = data.pop("starfit_params_filename", None)
        return cls(
            model,
            reservoir_name,
            storage_node,
            flow_parameter,
            run_starfit_sensitivity_analysis,
            sensitivity_analysis_scenarios,
            starfit_params_filename=starfit_params_filename,
            **data,
        )


# Register the parameter for use with Pywr
STARFITReservoirRelease.register()