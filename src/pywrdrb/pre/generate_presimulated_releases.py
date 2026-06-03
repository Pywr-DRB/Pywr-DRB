"""
Offline STARFIT reservoir simulator for pre-simulating reservoir operations.

Overview
--------
This module provides a standalone STARFIT simulator that replicates the release logic
from STARFITReservoirRelease (parameters/starfit.py) without requiring Pywr runtime.
It serves two purposes:
1. **Perfect foresight prediction** — pre-simulate STARFIT releases for use in
   the inflow prediction pipeline (predict_inflows.py)
2. **Trimmed model** — generate pre-simulated releases CSV for use_trimmed_model mode,
   replacing the need to run a full Pywr model first

Technical Notes
---------------
- The STARFITOfflineSimulator replicates the exact arithmetic from
  STARFITReservoirRelease.value() (starfit.py lines 480-560) to ensure
  identical releases when given the same inflows and initial storage.
- Seasonal lookup tables are pre-computed once per reservoir for efficiency.
- The sequential storage loop is unavoidable (S[t+1] depends on S[t]),
  but each iteration is pure scalar arithmetic and runs quickly.

Change Log
----------
TJA, 2025-11-26, Initial post-processing version (generate_presimulated_releases).
TJA, 2026-03, Added STARFITOfflineSimulator for offline pre-simulation.
"""

import os
import json
import numpy as np
import pandas as pd

from pywrdrb.path_manager import get_pn_object
from pywrdrb.utils.lists import (
    starfit_reservoir_list,
    modified_starfit_reservoir_list,
    reservoir_list,
)
from pywrdrb.parameters.lower_basin_ffmp import conservation_releases, max_discharges

pn = get_pn_object()

__all__ = ["STARFITOfflineSimulator"]


class STARFITOfflineSimulator:
    """
    Offline STARFIT reservoir simulator.

    Replicates the STARFIT release logic from STARFITReservoirRelease
    (parameters/starfit.py) without requiring Pywr runtime objects.
    Pre-computes seasonal lookup tables and simulates storage dynamics
    day-by-day for each reservoir.

    Parameters
    ----------
    initial_volume_frac : float
        Initial reservoir storage as a fraction of capacity. Default is 0.8,
        matching the default in ModelBuilder.Options.

    Examples
    --------
    >>> from pywrdrb.pre.generate_presimulated_releases import STARFITOfflineSimulator
    >>> sim = STARFITOfflineSimulator(initial_volume_frac=0.8)
    >>> sim.load_parameters()
    >>> releases_df = sim.simulate_all(catchment_inflows_df)
    """

    def __init__(self, initial_volume_frac=0.8):
        self.initial_volume_frac = initial_volume_frac
        self._params_loaded = False
        self._istarf = None
        self._catchment_wc = None
        # Cache for per-reservoir parameter dicts
        self._reservoir_params_cache = {}

    def load_parameters(self):
        """
        Load STARFIT parameters from istarf_conus.csv and water consumption data.

        Mirrors STARFITReservoirRelease.load_default_starfit_params()
        (starfit.py lines 162-177).
        """
        self._istarf = pd.read_csv(
            pn.operational_constants.get_str("istarf_conus.csv"),
            sep=",",
            index_col=0,
        )

        # Load catchment water consumption data for storage balance correction.
        # In the Pywr model, catchment flow is split between reservoir inflow and
        # withdrawal/consumption nodes. The net inflow to the reservoir is
        # gross_inflow - consumption. The release formula uses gross inflow
        # (matching the online STARFITReservoirRelease parameter), but the
        # storage balance must use net inflow to match the model's mass balance.
        wc_file = pn.catchment_withdrawals.get(
            "sw_avg_wateruse_pywrdrb_catchments_mgd.csv"
        )
        wc = pd.read_csv(wc_file)
        wc.index = wc["node"]
        self._catchment_wc = wc

        self._params_loaded = True
        self._reservoir_params_cache = {}

    def _get_reservoir_params(self, reservoir_name):
        """
        Extract STARFIT parameters for a single reservoir.

        Mirrors STARFITReservoirRelease.assign_starfit_param_values()
        (starfit.py lines 202-280) exactly.

        Parameters
        ----------
        reservoir_name : str
            Reservoir name matching starfit_reservoir_list entries.

        Returns
        -------
        dict
            Dictionary with keys: S_cap, I_bar, R_min, R_max, and all
            NOR/Release coefficients needed for simulation.
        """
        if reservoir_name in self._reservoir_params_cache:
            return self._reservoir_params_cache[reservoir_name]

        if not self._params_loaded:
            self.load_parameters()

        # Handle modified_starfit_reservoir_list prefix (starfit.py lines 218-222)
        starfit_name = (
            "modified_" + reservoir_name
            if reservoir_name in modified_starfit_reservoir_list
            else reservoir_name
        )

        if starfit_name not in self._istarf.index:
            raise ValueError(
                f"No STARFIT parameters found for '{starfit_name}' "
                f"(reservoir: {reservoir_name})"
            )

        row = self._istarf.loc[starfit_name]

        # Use adjusted storage (starfit.py lines 232-237, use_adjusted_storage=True)
        S_cap = row["Adjusted_CAP_MG"]
        I_bar = row["Adjusted_MEANFLOW_MGD"]

        # R_max override for DRBC lower basin (starfit.py lines 266-273)
        if reservoir_name in max_discharges:
            R_max = max_discharges[reservoir_name]
        else:
            # remove_R_max=False is the default (starfit.py line 146)
            R_max = (row["Release_max"] + 1) * I_bar

        # R_min override for DRBC lower basin (starfit.py lines 276-279)
        if reservoir_name in conservation_releases:
            R_min = conservation_releases[reservoir_name]
        else:
            R_min = (row["Release_min"] + 1) * I_bar

        params = {
            "S_cap": S_cap,
            "I_bar": I_bar,
            "R_min": R_min,
            "R_max": R_max,
            # NOR bounds (starfit.py lines 244-254)
            "NORhi_mu": row["NORhi_mu"],
            "NORhi_alpha": row["NORhi_alpha"],
            "NORhi_beta": row["NORhi_beta"],
            "NORhi_min": row["NORhi_min"] / 100,
            "NORhi_max": row["NORhi_max"] / 100,
            "NORlo_mu": row["NORlo_mu"],
            "NORlo_alpha": row["NORlo_alpha"],
            "NORlo_beta": row["NORlo_beta"],
            "NORlo_min": row["NORlo_min"] / 100,
            "NORlo_max": row["NORlo_max"] / 100,
            # Release coefficients (starfit.py lines 256-263)
            "Release_alpha1": row["Release_alpha1"],
            "Release_alpha2": row["Release_alpha2"],
            "Release_beta1": row["Release_beta1"],
            "Release_beta2": row["Release_beta2"],
            "Release_c": row["Release_c"],
            "Release_p1": row["Release_p1"],
            "Release_p2": row["Release_p2"],
        }

        self._reservoir_params_cache[reservoir_name] = params
        return params

    def _get_catchment_consumption(self, reservoir_name):
        """
        Get the daily water consumption for a reservoir's catchment.

        Returns the consumptive use (CU_ratio * withdrawal) which represents
        the water removed from the system before reaching the reservoir.
        This matches the model's catchmentConsumption node behavior.

        Parameters
        ----------
        reservoir_name : str
            Reservoir name matching starfit_reservoir_list entries.

        Returns
        -------
        float
            Daily consumption in MGD. Returns 0.0 if no data available.
        """
        if self._catchment_wc is None:
            return 0.0
        pywr_node = f"reservoir_{reservoir_name}"
        if pywr_node in self._catchment_wc.index:
            wd = self._catchment_wc.loc[pywr_node, "Total_WD_MGD"]
            cu = self._catchment_wc.loc[pywr_node, "Total_CU_WD_Ratio"]
            return cu * wd
        return 0.0

    def _precompute_seasonal_arrays(self, params):
        """
        Pre-compute 366-day lookup arrays for harmonic release, NORhi, NORlo.

        Mirrors STARFITReservoirRelease._precompute_seasonal_lookups()
        (starfit.py lines 294-330) exactly.

        Parameters
        ----------
        params : dict
            Reservoir parameters from _get_reservoir_params().

        Returns
        -------
        tuple of (harmonic, norhi, norlo)
            Each is a numpy array of shape (366,) indexed by (day_of_year - 1).
        """
        # WATER_YEAR_OFFSET = 0 (starfit.py line 149)
        WATER_YEAR_OFFSET = 0
        pi_over_365 = np.pi / 365

        days = np.arange(1, 367)  # Day of year 1-366
        c_values = pi_over_365 * (days + WATER_YEAR_OFFSET)

        # Pre-compute trig values (starfit.py lines 302-306)
        sin_2c = np.sin(2 * c_values)
        sin_4c = np.sin(4 * c_values)
        cos_2c = np.cos(2 * c_values)
        cos_4c = np.cos(4 * c_values)

        # Harmonic release lookup (starfit.py lines 309-314)
        harmonic = (
            params["Release_alpha1"] * sin_2c
            + params["Release_alpha2"] * sin_4c
            + params["Release_beta1"] * cos_2c
            + params["Release_beta2"] * cos_4c
        )

        # NOR bounds lookup (starfit.py lines 317-329)
        nor_hi_raw = (
            params["NORhi_mu"]
            + params["NORhi_alpha"] * sin_2c
            + params["NORhi_beta"] * cos_2c
        )
        norhi = (
            np.clip(
                nor_hi_raw,
                params["NORhi_min"] * 100,
                params["NORhi_max"] * 100,
            )
            / 100
        )

        nor_lo_raw = (
            params["NORlo_mu"]
            + params["NORlo_alpha"] * sin_2c
            + params["NORlo_beta"] * cos_2c
        )
        norlo = (
            np.clip(
                nor_lo_raw,
                params["NORlo_min"] * 100,
                params["NORlo_max"] * 100,
            )
            / 100
        )

        return harmonic, norhi, norlo

    def simulate_reservoir(self, reservoir_name, inflows, day_of_year):
        """
        Simulate a single STARFIT reservoir over all timesteps.

        Replicates STARFITReservoirRelease.value() (starfit.py lines 480-560)
        for the release formula, but uses net inflow (gross - consumption)
        for the storage balance to match the Pywr model's mass balance.

        In the Pywr model, the STARFIT parameter uses gross inflow (I_t) for
        its release formula, but the reservoir's actual storage dynamics use
        net inflow (gross - catchment consumption). This method replicates
        that behavior.

        Parameters
        ----------
        reservoir_name : str
            Reservoir name matching starfit_reservoir_list.
        inflows : np.ndarray
            Daily gross inflows in MGD, shape (n_days,).
        day_of_year : np.ndarray
            Day-of-year values (1-366), shape (n_days,).

        Returns
        -------
        releases : np.ndarray
            Daily releases in MGD, shape (n_days,).
        storage : np.ndarray
            Daily storage in MG, shape (n_days + 1,). storage[0] is initial.
        """
        params = self._get_reservoir_params(reservoir_name)
        harmonic, norhi, norlo = self._precompute_seasonal_arrays(params)

        S_cap = params["S_cap"]
        I_bar = params["I_bar"]
        R_min = params["R_min"]
        R_max = params["R_max"]
        Release_c = params["Release_c"]
        Release_p1 = params["Release_p1"]
        Release_p2 = params["Release_p2"]

        # Pre-compute inverse constants (starfit.py lines 240-241)
        inv_S_cap = 1.0 / S_cap
        inv_I_bar = 1.0 / I_bar

        # Get catchment consumption for storage balance correction
        consumption = self._get_catchment_consumption(reservoir_name)

        n = len(inflows)
        storage = np.empty(n + 1)
        releases = np.empty(n)

        # Initial storage (matches model_builder.py line 795)
        storage[0] = S_cap * self.initial_volume_frac

        # Main simulation loop — sequential due to storage dependency
        # Each iteration mirrors starfit.py lines 522-560
        for t in range(n):
            I_t = inflows[t]  # gross inflow (used in release formula)
            S_t = storage[t]

            # Inlined: standardize_inflow and calculate_percent_storage
            # (starfit.py lines 527-528)
            # NOTE: uses gross I_t, matching online STARFITReservoirRelease
            I_hat_t = (I_t - I_bar) * inv_I_bar
            S_hat_t = S_t * inv_S_cap

            # Lookup seasonal values (starfit.py lines 531-534)
            day_idx = day_of_year[t] - 1  # 0-indexed
            NORhi_t = norhi[day_idx]
            NORlo_t = norlo[day_idx]
            seasonal_release_t = harmonic[day_idx]

            # Inlined: calculate_release_adjustment (starfit.py lines 537-538)
            A_t = (S_hat_t - NORlo_t) / NORhi_t
            epsilon_t = Release_c + Release_p1 * A_t + Release_p2 * I_hat_t

            # Inlined: calculate_target_release (starfit.py lines 541-553)
            # linear_below_NOR=False (starfit.py line 147)
            if NORlo_t <= S_hat_t <= NORhi_t:
                target_release = min(
                    I_bar * (seasonal_release_t + epsilon_t + 1), R_max
                )
            elif S_hat_t > NORhi_t:
                target_release = min(
                    (S_cap * (S_hat_t - NORhi_t) + I_t * 7) / 7, R_max
                )
            else:
                # S_hat_t < NORlo_t, linear_below_NOR=False
                target_release = R_min

            # Constraints (starfit.py lines 556-560)
            # NOTE: uses gross I_t for available_water, matching online STARFIT
            available_water = I_t + S_t
            min_required = available_water - S_cap
            release_t = max(min(target_release, available_water), min_required)
            release_t = max(0.0, release_t)

            releases[t] = release_t

            # Storage balance uses NET inflow (gross - consumption)
            # to match the Pywr model's mass balance where catchment flow
            # is split between reservoir inflow and consumption nodes.
            net_inflow = max(I_t - consumption, 0.0)
            storage[t + 1] = S_t + net_inflow - release_t

        return releases, storage

    def simulate_all(self, catchment_inflows_df, reservoir_list=None):
        """
        Simulate all STARFIT reservoirs given a catchment inflows DataFrame.

        Parameters
        ----------
        catchment_inflows_df : pd.DataFrame
            DataFrame with DatetimeIndex, columns for each node.
            Must contain columns for each reservoir in reservoir_list.
        reservoir_list : list of str, optional
            Reservoirs to simulate. If None, uses starfit_reservoir_list.

        Returns
        -------
        pd.DataFrame
            DataFrame with same DatetimeIndex, columns = reservoir names,
            values = daily releases in MGD.
        """
        if not self._params_loaded:
            self.load_parameters()

        if reservoir_list is None:
            reservoir_list = starfit_reservoir_list

        dates = catchment_inflows_df.index
        day_of_year = dates.dayofyear.values

        releases_dict = {}
        for res in reservoir_list:
            if res not in catchment_inflows_df.columns:
                raise ValueError(
                    f"Reservoir '{res}' not found in catchment_inflows_df columns. "
                    f"Available: {list(catchment_inflows_df.columns[:10])}..."
                )
            inflows = catchment_inflows_df[res].values.astype(float)
            rel, _ = self.simulate_reservoir(res, inflows, day_of_year)
            releases_dict[res] = rel

        return pd.DataFrame(releases_dict, index=dates)

    def generate_and_save(self, flow_type, reservoir_list=None):
        """
        Load catchment inflows, simulate all STARFIT reservoirs, and save results.

        This is a convenience method that replaces the need to run a full Pywr
        model before using the trimmed model mode.

        Parameters
        ----------
        flow_type : str
            Inflow type label (e.g., 'nhmv10_withObsScaled').
        reservoir_list : list of str, optional
            Reservoirs to simulate. If None, uses starfit_reservoir_list.

        Returns
        -------
        dict
            Metadata dictionary with file paths and date range info.
        """
        if not self._params_loaded:
            self.load_parameters()

        # Load catchment inflows
        inflow_file = (
            pn.sc.get(f"flows/{flow_type}") / "catchment_inflow_mgd.csv"
        )
        catchment_inflows = pd.read_csv(
            str(inflow_file), index_col=0, parse_dates=True
        )
        catchment_inflows.index = pd.DatetimeIndex(catchment_inflows.index)

        # Simulate
        if reservoir_list is None:
            reservoir_list = starfit_reservoir_list
        releases_df = self.simulate_all(catchment_inflows, reservoir_list)

        # Save
        output_dir = str(pn.sc.get(f"flows/{flow_type}"))
        os.makedirs(output_dir, exist_ok=True)

        csv_file = os.path.join(output_dir, "presimulated_releases_mgd.csv")
        metadata_file = os.path.join(
            output_dir, "presimulated_releases_mgd_metadata.json"
        )

        # Format datetime index for CSV
        releases_out = releases_df.copy()
        releases_out.index.name = "datetime"
        releases_out.index = pd.to_datetime(releases_out.index).strftime("%Y-%m-%d")
        releases_out.to_csv(csv_file, float_format="%.10f")

        metadata = {
            "inflow_type": flow_type,
            "start_date": str(releases_out.index[0]),
            "end_date": str(releases_out.index[-1]),
            "reservoirs": reservoir_list,
            "initial_volume_frac": self.initial_volume_frac,
            "source": "STARFITOfflineSimulator",
            "output_file": csv_file,
            "metadata_file": metadata_file,
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved pre-simulated releases to: {csv_file}")
        print(
            f"  Date range: {metadata['start_date']} to {metadata['end_date']}"
        )
        print(f"  Reservoirs: {', '.join(reservoir_list)}")

        return metadata

