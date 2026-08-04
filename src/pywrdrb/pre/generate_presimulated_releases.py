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
- For ensembles, simulate_reservoir_ensemble() runs the same day loop with
  each iteration vectorized across realizations; results are bit-identical
  to the scalar path.

Change Log
----------
TJA, 2025-11-26, Initial post-processing version (generate_presimulated_releases).
TJA, 2026-03, Added STARFITOfflineSimulator for offline pre-simulation.
TJA, 2026-07-22, Enforce R_min in all storage conditions (matches starfit.py).
TJA, 2026-07-22, Match model consumption timing (CU_ratio * withdrawal_{t-1},
    withdrawal limited by inflow) and bound releases by net available water.
TJA, 2026-08-04, Vectorize ensemble pre-simulation across realizations
    (simulate_reservoir_ensemble); bit-identical to the per-realization path.
"""

import os
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from pywrdrb.path_manager import get_pn_object
from pywrdrb.pre._mpi_utils import bcast_with_error, point_to_point_gather
from pywrdrb.pre.datapreprocessor_ABC import DataPreprocessor
from pywrdrb.utils.lists import (
    starfit_reservoir_list,
    modified_starfit_reservoir_list,
    reservoir_list,
)
from pywrdrb.parameters.lower_basin_ffmp import conservation_releases, max_discharges

pn = get_pn_object()

__all__ = ["STARFITOfflineSimulator", "STARFITReleaseEnsemblePreprocessor"]


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
    starfit_params_filename : str, optional
        Path to an alternative STARFIT parameter CSV (same format as
        istarf_conus.csv). If None, the default packaged file is used.

    Examples
    --------
    >>> from pywrdrb.pre.generate_presimulated_releases import STARFITOfflineSimulator
    >>> sim = STARFITOfflineSimulator(initial_volume_frac=0.8)
    >>> sim.load_parameters()
    >>> releases_df = sim.simulate_all(catchment_inflows_df)
    """

    def __init__(self, initial_volume_frac=0.8, starfit_params_filename=None):
        self.initial_volume_frac = initial_volume_frac
        self.starfit_params_filename = starfit_params_filename
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
        istarf_path = (
            self.starfit_params_filename
            or pn.operational_constants.get_str("istarf_conus.csv")
        )
        self._istarf = pd.read_csv(
            istarf_path,
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

    def _get_withdrawal_params(self, reservoir_name):
        """
        Get the catchment withdrawal capacity and consumption ratio.

        These describe the model's catchmentWithdrawal/catchmentConsumption
        nodes: each day the catchment withdraws min(WD, inflow), and the
        consumptive loss is CU_ratio * withdrawal from the previous day
        (the non-consumed remainder returns to the reservoir).

        Parameters
        ----------
        reservoir_name : str
            Reservoir name matching starfit_reservoir_list entries.

        Returns
        -------
        tuple of (float, float)
            (Total_WD_MGD, Total_CU_WD_Ratio). (0.0, 0.0) if no data.
        """
        if self._catchment_wc is None:
            return 0.0, 0.0
        pywr_node = f"reservoir_{reservoir_name}"
        if pywr_node in self._catchment_wc.index:
            wd = self._catchment_wc.loc[pywr_node, "Total_WD_MGD"]
            cu = self._catchment_wc.loc[pywr_node, "Total_CU_WD_Ratio"]
            return wd, cu
        return 0.0, 0.0

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

        # Catchment withdrawal capacity and consumption ratio for the
        # storage balance (mirrors catchmentWithdrawal/catchmentConsumption)
        wd, cu_ratio = self._get_withdrawal_params(reservoir_name)

        n = len(inflows)
        storage = np.empty(n + 1)
        releases = np.empty(n)

        # Initial storage (matches model_builder.py line 795)
        storage[0] = S_cap * self.initial_volume_frac

        # Previous-day withdrawal; pywr's prev_flow starts at 0, so the
        # first day has zero consumption in the model as well.
        withdrawal_prev = 0.0

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
            # Enforce R_min in all storage conditions (matches starfit.py)
            target_release = max(target_release, R_min)

            # Constraints (starfit.py lines 556-560)
            # NOTE: uses gross I_t for available_water, matching online STARFIT
            available_water = I_t + S_t
            min_required = available_water - S_cap
            release_t = max(min(target_release, available_water), min_required)
            release_t = max(0.0, release_t)

            # Storage balance uses NET inflow (gross - consumption) to match
            # the Pywr model's mass balance, where consumption_t is
            # CU_ratio * withdrawal_{t-1}, and withdrawal is limited by the
            # day's inflow: withdrawal_t = min(WD, I_t).
            withdrawal_t = min(wd, I_t)
            consumption_t = min(cu_ratio * withdrawal_prev, withdrawal_t)
            withdrawal_prev = withdrawal_t
            net_inflow = I_t - consumption_t

            # The model's LP cannot release more than the net available
            # water; re-limit here so storage stays within [0, S_cap].
            release_t = min(release_t, S_t + net_inflow)
            releases[t] = release_t
            storage[t + 1] = S_t + net_inflow - release_t

        return releases, storage

    def simulate_reservoir_ensemble(self, reservoir_name, inflows, day_of_year):
        """
        Simulate a single STARFIT reservoir across many realizations at once.

        Vectorized counterpart of simulate_reservoir(): the sequential day
        loop is unchanged (S[t+1] depends on S[t]) but each iteration
        operates on all realizations simultaneously, so interpreted
        iterations drop by a factor of n_realizations. The arithmetic is
        the same elementwise IEEE-754 operation sequence as the scalar
        path, so results are bit-identical per realization: branches
        become np.where with identical selection (A_t and epsilon_t are
        already computed unconditionally in the scalar path), and clamps
        become np.minimum/np.maximum with the same argument order.

        Parameters
        ----------
        reservoir_name : str
            Reservoir name matching starfit_reservoir_list.
        inflows : np.ndarray
            Daily gross inflows in MGD, shape (n_days, n_realizations).
        day_of_year : np.ndarray
            Day-of-year values (1-366), shape (n_days,).

        Returns
        -------
        releases : np.ndarray
            Daily releases in MGD, shape (n_days, n_realizations).
        storage : np.ndarray
            Daily storage in MG, shape (n_days + 1, n_realizations).
            storage[0] is initial.
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

        inv_S_cap = 1.0 / S_cap
        inv_I_bar = 1.0 / I_bar

        wd, cu_ratio = self._get_withdrawal_params(reservoir_name)

        n, n_realizations = inflows.shape
        storage = np.empty((n + 1, n_realizations))
        releases = np.empty((n, n_realizations))

        storage[0] = S_cap * self.initial_volume_frac

        withdrawal_prev = np.zeros(n_realizations)

        # Main simulation loop — sequential due to storage dependency.
        # Each line is the elementwise twin of the scalar path in
        # simulate_reservoir(); keep the operation order in sync.
        for t in range(n):
            I_t = inflows[t]  # gross inflow, shape (n_realizations,)
            S_t = storage[t]

            I_hat_t = (I_t - I_bar) * inv_I_bar
            S_hat_t = S_t * inv_S_cap

            day_idx = day_of_year[t] - 1  # 0-indexed
            NORhi_t = norhi[day_idx]
            NORlo_t = norlo[day_idx]
            seasonal_release_t = harmonic[day_idx]

            A_t = (S_hat_t - NORlo_t) / NORhi_t
            epsilon_t = Release_c + Release_p1 * A_t + Release_p2 * I_hat_t

            in_nor = (NORlo_t <= S_hat_t) & (S_hat_t <= NORhi_t)
            above_nor = S_hat_t > NORhi_t
            target_in_nor = np.minimum(
                I_bar * (seasonal_release_t + epsilon_t + 1), R_max
            )
            target_above_nor = np.minimum(
                (S_cap * (S_hat_t - NORhi_t) + I_t * 7) / 7, R_max
            )
            target_release = np.where(
                in_nor,
                target_in_nor,
                np.where(above_nor, target_above_nor, R_min),
            )
            # Enforce R_min in all storage conditions (matches starfit.py)
            target_release = np.maximum(target_release, R_min)

            available_water = I_t + S_t
            min_required = available_water - S_cap
            release_t = np.maximum(
                np.minimum(target_release, available_water), min_required
            )
            release_t = np.maximum(0.0, release_t)

            withdrawal_t = np.minimum(wd, I_t)
            consumption_t = np.minimum(cu_ratio * withdrawal_prev, withdrawal_t)
            withdrawal_prev = withdrawal_t
            net_inflow = I_t - consumption_t

            release_t = np.minimum(release_t, S_t + net_inflow)
            releases[t] = release_t
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





# ---------------------------------------------------------------------------
# Ensemble preprocessor (HDF5 in / HDF5 out, MPI-parallel).
# ---------------------------------------------------------------------------

INPUT_HDF5_NAME = "catchment_inflow_mgd.hdf5"
OUTPUT_HDF5_NAME = "presimulated_releases_mgd.hdf5"
OUTPUT_METADATA_NAME = "presimulated_releases_mgd_metadata.json"


class STARFITReleaseEnsemblePreprocessor(DataPreprocessor):
    """
    Ensemble counterpart of ``STARFITOfflineSimulator.generate_and_save()``.

    Reads a node-first ensemble inflow HDF5 (``catchment_inflow_mgd.hdf5``),
    runs STARFIT per realization for every reservoir in
    ``starfit_reservoir_list`` (all 14 reservoirs by default), and writes a
    node-first HDF5 (``presimulated_releases_mgd.hdf5``) of daily releases in
    MGD plus a JSON metadata sidecar. The HDF5 layout mirrors
    ``catchment_inflow_with_flood_nodes_mgd.hdf5`` exactly so it loads
    cleanly through the FlowEnsemble-style code path.

    The output is consumed by:
    - ``PresimulatedReleaseEnsemble`` — the ensemble trimmed-model parameter.
    - ``PredictedInflowEnsemblePreprocessor`` (perfect_foresight mode) — reads
      precomputed STARFIT releases instead of recomputing them per realization.

    Parallelization mirrors ``FloodNodeInflowEnsemblePreprocessor``:
    ``np.array_split`` distributes realizations across ranks; each rank reads
    its slice of the input HDF5 concurrently; STARFIT runs once per reservoir
    via ``STARFITOfflineSimulator.simulate_reservoir_ensemble``, vectorized
    across the rank's realizations (bit-identical to per-realization runs);
    results are merged onto rank 0 via point-to-point gather and saved.

    Parameters
    ----------
    inflow_type : str
        Flow data source label (e.g., ``'nhmv10'``).
    realization_ids : list of int or str, optional
        Realization IDs to process. If None, every realization in the input
        HDF5 is processed.
    use_mpi : bool, default False
        If True, use MPI for per-realization parallelism.
    comm : mpi4py.MPI.Comm, optional
        MPI communicator. Defaults to ``MPI.COMM_WORLD`` when ``use_mpi=True``.
    force : bool, default False
        If True, rebuild the output HDF5 even if it already exists.
    initial_volume_frac : float, default 0.8
        Initial reservoir storage as fraction of capacity. Must match
        ``ModelBuilder.Options.initial_volume_frac`` for downstream consistency.
    reservoir_list : list of str, optional
        Subset of reservoirs to simulate. Defaults to ``starfit_reservoir_list``
        (all 14). Trimmed-model only consumes the 11 ``independent_starfit_reservoirs``
        but the default writes all 14 so perfect_foresight can read the same artifact.
    """

    def __init__(
        self,
        inflow_type="nhmv10",
        realization_ids=None,
        use_mpi=False,
        comm=None,
        force=False,
        initial_volume_frac=0.8,
        reservoir_list=None,
    ):
        super().__init__()
        self.inflow_type = inflow_type
        self.realization_ids = realization_ids
        self.force = force
        self.initial_volume_frac = initial_volume_frac
        self.reservoir_list = (
            list(reservoir_list)
            if reservoir_list is not None
            else list(starfit_reservoir_list)
        )

        if use_mpi:
            if comm is None:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
            self.comm = comm
            self.rank = comm.Get_rank()
            self.size = comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
        self.use_mpi = use_mpi

        # Use the shortcut registry (pn.sc) rather than pn.flows.get_str so
        # users who registered a customized flow_type via load_pn_config can
        # point the preprocessor at it. The rest of the ensemble pipeline
        # (FlowEnsemble, FloodNodeInflowEnsemblePreprocessor) uses pn.sc.get
        # for the same reason.
        flows_dir = str(self.pn.sc.get(f"flows/{inflow_type}"))
        self.input_dirs = {INPUT_HDF5_NAME: flows_dir}
        self.output_dirs = {OUTPUT_HDF5_NAME: flows_dir}

        self._input_path = os.path.join(flows_dir, INPUT_HDF5_NAME)
        self._output_path = os.path.join(flows_dir, OUTPUT_HDF5_NAME)
        self._metadata_path = os.path.join(flows_dir, OUTPUT_METADATA_NAME)

        # Populated by load() / process().
        self._node_names = None
        self._dates = None
        self._my_realization_inflows = {}
        self._local_releases = {}
        self._simulator = None

    # ---- I/O helpers --------------------------------------------------

    def _resolve_realization_ids(self):
        """Read realization IDs (as strings) from the input HDF5 on rank 0."""
        if self.realization_ids is not None:
            return [str(r) for r in self.realization_ids]
        with h5py.File(self._input_path, "r") as f:
            first_node = next(iter(f.keys()))
            labels = f[first_node].attrs["column_labels"]
            return [str(label) for label in labels]

    def _resolve_node_names(self):
        """Read input HDF5 top-level group names on rank 0."""
        with h5py.File(self._input_path, "r") as f:
            return [k for k in f.keys() if isinstance(f[k], h5py.Group)]

    def _read_one_realization(self, hdf5_file, realization_id, node_names):
        """Build a per-realization dict of {node: 1D inflow array} from an open file."""
        return {
            node: hdf5_file[node][str(realization_id)][:] for node in node_names
        }

    def _read_dates(self, hdf5_file, node_names):
        """Read the canonical date axis from the input HDF5 (matches flood-node helper)."""
        node_group = hdf5_file[node_names[0]]
        if "date" in node_group:
            raw = node_group["date"][:]
        elif "datetime" in node_group:
            raw = node_group["datetime"][:]
        else:
            raise KeyError(
                f"Neither 'date' nor 'datetime' dataset present in "
                f"/{node_names[0]} of {self._input_path}"
            )
        return [d.decode() if isinstance(d, bytes) else str(d) for d in raw]

    # ---- ABC interface -------------------------------------------------

    def load(self):
        """Resolve realization IDs and node names; each rank reads its slice."""
        if not os.path.exists(self._input_path):
            raise FileNotFoundError(
                f"Base ensemble inflow file not found: {self._input_path}\n"
                f"Generate it first via the ensemble inflow pipeline "
                f"(see pywrdrb.pre flows preprocessing)."
            )

        if self.use_mpi:
            self.realization_ids = bcast_with_error(
                self.comm, self.rank, self._resolve_realization_ids
            )
            self._node_names = bcast_with_error(
                self.comm, self.rank, self._resolve_node_names
            )
        else:
            self.realization_ids = self._resolve_realization_ids()
            self._node_names = self._resolve_node_names()

        # Validate that every reservoir we plan to simulate is present in the input.
        missing = [r for r in self.reservoir_list if r not in self._node_names]
        if missing:
            raise ValueError(
                f"Reservoirs missing from {self._input_path}: {missing}. "
                f"Cannot simulate STARFIT for nodes that have no inflow data."
            )

        if self.use_mpi:
            my_ids = list(
                np.array_split(self.realization_ids, self.size)[self.rank]
            )
            self.comm.Barrier()
        else:
            my_ids = list(self.realization_ids)

        t0 = time.time()
        with h5py.File(self._input_path, "r") as f:
            for rid in my_ids:
                # Only reservoir inflows are consumed downstream; skip the
                # other node groups to cut input read time.
                self._my_realization_inflows[str(rid)] = (
                    self._read_one_realization(f, rid, self.reservoir_list)
                )
            # Read canonical date axis from one (any) node group on every rank
            # so save() on rank 0 always has it without an extra gather.
            self._dates = self._read_dates(f, self._node_names)

        # Build the engine on each rank (one instance, parameters cached internally).
        self._simulator = STARFITOfflineSimulator(
            initial_volume_frac=self.initial_volume_frac
        )
        self._simulator.load_parameters()

        if self.rank == 0:
            print(
                f"[rank {self.rank}/{self.size}] load: read "
                f"{len(my_ids)} realizations in {time.time() - t0:.1f}s "
                f"(of {len(self.realization_ids)} total)"
            )
        if self.use_mpi:
            self.comm.Barrier()

    def process(self):
        """Run STARFIT per assigned realization via STARFITOfflineSimulator."""
        # Idempotency: if output already exists and force is False, skip.
        if not self.force and os.path.exists(self._output_path):
            if self.rank == 0:
                print(
                    f"Output already exists at {self._output_path}; "
                    "skipping (set force=True to rebuild)."
                )
            self._local_releases = {}
            self.processed_data["releases"] = {}
            return

        if not self._my_realization_inflows:
            self.load()

        index = pd.to_datetime(pd.Index(self._dates))
        day_of_year = index.dayofyear.values
        rid_order = list(self._my_realization_inflows.keys())
        local = {}
        if rid_order:
            # One vectorized run per reservoir over all of this rank's
            # realizations (columns), instead of one scalar run per
            # realization; bit-identical to the per-realization path.
            releases_per_reservoir = {}
            for res in self.reservoir_list:
                inflows_2d = np.column_stack(
                    [self._my_realization_inflows[rid][res] for rid in rid_order]
                ).astype(float)
                rel, _ = self._simulator.simulate_reservoir_ensemble(
                    res, inflows_2d, day_of_year
                )
                releases_per_reservoir[res] = rel
            for j, rid in enumerate(rid_order):
                local[rid] = pd.DataFrame(
                    {
                        res: releases_per_reservoir[res][:, j]
                        for res in self.reservoir_list
                    },
                    index=index,
                )
        self._local_releases = local

        if self.use_mpi:
            merged = point_to_point_gather(
                self.comm, self.rank, self.size, local
            )
            if self.rank == 0:
                self.processed_data["releases"] = merged
            else:
                self.processed_data["releases"] = {}
        else:
            self.processed_data["releases"] = dict(local)

    def save(self):
        """Write the release HDF5 in node-first schema (rank 0 only) + JSON sidecar."""
        if self.use_mpi and self.rank != 0:
            self.comm.Barrier()
            return

        releases = self.processed_data.get("releases", {})
        if not releases:
            # No-op: process() short-circuited (output already present).
            if self.use_mpi:
                self.comm.Barrier()
            return

        rid_order = [str(r) for r in self.realization_ids]
        missing = [r for r in rid_order if r not in releases]
        if missing:
            raise RuntimeError(
                f"Releases missing realizations: {missing[:5]}..."
            )

        # Use the first realization's reservoir list as canonical column order.
        first_rid = rid_order[0]
        all_reservoirs = list(releases[first_rid].columns)

        # Sanity: every realization must share the same reservoir set.
        for rid in rid_order[1:]:
            if list(releases[rid].columns) != all_reservoirs:
                raise RuntimeError(
                    f"Release column set for realization {rid} differs "
                    f"from realization {first_rid}."
                )

        Path(self._output_path).parent.mkdir(parents=True, exist_ok=True)

        date_strings = np.asarray([str(d) for d in self._dates], dtype=object)
        str_dtype = h5py.string_dtype(encoding="utf-8")
        column_labels = np.asarray(rid_order, dtype=object)

        with h5py.File(self._output_path, "w") as hf:
            for reservoir_name in all_reservoirs:
                node_group = hf.create_group(reservoir_name)
                node_group.attrs.create(
                    "column_labels", column_labels, dtype=str_dtype
                )
                for rid in rid_order:
                    node_group.create_dataset(
                        rid,
                        data=releases[rid][reservoir_name].to_numpy(dtype=float),
                    )
                node_group.create_dataset(
                    "date", data=date_strings, dtype=str_dtype
                )

        # Date range for metadata. _dates entries are 'YYYY-MM-DD' strings or
        # numpy bytes; pd.to_datetime handles both and gives clean ISO output.
        start_date = str(pd.to_datetime(self._dates[0]).date())
        end_date = str(pd.to_datetime(self._dates[-1]).date())

        metadata = {
            "inflow_type": self.inflow_type,
            "start_date": start_date,
            "end_date": end_date,
            "reservoirs": list(all_reservoirs),
            "realization_ids": list(rid_order),
            "initial_volume_frac": self.initial_volume_frac,
            "source": "STARFITReleaseEnsemblePreprocessor",
            "output_file": str(self._output_path),
            "metadata_file": str(self._metadata_path),
        }
        with open(self._metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(
            f"Saved ensemble pre-simulated releases to {self._output_path} "
            f"({len(all_reservoirs)} reservoirs x {len(rid_order)} realizations x "
            f"{len(date_strings)} timesteps)"
        )
        if self.use_mpi:
            self.comm.Barrier()

    def run(self):
        """Convenience entry point: load → process → save."""
        self.load()
        self.process()
        self.save()
