"""
STARFIT policy class for reservoir operation.

Overview
--------
This module implements the STARFIT policy class used by the release_policies framework.
The logic is aligned with `pywrdrb.parameters.starfit.STARFITReservoirRelease` to ensure
structural and conceptual consistency between the two implementations.

It is meant to give the same releases as parameters/starfit.py for the same inputs:
  - Below NOR_lo, the release is R_min (linear_below_NOR defaults to False).
  - The STARFIT math runs on raw storage and inflow, with no clipping into [0,1].
  - R_max/R_min come from the lower-basin override dicts, or (Release_*+1)*I_bar.
  - The final step clamps to available water and capacity, then max(0, .). R_min
    is only applied inside the target, never after the clamp.
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import Mapping, Any, Optional, Sequence
import pandas as pd
from math import pi, sin, cos
import os
from functools import lru_cache

from pywrdrb.release_policies.abstract_policy import AbstractPolicy
from pywrdrb.release_policies.config import (
    policy_n_params,
    policy_param_bounds,
    get_starfit_param_bounds,
    n_starfit_inputs,   # should be 3 for [S, I, D]
    STARFIT_PARAM_NAMES,
    DATA_DIR,
    CONFIG_DIR,
)
from pywrdrb.path_manager import get_pn_object
from pywrdrb.utils.lists import modified_starfit_reservoir_list
# conservation_releases / max_discharges are imported lazily in
# _reconstruct_release_limits() to avoid a circular import.

pn = get_pn_object()


class STARFIT(AbstractPolicy):
    """
    STARFIT policy class for reservoir operation.

    Overview
    --------
    Seasonal, rule-based surrogate policy (after Turner et al., 2021) combining:
      • Harmonic seasonal component (sin/cos with 1- and 2-year harmonics)
      • A Normal Operating Range (NOR) envelope varying seasonally (NOR_lo/hi)
      • Adjustments using normalized storage and standardized inflow

    Match with parameters/starfit.py
    --------------------------------
    Gives the same releases as `pywrdrb.parameters.starfit.STARFITReservoirRelease`
    for the same inputs:

    - Below NOR_lo, release is R_min (linear_below_NOR defaults to False).
    - The STARFIT math uses raw storage and inflow (no clipping into [0,1]).
    - R_max/R_min are reconstructed from the override dicts, or (Release_*+1)*I_bar.
    - The final clamp is to available water and capacity, then max(0, .).

    Interface (matches PWL/RBF)
    ---------------------------
    - Call `set_context(release_min, release_max, storage_capacity, x_min, x_max)` once.
      * x_min/x_max define min-max normalization for inputs [S, I, D].
      * Typically: x_min = (0, I_min, 1), x_max = (S_cap, I_max, 366).
    - `evaluate([S_norm, I_norm, D_norm]) -> z in [0,1]`, used for plotting.
    - `get_release(S, I, D)` returns the release in MGD from raw inputs.

    Parameters
    ----------
    Optimizer-vector order (len = policy_n_params["STARFIT"]):
      [ NORhi_mu, NORhi_min, NORhi_max, NORhi_alpha, NORhi_beta,
        NORlo_mu, NORlo_min, NORlo_max, NORlo_alpha, NORlo_beta,
        Release_alpha1, Release_alpha2, Release_beta1, Release_beta2,
        Release_c, Release_p1, Release_p2 ]

    CSV/row keys (when using assign_policy_params):
      - Above parameter names as columns
      - For context & standardization: one of S_cap / Adjusted_CAP_MG / GRanD_CAP_MG,
        one of Adjusted_MEANFLOW_MGD / GRanD_MEANFLOW_MGD as I_bar,
        I_min, I_max; optional R_min, R_max.

    Notes
    -----
    - NOR min/max come in as percent in the CSV and are divided by 100 here.
    - Standardized inflow is I_hat = (I - I_bar) / I_bar, so I_bar must be set
      (via assign_policy_params, load_starfit_params, or set_context).
    - The final release is limited by available water and capacity only; R_min
      and R_max enter inside the target, matching parameters/starfit.py.
    """

    def __init__(self, policy_params, reservoir_name: Optional[str] = None,):

        super().__init__(policy_params=policy_params)

        self.reservoir_name = self._effective_starfit_name(reservoir_name)

        self.n_inputs = int(n_starfit_inputs)  # expected 3 for [S, I, D]
        self.n_params = policy_n_params["STARFIT"]
        # Match Borg / MOEA search box: per-reservoir NOR envelope when configured.
        self.param_bounds = (
            get_starfit_param_bounds(reservoir_name)
            if reservoir_name
            else policy_param_bounds["STARFIT"]
        )
        self.policy_params = policy_params

        # seasonal phase offset (days)
        self.WATER_YEAR_OFFSET: float = 0.0

        # STARFIT scalars (filled by parse/assign)
        self.NORhi_mu = self.NORhi_min = self.NORhi_max = None
        self.NORhi_alpha = self.NORhi_beta = None

        self.NORlo_mu = self.NORlo_min = self.NORlo_max = None
        self.NORlo_alpha = self.NORlo_beta = None

        self.Release_alpha1 = self.Release_alpha2 = None
        self.Release_beta1 = self.Release_beta2 = None
        self.Release_c = self.Release_p1 = self.Release_p2 = None

        # standardized inflow mean (must be set for evaluate)
        self.I_bar = None

        # optional behavior toggle (aligned with parameters/starfit.py)
        # When False: uses R_min directly when storage < NOR_lo (reference default)
        # When True: linearly scales release by S_hat/NOR_lo before applying R_min
        self.linear_below_NOR: bool = False

        # release-limit reconstruction inputs (aligned with parameters/starfit.py)
        self.remove_R_max: bool = False
        self.Release_max = None
        self.Release_min = None

        # log file path (created once we know the name)
        self.log_path = None

        if policy_params is not None:
            self.parse_policy_params()

    # ---------- capacity constants ----------
    @lru_cache(maxsize=1)
    def _capacity_df(self) -> pd.DataFrame:
        """Read istarf_capacity.csv once and cache; index is lowercase reservoir."""
        path = pn.operational_constants.get_str("istarf_capacity.csv")
        df = pd.read_csv(path)
        if "reservoir" not in df.columns:
            raise KeyError("[STARFIT] istarf_capacity.csv must contain a 'reservoir' column.")
        df["_key"] = df["reservoir"].astype(str).str.strip().str.lower()
        df = df.set_index("_key", drop=False)
        return df

    def _ensure_I_bar_from_capacity(self) -> None:
        """Ensure self.I_bar is populated from istarf_capacity.csv using reservoir only."""
        if self.I_bar not in (None, 0.0):
            return
        if not self.reservoir_name:
            raise ValueError("[STARFIT] reservoir_name is required to resolve I_bar from capacity.")

        key = str(self.reservoir_name).strip().str.lower() if isinstance(self.reservoir_name, pd.Series) \
              else str(self.reservoir_name).strip().lower()

        df = self._capacity_df()
        if key not in df.index:
            raise KeyError(f"[STARFIT] '{self.reservoir_name}' not found in istarf_capacity.csv.")

        row = df.loc[key]
        if "Adjusted_MEANFLOW_MGD" in row and pd.notna(row["Adjusted_MEANFLOW_MGD"]):
            self.I_bar = float(row["Adjusted_MEANFLOW_MGD"])
        elif "GRanD_MEANFLOW_MGD" in row and pd.notna(row["GRanD_MEANFLOW_MGD"]):
            self.I_bar = float(row["GRanD_MEANFLOW_MGD"])
        else:
            raise KeyError(
                f"[STARFIT] istarf_capacity.csv missing Adjusted_MEANFLOW_MGD/GRanD_MEANFLOW_MGD "
                f"for reservoir '{row.get('reservoir', self.reservoir_name)}'."
            )
    @staticmethod
    def _pct_to_unit(v: float) -> float:
        # NOR min/max come in as percent; divide by 100.
        return float(v) / 100.0
    
    # ---------- optimizer-path: flat vector ----------
    def validate_policy_params(self) -> None:
        if self.policy_params is None:
            raise ValueError("STARFIT.policy_params is None.")
        if len(self.policy_params) != self.n_params:
            raise AssertionError(
                f"STARFIT expected {self.n_params} parameters, got {len(self.policy_params)}."
            )
        for i, p in enumerate(self.policy_params):
            lo, hi = self.param_bounds[i]
            if not (lo <= p <= hi):
                raise AssertionError(f"Param idx {i} out of bounds {lo, hi}. Value: {p}")

    def parse_policy_params(self) -> None:
        """Parse optimizer vector into STARFIT scalars."""
        self.validate_policy_params()
        (
            self.NORhi_mu, self.NORhi_min, self.NORhi_max,
            self.NORhi_alpha, self.NORhi_beta,
            self.NORlo_mu, self.NORlo_min, self.NORlo_max,
            self.NORlo_alpha, self.NORlo_beta,
            self.Release_alpha1, self.Release_alpha2,
            self.Release_beta1, self.Release_beta2,
            self.Release_c, self.Release_p1, self.Release_p2,
        ) = self.policy_params

        # allow percent inputs like the old code
        self.NORhi_min = self._pct_to_unit(self.NORhi_min)
        self.NORhi_max = self._pct_to_unit(self.NORhi_max)
        self.NORlo_min = self._pct_to_unit(self.NORlo_min)
        self.NORlo_max = self._pct_to_unit(self.NORlo_max)

    def _effective_starfit_name(self, reservoir_name: str) -> str:
        """Row to read coefficients from. DRBC-modified reservoirs use the
        'modified_<reservoir>' row, same as parameters/starfit.py."""
        name = str(reservoir_name)
        return ("modified_" + name) if name in modified_starfit_reservoir_list else name

    def _read_starfit_row(self, reservoir_name=None, csv_path=None):
        """Read the (possibly modified_<reservoir>-remapped) CSV row for this reservoir."""
        path = csv_path or os.path.join(CONFIG_DIR, "drb_model_istarf_conus.csv")
        if not os.path.isabs(path):
            path = os.path.abspath(path) # make absolute for worker nodes
        if not os.path.exists(path):
            raise FileNotFoundError(f"STARFIT CSV not found at: {path}")

        df = pd.read_csv(path)
        starfit_name = self._effective_starfit_name(reservoir_name)
        row = df.loc[df["reservoir"] == starfit_name]
        if row.empty:
            raise ValueError(f"STARFIT parameters not found for '{starfit_name}' in {path}.")
        return row.iloc[0]

    def load_starfit_constants(self, reservoir_name=None, csv_path=None):
        """Load the fixed constants (I_bar, Release_max/min) from CSV, leaving the 17
        policy params untouched. Use this on the optimization path so Borg's decision
        variables survive. I_bar standardizes inflow; Release_max/min feed R_max/R_min
        reconstruction in set_context.
        """
        rec = self._read_starfit_row(reservoir_name, csv_path)

        self.Release_max = float(rec["Release_max"]) if "Release_max" in rec and pd.notna(rec["Release_max"]) else None
        self.Release_min = float(rec["Release_min"]) if "Release_min" in rec and pd.notna(rec["Release_min"]) else None

        self.I_bar = float(rec["Adjusted_MEANFLOW_MGD"]) if pd.notna(rec["Adjusted_MEANFLOW_MGD"]) \
                    else float(rec["GRanD_MEANFLOW_MGD"])

        # optional log file, same as old behavior
        self.log_path = f"STARFIT_release_log_{self.reservoir_name}.txt"
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

    def load_starfit_params(self, reservoir_name=None, csv_path=None):
        """Load the 17 published coefficients plus constants from CSV.

        Reference / non-optimization path: overwrites self.policy_params with the
        coefficients from the (possibly remapped) row. On the optimization path use
        load_starfit_constants instead, so Borg's decision variables are not clobbered.
        """
        rec = self._read_starfit_row(reservoir_name, csv_path)
        self.policy_params = [float(rec[k]) for k in STARFIT_PARAM_NAMES]
        self.parse_policy_params()
        self.load_starfit_constants(reservoir_name, csv_path)
        
    
    def test_nor_constraint(self) -> bool:
            """
            Return False if STARFIT violates NOR structure:

            Scalar checks:
            --------------
            - 0 <= NORlo_min < NORlo_max <= 1
            - 0 <= NORhi_min < NORhi_max <= 1
            - NORlo_min <= NORhi_min
            - NORlo_max <= NORhi_max

            Daily curve checks (doy = 1..366):
            ----------------------------------
            Using the same harmonic form as in `evaluate()`:
                NOR_hi_raw = mu + alpha * s2 + beta * c2
                NOR_lo_raw = mu + alpha * s2 + beta * c2

            After clipping to [min,max], we require for all days:
                0 <= NOR_lo(t) <= NOR_hi(t) <= 1
            """
            tol = 1e-8
            log_file = "violated_params.log"

            # ---- 1) Scalar bounds and ordering ---------------------------------
            # mins/maxs are already converted to unit space [0,1] by parse_policy_params()
            # via _pct_to_unit, so here we treat them as unit values.
            if any(v is None for v in [
                self.NORhi_min, self.NORhi_max,
                self.NORlo_min, self.NORlo_max,
            ]):
                raise RuntimeError("STARFIT parameters must be parsed before test_nor_constraint().")

            # basic 0–1 bounds
            if not (0.0 <= self.NORlo_min <= 1.0 and 0.0 <= self.NORlo_max <= 1.0 and
                    0.0 <= self.NORhi_min <= 1.0 and 0.0 <= self.NORhi_max <= 1.0):
                with open(log_file, "a") as f:
                    f.write(f"\n[SCALAR-BOUNDS] Violation for {self.reservoir_name} at {pd.Timestamp.now()}:\n")
                    f.write(f"policy_params = {self.policy_params}\n")
                    f.write(f"NORhi_min={self.NORhi_min}, NORhi_max={self.NORhi_max}\n")
                    f.write(f"NORlo_min={self.NORlo_min}, NORlo_max={self.NORlo_max}\n")
                    f.write("--------\n")
                return False

            # ordering: min < max
            if not (self.NORhi_min + tol < self.NORhi_max and
                    self.NORlo_min + tol < self.NORlo_max):
                with open(log_file, "a") as f:
                    f.write(f"\n[SCALAR-ORDER] Violation for {self.reservoir_name} at {pd.Timestamp.now()}:\n")
                    f.write(f"policy_params = {self.policy_params}\n")
                    f.write(f"NORhi_min={self.NORhi_min}, NORhi_max={self.NORhi_max}\n")
                    f.write(f"NORlo_min={self.NORlo_min}, NORlo_max={self.NORlo_max}\n")
                    f.write("--------\n")
                return False

            # low band must not sit above high band at the scalar level
            if not (self.NORlo_min <= self.NORhi_min + tol and
                    self.NORlo_max <= self.NORhi_max + tol):
                with open(log_file, "a") as f:
                    f.write(f"\n[SCALAR-CROSS] Violation for {self.reservoir_name} at {pd.Timestamp.now()}:\n")
                    f.write(f"policy_params = {self.policy_params}\n")
                    f.write(f"NORhi_min={self.NORhi_min}, NORhi_max={self.NORhi_max}\n")
                    f.write(f"NORlo_min={self.NORlo_min}, NORlo_max={self.NORlo_max}\n")
                    f.write("--------\n")
                return False

            # ---- 2) Daily curve checks over doy=1..366 --------------------------
            doys = np.arange(1.0, 367.0, dtype=float)
            hi_vals = []
            lo_vals = []

            for doy in doys:
                s2, c2, s4, c4 = self._seasonal_terms(doy)

                NOR_hi_raw = self.NORhi_mu + self.NORhi_alpha * s2 + self.NORhi_beta * c2
                NOR_lo_raw = self.NORlo_mu + self.NORlo_alpha * s2 + self.NORlo_beta * c2

                NOR_hi = float(np.clip(NOR_hi_raw, self.NORhi_min * 100.0, self.NORhi_max * 100.0)) / 100.0
                NOR_lo = float(np.clip(NOR_lo_raw, self.NORlo_min * 100.0, self.NORlo_max * 100.0)) / 100.0

                hi_vals.append(NOR_hi)
                lo_vals.append(NOR_lo)

            hi_vals = np.asarray(hi_vals)
            lo_vals = np.asarray(lo_vals)

            # constraints for all days
            bad_lo = np.any(lo_vals < -tol)
            bad_hi = np.any(hi_vals > 1.0 + tol)
            cross  = np.any(lo_vals > hi_vals + tol)

            if bad_lo or bad_hi or cross:
                with open(log_file, "a") as f:
                    f.write(f"\n[CURVE-DAILY] Violation for {self.reservoir_name} at {pd.Timestamp.now()}:\n")
                    f.write(f"policy_params = {self.policy_params}\n")
                    f.write(f"min(lo)={lo_vals.min():.4f}, max(lo)={lo_vals.max():.4f}\n")
                    f.write(f"min(hi)={hi_vals.min():.4f}, max(hi)={hi_vals.max():.4f}\n")
                    f.write("--------\n")
                return False

            return True

    def _reconstruct_release_limits(self) -> None:
        """Set R_max/R_min the way parameters/starfit.py does, overriding context.

        R_max = max_discharges[name], else 999999 if remove_R_max, else (Release_max+1)*I_bar.
        R_min = conservation_releases[name], else (Release_min+1)*I_bar.
        Limits that can't be reconstructed are left at the context value.
        """
        # lazy import avoids a circular import
        from pywrdrb.parameters.lower_basin_ffmp import (
            conservation_releases,
            max_discharges,
        )
        name = str(self.reservoir_name)
        # R_max
        if name in max_discharges:
            self.release_max = float(max_discharges[name])
        elif self.remove_R_max:
            self.release_max = 999999.0
        elif self.Release_max is not None and self.I_bar not in (None, 0.0):
            self.release_max = float((self.Release_max + 1) * self.I_bar)
        # R_min
        if name in conservation_releases:
            self.release_min = float(conservation_releases[name])
        elif self.Release_min is not None and self.I_bar not in (None, 0.0):
            self.release_min = float((self.Release_min + 1) * self.I_bar)

    def set_context(self, **ctx):
        """
        Set context via the base class, then fill in I_bar from istarf_capacity.csv
        and reconstruct R_max/R_min to match parameters/starfit.py.
        """
        ret = super().set_context(**ctx) if hasattr(super(), "set_context") else None
        self._ensure_I_bar_from_capacity()
        self._reconstruct_release_limits()  # override context R_max/R_min
        return ret
        
    # ---------- Pywr-path: CSV row params ----------
    def assign_policy_params(
        self,
        row: Mapping[str, Any],
        *,
        set_context_from_row: bool = False,
    ) -> None:
        """
        Assign STARFIT parameters from a pandas Series / dict-like row.

        Expects columns named exactly as in the docstring. For context:
          - S_cap or Adjusted_CAP_MG / GRanD_CAP_MG
          - I_min, I_max (for normalization)
          - optional R_min, R_max
        """
        # pull policy scalars
        names = [
            "NORhi_mu", "NORhi_min", "NORhi_max", "NORhi_alpha", "NORhi_beta",
            "NORlo_mu", "NORlo_min", "NORlo_max", "NORlo_alpha", "NORlo_beta",
            "Release_alpha1", "Release_alpha2", "Release_beta1", "Release_beta2",
            "Release_c", "Release_p1", "Release_p2",
        ]
        vals = []
        for k in names:
            if k not in row:
                raise KeyError(f"assign_policy_params: missing '{k}'")
            vals.append(float(row[k]))
        self.policy_params = vals
        self.parse_policy_params()

        # Release_max/min feed R_max/R_min reconstruction in set_context
        if "Release_max" in row and pd.notna(row["Release_max"]):
            self.Release_max = float(row["Release_max"])
        if "Release_min" in row and pd.notna(row["Release_min"]):
            self.Release_min = float(row["Release_min"])

        if set_context_from_row:
            # capacity
            S_cap = row.get("S_cap", row.get("Adjusted_CAP_MG", row.get("GRanD_CAP_MG", None)))
            if S_cap is None:
                raise KeyError("assign_policy_params: missing S_cap/Adjusted_CAP_MG/GRanD_CAP_MG")

            # inflow bounds
            I_min = row.get("I_min")
            I_max = row.get("I_max")
            if I_min is None or I_max is None:
                raise KeyError("assign_policy_params: missing I_min / I_max")

            # release limits (optional)
            R_min = float(row.get("R_min", 0.0))
            R_max = float(row.get("R_max", 1e12))

            self.set_context(
                release_min=float(R_min),
                release_max=float(R_max),
                storage_capacity=float(S_cap),
                x_min=(0.0, float(I_min), 1.0),
                x_max=(float(S_cap), float(I_max), 366.0),
            )

    def assign_from_df(self, df, reservoir_name: str, policy_id: str = "default", **kwargs) -> None:
        """
        Convenience for MultiIndex DataFrame indexed by (reservoir_name, policy_id).
        Falls back to (reservoir_name, 'default') if the exact row is missing.
        """
        if df.index.nlevels < 2:
            raise ValueError("assign_from_df expects a MultiIndex with (reservoir_name, policy_id).")
        key = (reservoir_name, policy_id)
        if key not in df.index:
            fallback = (reservoir_name, "default")
            if fallback in df.index:
                key = fallback
            else:
                raise KeyError(f"Parameters not found for {reservoir_name}/{policy_id} (and no 'default').")
        row = df.loc[key]
        self.assign_policy_params(row, **kwargs)

    # ---------- core math ----------
    def _seasonal_terms(self, doy: float):
        """Return sin/cos terms for day-of-year in [1,366]."""
        c = np.pi * (doy + self.WATER_YEAR_OFFSET) / 365.0
        # match legacy: sin(2c), cos(2c), sin(4c), cos(4c)
        s2 = np.sin(2.0 * c)
        c2 = np.cos(2.0 * c)
        s4 = np.sin(4.0 * c)
        c4 = np.cos(4.0 * c)
        return s2, c2, s4, c4

    def _target_release(self, storage: float, inflow: float, day_of_year: float) -> float:
        """
        Unconstrained STARFIT target release (MGD), matching
        parameters/starfit.py.calculate_target_release().

        Runs on raw inputs: S_hat = S / S_cap may exceed 1, inflow is not capped,
        and the spill term uses raw inflow.
        """
        if self.I_bar is None:
            raise RuntimeError("I_bar not set. Use assign_policy_params(...) or load_starfit_params(...).")

        S_cap = float(self.storage_capacity)
        S_hat = float(storage) / S_cap
        I = float(inflow)
        I_hat = (I - self.I_bar) / self.I_bar

        # seasonal baseline
        s2, c2, s4, c4 = self._seasonal_terms(float(day_of_year))
        harmonic = (
            self.Release_alpha1 * s2 +
            self.Release_alpha2 * s4 +
            self.Release_beta1  * c2 +
            self.Release_beta2  * c4
        )

        # NOR bands: harmonic is in percent, so clip in percent then /100 -> [0,1]
        NOR_hi_raw = self.NORhi_mu + self.NORhi_alpha * s2 + self.NORhi_beta * c2
        NOR_lo_raw = self.NORlo_mu + self.NORlo_alpha * s2 + self.NORlo_beta * c2
        NOR_hi = float(np.clip(NOR_hi_raw, self.NORhi_min * 100.0, self.NORhi_max * 100.0)) / 100.0
        NOR_lo = float(np.clip(NOR_lo_raw, self.NORlo_min * 100.0, self.NORlo_max * 100.0)) / 100.0

        # storage/inflow adjustment (bare denominator, no +1e-6)
        A_t = (S_hat - NOR_lo) / NOR_hi
        epsilon = self.Release_c + self.Release_p1 * A_t + self.Release_p2 * I_hat

        R_max = float(self.release_max if self.release_max is not None else 999999.0)
        R_min = float(self.release_min if self.release_min is not None else 0.0)

        if NOR_lo <= S_hat <= NOR_hi:
            target = min(self.I_bar * (harmonic + epsilon + 1.0), R_max)
        elif S_hat > NOR_hi:
            target = min((S_cap * (S_hat - NOR_hi) + I * 7.0) / 7.0, R_max)
        else:
            if self.linear_below_NOR:
                target = (self.I_bar * (harmonic + epsilon + 1.0)) * (S_hat / NOR_lo)
                target = max(target, R_min)
            else:
                target = R_min
        return float(target)

    def evaluate(self, X_norm: Sequence[float]) -> float:
        """
        Return z in [0,1] for normalized inputs [S_norm, I_norm, D_norm].
        Used for plotting; converts back to raw units and calls _target_release.
        """
        if self.x_min is None or self.x_max is None:
            raise RuntimeError("set_context(...) must be called before evaluate().")
        if self.I_bar is None:
            raise RuntimeError("I_bar not set. Use assign_policy_params(...) or set_mean_inflow(...).")

        X = np.asarray(X_norm, dtype=float)
        if len(X) != self.n_inputs:
            raise AssertionError(f"Expected {self.n_inputs} inputs; got {len(X)}.")
        if not np.all((0.0 <= X) & (X <= 1.0)):
            raise AssertionError(f"Inputs must be in [0,1]. Got {X}.")

        # back to raw units
        S = float(X[0]) * float(self.storage_capacity)
        I = self.x_min[1] + float(X[1]) * (self.x_max[1] - self.x_min[1])
        doy = 1.0 + float(X[2]) * (366.0 - 1.0)

        target = self._target_release(S, I, doy)
        R_max = float(self.release_max if self.release_max is not None else 999999.0)
        z = target / R_max if R_max > 0 else 0.0
        return max(0.0, min(1.0, float(z)))

    def get_release(self, storage: float, inflow: float, day_of_year: float) -> float:
        """
        Release in MGD for raw inputs, matching parameters/starfit.py.value():
        clamp the target to available water and capacity, then max(0, .).
        """
        S_t = float(storage)
        I_t = float(inflow)
        D_t = float(day_of_year)

        target_release = self._target_release(S_t, I_t, D_t)

        available_water = I_t + S_t
        min_required = available_water - self.storage_capacity
        release_t = max(min(target_release, available_water), min_required)
        return max(0.0, release_t)

    # ---------- utilities / plots ----------
    def calculate_weekly_NOR(self):
        weekly_NORhi = []
        weekly_NORlo = []

        dummy_dates = pd.date_range("2020-10-01", periods=52, freq='W')
        for dt in dummy_dates:
            doy = (dt.timetuple().tm_yday) % 365
            c = pi * doy / 365

            NOR_hi = self.NORhi_mu + self.NORhi_alpha * sin(c * 2) + self.NORhi_beta * cos(c * 2)
            NOR_lo = self.NORlo_mu + self.NORlo_alpha * sin(c * 2) + self.NORlo_beta * cos(c * 2)
            # clip in percent, then /100 -> [0,1] (see _target_release)
            weekly_NORhi.append(np.clip(NOR_hi, self.NORhi_min * 100.0, self.NORhi_max * 100.0) / 100.0)
            weekly_NORlo.append(np.clip(NOR_lo, self.NORlo_min * 100.0, self.NORlo_max * 100.0) / 100.0)

        self.weekly_NORhi_array = np.array(weekly_NORhi)
        self.weekly_NORlo_array = np.array(weekly_NORlo)

    # ---------- optional: surface plot ----------
    def plot_surfaces_for_different_weeks(self, fname=None, save=False, *, grid=30, weeks=None, n_weeks=5):
        """
        3D surfaces: X=inflow (normalized), Y=storage (normalized), Z=policy output,
        with multiple 'week' (D_norm) slices. Uses STARFIT.evaluate() (requires context + I_bar).
        """

        # sanity checks for STARFIT
        if self.x_min is None or self.x_max is None:
            raise RuntimeError("STARFIT: set_context(...) must be called before plotting.")
        if self.I_bar is None:
            raise RuntimeError("STARFIT: I_bar must be set (assign_policy_params(...) or load_starfit_params(...)).")

        inflow  = np.linspace(0.0, 1.0, grid)
        storage = np.linspace(0.0, 1.0, grid)
        weeks   = np.linspace(0.0, 1.0, n_weeks) if weeks is None else np.asarray(weeks, float)

        I, S = np.meshgrid(inflow, storage, indexing="xy")

        fig = plt.figure(figsize=(12, 9))
        ax  = fig.add_subplot(111, projection='3d')
        cmap = plt.cm.viridis

        for idx, week in enumerate(weeks):
            Z = np.zeros(I.shape, dtype=float)
            for i in range(I.shape[0]):
                for j in range(I.shape[1]):
                    Z[i, j] = float(self.evaluate([S[i, j], I[i, j], week]))
            color = cmap(idx / max(1, len(weeks)-1))
            ax.plot_surface(I, S, Z, color=color, alpha=0.6, linewidth=0, antialiased=True)

        ax.set_xlabel('Inflow')
        ax.set_ylabel('Storage')
        ax.set_zlabel('Release')  # keep legacy label (unitless z)
        ax.set_title(f'STARFIT: 3D Policy Output for Different Weeks')

        custom = [plt.Line2D([0],[0], linestyle="none", marker='s', markersize=10,
                            markerfacecolor=cmap(i / max(1, len(weeks)-1)), alpha=0.6)
                for i in range(len(weeks))]
        ax.legend(custom, [f'Week {w:.2f}' for w in weeks], loc='upper left', framealpha=0.9)

        if save:
            assert fname is not None, "Filename must be provided to save the plot."
            plt.savefig(fname, dpi=300)
        plt.show()

    def plot(self, N: int = 41) -> None:
        return self.plot_surfaces_for_different_weeks(grid=N)
