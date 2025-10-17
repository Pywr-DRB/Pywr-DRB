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
    n_starfit_inputs,   # should be 3 for [S, I, D]
    DATA_DIR,
    CONFIG_DIR,
)
from pywrdrb.path_manager import get_pn_object

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

    Interface (matches PWL/RBF)
    ---------------------------
    - Call `set_context(release_min, release_max, storage_capacity, x_min, x_max)` once.
      * x_min/x_max define min–max normalization for inputs [S, I, D].
      * Typically: x_min = (0, I_min, 1), x_max = (S_cap, I_max, 366).
    - `evaluate([S_norm, I_norm, D_norm]) -> z in [0,1]`
    - `get_release(S, I, D)` normalizes with the base class, then scales by `release_max`
      and enforces constraints via `enforce_constraints(..., available=S+I)`.

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
    - NOR min/max may be given as percentages in CSV; this class accepts either
      unit interval (0–1) or percent (0–100) and converts automatically.
    - Standardized inflow uses I_hat = (I - I_bar) / I_bar (I in original units).
      We reconstruct I from I_norm using x_min/x_max, so `I_bar` must be set
      (via `assign_policy_params` or `set_mean_inflow`).
    """

    def __init__(self, policy_params, reservoir_name: Optional[str] = None,):

        super().__init__(policy_params=policy_params)

        self.reservoir_name = reservoir_name

        self.n_inputs = int(n_starfit_inputs)  # expected 3 for [S, I, D]
        self.n_params = policy_n_params["STARFIT"]
        self.param_bounds = policy_param_bounds["STARFIT"]
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

        # optional behavior toggle (kept for parity with legacy)
        self.linear_below_NOR: bool = False

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
        v = float(v)
        return v / 100.0 if v > 1.0 else v
    
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

    def load_starfit_params(self, reservoir_name=None, csv_path=None):
        """Load I_bar (and set a log path) like the old code, but not in __init__."""
        if reservoir_name is not None:
            self.reservoir_name = reservoir_name
        if not self.reservoir_name:
            raise ValueError("load_starfit_params requires reservoir_name.")

        path = csv_path or os.path.join(CONFIG_DIR, "drb_model_istarf_conus.csv")
        if not os.path.isabs(path):
            path = os.path.abspath(path) # make absolute for worker nodes
        if not os.path.exists(path):
            raise FileNotFoundError(f"STARFIT CStV not found at: {path}")

        df = pd.read_csv(path)
        row = df.loc[df["reservoir"] == self.reservoir_name]
        if row.empty:
            raise ValueError(f"STARFIT parameters not found for '{self.reservoir_name}' in {path}.")

        rec = row.iloc[0]
        self.I_bar = float(rec["Adjusted_MEANFLOW_MGD"]) if pd.notna(rec["Adjusted_MEANFLOW_MGD"]) \
                    else float(rec["GRanD_MEANFLOW_MGD"])

        # optional log file, same as old behavior
        self.log_path = f"STARFIT_release_log_{self.reservoir_name}.txt"
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

    def test_nor_constraint(self):
        self.calculate_weekly_NOR()
        if np.any(self.weekly_NORhi_array < self.weekly_NORlo_array):
            # Optional: Write violating parameters to a file
            with open("violated_params.log", "a") as f:
                f.write(f"\nViolation for {self.reservoir_name} at {pd.Timestamp.now()}:\n")
                f.write(f"{self.policy_params}\n")
                f.write("--------\n")
            return False
        else:
            return True
        
    def set_context(self, **ctx):
        """
        Set STARFIT context (min/max releases, capacity, normalization) via base class,
        then ensure I_bar from istarf_capacity.csv (keyed by reservoir only).
        """
        # call base set_context to populate release_min/max, storage_capacity, x_min/x_max
        ret = super().set_context(**ctx) if hasattr(super(), "set_context") else None
        # now make sure we have I_bar
        self._ensure_I_bar_from_capacity()
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

    def evaluate(self, X_norm: Sequence[float]) -> float:
        """
        Evaluate STARFIT at normalized inputs and return z in [0,1].

        X_norm = [S_norm, I_norm, D_norm] where each in [0,1].
        Uses I_bar (mean inflow) to standardize inflow internally.
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

        # unpack normalized inputs
        S_hat = float(X[0])  # already S / S_cap
        I_norm = float(X[1])
        D_norm = float(X[2])

        # reconstruct I in original units to compute I_hat
        I = self.x_min[1] + I_norm * (self.x_max[1] - self.x_min[1])
        I_hat = (I - self.I_bar) / self.I_bar

        # day-of-year in 1..366 from normalized D
        doy = 1.0 + D_norm * (366.0 - 1.0)

        # harmonic seasonal baseline
        s2, c2, s4, c4 = self._seasonal_terms(doy)
        harmonic = (
            self.Release_alpha1 * s2 +
            self.Release_alpha2 * s4 +
            self.Release_beta1  * c2 +
            self.Release_beta2  * c4
        )

        # seasonal NOR envelope
        NOR_hi_raw = self.NORhi_mu + self.NORhi_alpha * s2 + self.NORhi_beta * c2
        NOR_lo_raw = self.NORlo_mu + self.NORlo_alpha * s2 + self.NORlo_beta * c2

        NOR_hi = float(np.clip(NOR_hi_raw, self.NORhi_min, self.NORhi_max))
        NOR_lo = float(np.clip(NOR_lo_raw, self.NORlo_min, self.NORlo_max))

        # adjustment term
        A_t = (S_hat - NOR_lo) / (NOR_hi + 1e-6)
        epsilon = self.Release_c + self.Release_p1 * A_t + self.Release_p2 * I_hat

        # target release in original units (MGD)
        # use release_max (context) as cap later; for within-NOR branch, the
        # Turner-style formula multiplies by I_bar and adds +1*I_bar baseline
        if NOR_lo <= S_hat <= NOR_hi:
            target = self.I_bar * (harmonic + epsilon + 1.0)
        elif S_hat > NOR_hi:
            # spill-like logic; weekly smoothing per legacy
            S_cap = float(self.storage_capacity)
            target = (S_cap * (S_hat - NOR_hi) + I * 7.0) / 7.0
        else:
            if self.linear_below_NOR and NOR_lo > 0.0:
                base = self.I_bar * (harmonic + epsilon + 1.0)
                target = base * (S_hat / NOR_lo)
            else:
                target = float(self.release_min if self.release_min is not None else 0.0)

        # convert to z in [0,1] by scaling with release_max
        R_max = float(self.release_max if self.release_max is not None else 1.0)
        z = target / R_max
        return max(0.0, min(1.0, float(z)))

    def get_release(self, storage: float, inflow: float, day_of_year: float) -> float:
        """
        Normalize via base class, scale by release_max, then clamp with constraints and availability.
        """
        S_t = float(storage)
        I_t = float(inflow)
        D_t = float(day_of_year)

        forced = self._storage_safety_override(S_t, I_t)
        if forced is not None:
            return self.enforce_constraints(forced, available=S_t + I_t)

        Xn = self._normalize(S_t, I_t, D_t)
        z = self.evaluate(Xn)              # [0,1]
        release = z * float(self.release_max)
        return self.enforce_constraints(release, available=S_t + I_t)

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
            # NOTE: parse_policy_params() already converts *_min/max to [0,1] via _pct_to_unit.
            # Keep everything in [0,1] here (no "/100").
            weekly_NORhi.append(np.clip(NOR_hi, self.NORhi_min, self.NORhi_max))
            weekly_NORlo.append(np.clip(NOR_lo, self.NORlo_min, self.NORlo_max))

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
