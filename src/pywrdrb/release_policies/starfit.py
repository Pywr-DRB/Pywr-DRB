import numpy as np
import matplotlib.pyplot as plt
from typing import Mapping, Any, Optional, Sequence

from pywrdrb.release_policies.abstract_policy import AbstractPolicy
from pywrdrb.release_policies.config import (
    policy_n_params,
    policy_param_bounds,
    n_starfit_inputs,   # should be 3 for [S, I, D]
)


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

    def __init__(self, policy_params):
        super().__init__(policy_params=policy_params)

        self.n_inputs = int(n_starfit_inputs)  # expected 3 for [S, I, D]
        self.n_params = policy_n_params["STARFIT"]
        self.param_bounds = policy_param_bounds["STARFIT"]

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
        self.I_bar: Optional[float] = None

        # optional behavior toggle (kept for parity with legacy)
        self.linear_below_NOR: bool = False

        if policy_params is not None:
            self.parse_policy_params()

    # ---------- helpers ----------
    @staticmethod
    def _pct_to_unit(v):
        """Allow 0–100 or 0–1 inputs for min/max; convert to 0–1."""
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
        ) = map(float, self.policy_params)

        # allow percent-style inputs for bounds
        self.NORhi_min = self._pct_to_unit(self.NORhi_min)
        self.NORhi_max = self._pct_to_unit(self.NORhi_max)
        self.NORlo_min = self._pct_to_unit(self.NORlo_min)
        self.NORlo_max = self._pct_to_unit(self.NORlo_max)

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
          - Adjusted_MEANFLOW_MGD or GRanD_MEANFLOW_MGD (for I_bar)
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

        # set mean inflow for standardization
        I_bar = row.get("Adjusted_MEANFLOW_MGD", row.get("GRanD_MEANFLOW_MGD", None))
        if I_bar is None:
            raise KeyError("assign_policy_params: missing Adjusted_MEANFLOW_MGD / GRanD_MEANFLOW_MGD")
        self.I_bar = float(I_bar)

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

    # Optional: allow setting I_bar directly if needed
    def set_mean_inflow(self, I_bar: float) -> None:
        self.I_bar = float(I_bar)

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

        Xn = self._normalize(S_t, I_t, D_t)
        z = self.evaluate(Xn)              # [0,1]
        release = z * float(self.release_max)
        return self.enforce_constraints(release, available=S_t + I_t)

    # ---------- utilities / plots ----------
    def calculate_weekly_NOR(self, weeks: int = 52) -> tuple[np.ndarray, np.ndarray]:
        xs = np.arange(weeks, dtype=float)
        # map week to representative day-of-year
        doy = 1.0 + xs * (366.0 / weeks)
        s2, c2, s4, c4 = self._seasonal_terms(doy)
        hi_raw = self.NORhi_mu + self.NORhi_alpha * s2 + self.NORhi_beta * c2
        lo_raw = self.NORlo_mu + self.NORlo_alpha * s2 + self.NORlo_beta * c2
        hi = np.clip(hi_raw, self.NORhi_min, self.NORhi_max)
        lo = np.clip(lo_raw, self.NORlo_min, self.NORlo_max)
        return lo, hi

    def plot(self, N: int = 41) -> None:
        """Plot a D_norm=0.5 slice of z(S_norm, I_norm)."""
        xs = np.linspace(0.0, 1.0, N)
        ys = np.linspace(0.0, 1.0, N)
        Z = np.zeros((N, N))
        for i, s in enumerate(xs):
            for j, q in enumerate(ys):
                Z[i, j] = self.evaluate([s, q, 0.5])

        X, Y = np.meshgrid(xs, ys)
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z.T, alpha=0.7)
        ax.set_xlabel("S_norm"); ax.set_ylabel("I_norm"); ax.set_zlabel("z")
        ax.set_title("STARFIT policy surface (D_norm=0.5)")
        plt.tight_layout()
        plt.show()
