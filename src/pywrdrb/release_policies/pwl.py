"""
Piecewise-Linear (PWL) reservoir release policy.

Overview
--------
`PWL` implements a simple, interpretable operating rule where the final release
decision is the average of three **independent 1-D piecewise-linear (PWL) mappings**
defined over *normalized* inputs:
  1) storage S_norm ∈ [0, 1]
  2) inflow  I_norm ∈ [0, 1]
  3) season/day-of-year D_norm ∈ [0, 1]

Each 1-D PWL is specified by:
- `M` segments (configured globally as `n_segments`).
- `M-1` internal breakpoints (x₁..x_{M-1}) in (0,1), with x₀=0 and x_M=1 implied.
- `M` segment angles (θ₁..θ_M). Slopes are `tan(θ_k)`. Intercepts are computed to
  ensure continuity at each breakpoint.

The policy:
1) Normalizes raw inputs (S, I, D) using the context provided by
   `AbstractPolicy.set_context(...)`.
2) Evaluates the three PWLs to obtain z_S, z_I, z_D (each clamped to [0,1]).
3) Averages them: `z = (z_S + z_I + z_D)/3`.
4) Scales by the policy maximum: `release_target = z * release_max`.
5) Enforces constraints (policy bounds & physical availability S + I) via
   `enforce_constraints(...)`.

Units & Inputs
--------------
- Storage S: **MG**
- Inflow  I: **MGD**
- Release R: **MGD**
- Day-of-year D: integer in [1, 366]; normalized using the day bounds in context.
Normalization is handled by `AbstractPolicy` using `x_min`/`x_max` per input.

Parameterization (two supported paths)
--------------------------------------
1) **Optimizer vector path** (flattened numeric vector):
   - Expected length = `(2*M - 1) * n_pwl_inputs`
   - Packing order (three contiguous blocks of equal length):
       `[storage block | inflow block | day block]`
   - Within each block (length `2*M-1`):
       `[x₁, …, x_{M-1}, θ₁, …, θ_M]`
     where `x_k ∈ (0,1)` are strictly increasing; θ are angles (radians).

   Use:
     - `validate_policy_params()` to check shape and bounds.
     - `parse_policy_params()` to convert the flat vector into three PWLs
       (bounds, slopes, intercepts).

2) **CSV row path** (Pywr integration):
   - Provide a pandas‐like `row` with columns:

     Storage block:
       `storage_x1..storage_x{M-1}`, `storage_theta1..storage_theta{M}`

     Inflow block:
       `inflow_x1..inflow_x{M-1}`,  `inflow_theta1..inflow_theta{M}`

     Season/Day block:
       `season_x1..season_x{M-1}`,  `season_theta1..season_theta{M}`

   - Call `assign_policy_params(row, set_context_from_row=False/True)`.
     If `set_context_from_row=True`, the row must also provide context fields:
       - `S_cap` (or `Adjusted_CAP_MG` / `GRanD_CAP_MG`)
       - `I_min`, `I_max`
       - optional `R_min`, `R_max`
     These are forwarded to `set_context(...)` as:
       `x_min = (0.0, I_min, 1.0)`, `x_max = (S_cap, I_max, 366.0)`.

Core Evaluation
---------------
- Segment math (on a single axis):
  For `x_norm ∈ [0,1]`, find segment i such that `x_i ≤ x_norm < x_{i+1}` and compute
  `f(x_norm) = m_i * (x_norm - x_i) + b_i`,
  where `m_i = tan(θ_i)` and intercepts `b_i` are constructed for continuity.
- The class evaluates each axis independently, clamps each to [0,1], averages,
  scales by `release_max`, and then calls `enforce_constraints(...)`.

Configuration Hooks
-------------------
- `n_segments` and `n_pwl_inputs` are imported from config. Default usage assumes
  `n_pwl_inputs == 3` for (S, I, D).
- Parameter bounds for the optimizer vector are taken from
  `policy_param_bounds["PWL"]`. Count is `policy_n_params["PWL"]`.

API Summary
-----------
- `__init__(policy_params)`: optional flat vector; will `parse_policy_params()` if provided.
- `validate_policy_params()`: shape/value checks against bounds and expected length.
- `parse_policy_params()`: splits the flat vector into three PWLs (storage/inflow/day).
- `assign_policy_params(row, set_context_from_row=False)`: Pywr CSV row interface.
- `evaluate(X_norm) -> z`: maps normalized (S,I,D) to z ∈ [0,1].
- `get_release(storage, inflow, day_of_year) -> float`: normalize → evaluate → scale →
  `enforce_constraints`; returns **MGD**.
- `plot(N=41)`: quick surface plot of z over (S_norm, I_norm) with D_norm fixed at 0.5.

Practical Notes & Guardrails
----------------------------
- **Monotonic breakpoints**: `x₁ < x₂ < … < x_{M-1}` in (0,1) ensure well-defined segments.
- **Angle stability**: avoid θ near ±π/2 to prevent extreme slopes; if needed, restrict
  θ to `(-π/2 + ε, π/2 - ε)`.
- **Clamping**: each axis output and the final average are clamped to [0,1], limiting
  the impact of minor extrapolations or steep local slopes.
- **Interpretability**: keeping M small (e.g., 3–5) yields smooth, explainable rule shapes.

Example (optimizer vector path)
-------------------------------
>>> M = 3  # segments
>>> # For each axis: [x1, x2, theta1, theta2, theta3]  -> length 5
>>> # Whole vector packs: storage(5) + inflow(5) + day(5) = 15 params
>>> params = [
...   0.3, 0.7,  0.10, 0.00, -0.05,   # storage
...   0.2, 0.6,  0.20, 0.05, -0.05,   # inflow
...   0.25,0.75, 0.00, 0.10,  0.00,   # day
... ]
>>> pol = PWL(policy_params=params)
>>> pol.set_context(
...   release_min=10.0, release_max=1200.0,
...   storage_capacity=18000.0,
...   x_min=(0.0, 0.0, 1.0), x_max=(18000.0, 3000.0, 366.0),
... )
>>> r = pol.get_release(storage=12000.0, inflow=250.0, day_of_year=200)

Change Log
----------
- 2025-09-24 — Detailed documentation added; clarified vector packing, CSV schema,
  continuity construction, and constraint order. Minor wording & guardrails.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Mapping, Any, Optional

from pywrdrb.release_policies.abstract_policy import AbstractPolicy
from pywrdrb.release_policies.config import policy_n_params, policy_param_bounds, drbc_conservation_releases
from pywrdrb.release_policies.config import n_segments, n_pwl_inputs


class PWL(AbstractPolicy):
    """
    A piecewise linear policy class.

    This policy defines a reservoir release function as a series of linear 
    segments. Each segment is defined by:
    - x_i: the lower bound of the segment (storage threshold)
    - θ_i: the angle of the segment relative to the x-axis

    The first segment always starts at x_0 = 0. The number of segments is 
    determined by the first value in `policy_params`. The remaining values 
    define x_i and θ_i for the segments.

    Attributes:
        Reservoir (Reservoir): The reservoir instance associated with the policy.
        n_segments (int): The number of linear segments in the policy.
        segment_x_bounds (list): Storage thresholds defining segment boundaries.
        segment_theta_vals (list): Angles defining segment slopes.
        slopes (list): Computed slopes for each segment.
        intercepts (list): Computed intercepts for each segment.
    """

    def __init__(self,
                 policy_params):
        """
        Initializes the PiecewiseLinear policy.

        Args:
            Reservoir (Reservoir): The reservoir instance.
            policy_params (list): A list of policy parameters where:
                - policy_params[0] defines the number of segments (M).
                - The next M-1 values define x_i (excluding x_0 = 0).
                - The last M values define θ_i.
        """
        
        # Policy parameters
        super().__init__(policy_params=policy_params)
        self.n_segments = n_segments
        self.n_inputs = n_pwl_inputs   # should be 3 for [S,I,D]
        self.n_params = policy_n_params["PWL"]
        self.param_bounds = policy_param_bounds["PWL"]

        # storage/inflow/day PWL pieces set by parse or assign
        self.storage_bounds = self.storage_slopes = self.storage_intercepts = None
        self.inflow_bounds  = self.inflow_slopes  = self.inflow_intercepts  = None
        self.day_bounds     = self.day_slopes     = self.day_intercepts     = None

        # If params provided (optimizer path), parse them now
        if policy_params is not None:
            self.parse_policy_params()

    # ---------- Optimizer-path: vector params ----------
    def validate_policy_params(self):
        """Validate the optimizer vector."""
        if self.policy_params is None:
            raise ValueError("PWL.policy_params is None; provide a vector or use assign_policy_params(...).")
        if len(self.policy_params) != self.n_params:
            raise AssertionError(
                f"PWL expected {self.n_params} parameters, got {len(self.policy_params)}."
            )
        for i, p in enumerate(self.policy_params):
            lo, hi = self.param_bounds[i]
            if not (lo <= p <= hi):
                raise AssertionError(f"Param idx {i} out of bounds {lo, hi}. Value: {p}")
        return

    def parse_policy_params(self):
        """Parse flattened vector into three 1-D PWLs (storage, inflow, day-of-year)."""
        self.validate_policy_params()

        def _parse_segment_params(segment_params, M=self.n_segments):
            """
            Decomposes the policy parameters into segment boundaries, slopes, and intercepts.
            
            Args:
                segment_params (list): A slice of the policy parameters, must be of length 2M-1 for M segments.
                
            Returns:
                tuple: A tuple containing:
                    - x_bounds (list): Segment boundaries, length n_segments-1.
                    - slopes (list): Slopes of the segments, length n_segments.
                    - intercepts (list): Intercepts of the segments, length n_segments.
            """
            # x bounds: [0, x1, x2, ..., 1]; slopes = tan(theta_k); intercepts continuous
            x_bounds   = [0.0] + list(segment_params[:M-1]) + [1.0]
            theta_vals = segment_params[M-1:]
            slopes     = [np.tan(theta) for theta in theta_vals]
            intercepts = [0.0]
            for k in range(1, M):
                dx = x_bounds[k] - x_bounds[k-1]
                intercepts.append(intercepts[k-1] + slopes[k-1] * dx)
            return x_bounds, slopes, intercepts

        # vector packs [storage block | inflow block | day block]
        block = len(self.policy_params) // 3
        s_params = self.policy_params[:block]
        i_params = self.policy_params[block:2*block]
        d_params = self.policy_params[2*block:]

        (self.storage_bounds, self.storage_slopes, self.storage_intercepts) = _parse_segment_params(s_params)
        (self.inflow_bounds,  self.inflow_slopes,  self.inflow_intercepts)  = _parse_segment_params(i_params)
        (self.day_bounds,     self.day_slopes,     self.day_intercepts)     = _parse_segment_params(d_params)

    # ---------- Pywr-path: CSV row params ----------
    def assign_policy_params(self, row: Mapping[str, Any], *, set_context_from_row: bool = False):
        """
        Define the three PWLs from a pandas Series / dict-like row.

        Expects per-axis columns for M = self.n_segments:
          storage_x1..x{M-1}, storage_theta1..theta{M}
          inflow_x1..x{M-1},  inflow_theta1..theta{M}
          season_x1..x{M-1},  season_theta1..theta{M}

        If set_context_from_row=True, also expects:
          - S_cap (or Adjusted_CAP_MG / GRanD_CAP_MG)
          - I_min, I_max
          - R_min (optional), R_max (optional)
        """
        def seg_params(prefix: str):
            xs = [float(row[f"{prefix}_x{i}"]) for i in range(1, self.n_segments)]
            thetas = [float(row[f"{prefix}_theta{i}"]) for i in range(1, self.n_segments + 1)]
            return xs + thetas

        def parse_segment_params(segment_params, M=self.n_segments):
            x_bounds   = [0.0] + list(segment_params[:M-1]) + [1.0]
            theta_vals = segment_params[M-1:]
            slopes     = [np.tan(theta) for theta in theta_vals]
            intercepts = [0.0]
            for k in range(1, M):
                dx = x_bounds[k] - x_bounds[k-1]
                intercepts.append(intercepts[k-1] + slopes[k-1] * dx)
            return x_bounds, slopes, intercepts

        s_params = seg_params("storage")
        i_params = seg_params("inflow")
        d_params = seg_params("season")

        (self.storage_bounds, self.storage_slopes, self.storage_intercepts) = parse_segment_params(s_params)
        (self.inflow_bounds,  self.inflow_slopes,  self.inflow_intercepts)  = parse_segment_params(i_params)
        (self.day_bounds,     self.day_slopes,     self.day_intercepts)     = parse_segment_params(d_params)

        if set_context_from_row:
            # capacity
            S_cap = row.get("S_cap", row.get("Adjusted_CAP_MG", row.get("GRanD_CAP_MG", None)))
            if S_cap is None:
                raise KeyError("assign_policy_params: missing S_cap/Adjusted_CAP_MG/GRanD_CAP_MG")
            # inflow bounds
            I_min = row.get("I_min"); I_max = row.get("I_max")
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

    # ---------- Core math ----------
    def _segment_eval(self, x, bounds, slopes, intercepts):
        # f(x) = m_i * (x - x_i) + b_i on the active segment
        for i in range(self.n_segments):
            if bounds[i] <= x < bounds[i+1]:
                return slopes[i] * (x - bounds[i]) + intercepts[i]
        # right-closed
        if x >= bounds[-1]:
            return slopes[-1] * (x - bounds[-2]) + intercepts[-1]
        raise ValueError(f"x={x} outside bounds {bounds}")

    def evaluate(self, X_norm):
        """X_norm = [S_norm, I_norm, D_norm] in [0,1]^3  -> z in [0,1]."""
        if len(X_norm) != self.n_inputs:
            raise AssertionError(f"Expected {self.n_inputs} inputs; got {len(X_norm)}.")
        if not all(0.0 <= x <= 1.0 for x in X_norm):
            raise AssertionError(f"Inputs must be in [0,1]. Got {X_norm}.")

        S, I, D = X_norm
        zS = self._segment_eval(S, self.storage_bounds, self.storage_slopes, self.storage_intercepts)
        zI = self._segment_eval(I, self.inflow_bounds,  self.inflow_slopes,  self.inflow_intercepts)
        zD = self._segment_eval(D, self.day_bounds,     self.day_slopes,     self.day_intercepts)

        # clamp each piece and average
        z = (max(0.0, min(1.0, zS)) +
             max(0.0, min(1.0, zI)) +
             max(0.0, min(1.0, zD))) / 3.0
        return max(0.0, min(1.0, z))

    def get_release(self, storage, inflow, day_of_year):
        """Normalize raw (S,I,D) -> evaluate -> scale -> enforce constraints."""
        S_t = float(storage)
        I_t = float(inflow)
        D_t = float(day_of_year)

        forced = self._storage_safety_override(S_t, I_t)
        if forced is not None:
            return self.enforce_constraints(forced, available=S_t + I_t)
        
        X_norm = self._normalize(S_t, I_t, D_t)   # << use AbstractPolicy normalizer
        z = self.evaluate(X_norm)                 # in [0,1]
        release = float(z) * float(self.release_max)
        return self.enforce_constraints(release, available=S_t + I_t)

    # ---------- optional: surface plot ----------
    def plot(self, N=41):
        """
        Creates a 3D plot with inflow (X), storage (Y), release (Z),
        and multiple surfaces for different weeks of the year.

        Args:
            fname (str): Filename to save the plot.
            save (bool): Flag to save the plot.
        """
        xs = np.linspace(0.0, 1.0, N)
        ys = np.linspace(0.0, 1.0, N)
        Z = np.zeros((N, N))
        # hold D at mid-season for a slice
        for i, s in enumerate(xs):
            for j, q in enumerate(ys):
                Z[i, j] = self.evaluate([s, q, 0.5])
        X, Y = np.meshgrid(xs, ys)
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z.T, alpha=0.7)
        ax.set_xlabel("S_norm"); ax.set_ylabel("I_norm"); ax.set_zlabel("z")
        ax.set_title("PWL policy surface (D_norm=0.5)")
        plt.tight_layout()
        plt.show()

    # --- add inside class PWL ---

    def plot_surfaces_for_different_weeks(self, fname=None, save=False, *, grid=30, weeks=None, n_weeks=5):
        """
        3D surfaces: X=inflow (normalized), Y=storage (normalized), Z=policy output,
        with multiple 'week' (D_norm) slices. Matches your legacy look.
        """

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
        ax.set_title('PWL: 3D Policy Output for Different Weeks')

        # legend squares
        custom = [plt.Line2D([0],[0], linestyle="none", marker='s', markersize=10,
                            markerfacecolor=cmap(i / max(1, len(weeks)-1)), alpha=0.6)
                  for i in range(len(weeks))]
        ax.legend(custom, [f'Week {w:.2f}' for w in weeks], loc='upper left', framealpha=0.9)

        if save:
            assert fname is not None, "Filename must be provided to save the plot."
            plt.savefig(fname, dpi=300)
        plt.show()

    def plot(self, N=41):
        # default to the legacy multi-week surfaces
        return self.plot_surfaces_for_different_weeks(grid=N)

        
        