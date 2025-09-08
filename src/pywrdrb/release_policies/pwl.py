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

        X_norm = self._normalize(S_t, I_t, D_t)   # << use AbstractPolicy normalizer
        z = self.evaluate(X_norm)                 # in [0,1]
        release = float(z) * float(self.release_max)
        return self.enforce_constraints(release, available=S_t + I_t)

    # ---------- optional: quick surface plot ----------
    def plot(self, N=41):
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

        
        