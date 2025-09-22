"""
Radial Basis Function (RBF) reservoir release policy.

Overview
--------
`RBF` implements a smooth, flexible operating rule using a weighted sum of
Gaussian radial basis functions (RBFs). The policy maps **normalized inputs**
(storage, inflow, season/day-of-year) to a unitless control signal `z ∈ [0,1]`,
then scales to a physical release and enforces policy/physics constraints.

Inputs are normalized by the base class (`AbstractPolicy.set_context(...)`)
using per‐axis min/max ranges. The policy supports two ways of supplying
parameters:
  1) a **flat optimizer vector** (preferred for calibration/optimization), or
  2) a **row from a CSV table** (preferred for Pywr integration).

Units & Normalization
---------------------
- Storage S: **MG**
- Inflow  I: **MGD**
- Release R: **MGD**
- Day-of-year D: integer in [1, 366]

Normalization is handled by `AbstractPolicy` with
`x_min = (S_min, I_min, D_min)` and `x_max = (S_max, I_max, D_max)` provided via
`set_context(...)`. Typical context for DRB models:
  - `S_min=0`, `S_max=storage_capacity`
  - `I_min`, `I_max` from configuration bounds
  - `D_min=1`, `D_max=366`
The method `_normalize(S, I, D)` returns `(S_norm, I_norm, D_norm) ∈ [0,1]^3`.

Policy Formulation
------------------
Let there be `n` RBFs and `d` inputs (usually `d=3` for S, I, D).
For normalized input vector `x ∈ [0,1]^d`, the policy computes

    z(x) = Σ_{i=1..n} w_i · exp( - Σ_{j=1..d} ((x_j - c_{ij}) / r_{ij})^2 ),

where:
- `w_i`   are nonnegative weights (normalized to sum to 1 by the parser),
- `c_{ij}` are RBF centers (in normalized units),
- `r_{ij}` are RBF scales (strictly positive; clipped to ≥ 1e-6).

The final (unconstrained) release is `r* = z · release_max`.
The **constrained** release `r` is returned by
`AbstractPolicy.enforce_constraints(r*, available=S+I)`, which:
  1) applies `release_max` and `release_min`,
  2) caps by physically available water (S + I),
  3) records binding/violation flags for diagnostics.

Parameterization
----------------
Two equivalent parameter sources are supported:

1) **Optimizer vector** (flat list; recommended for search/optimization):
   - Count: `policy_n_params["RBF"]`
   - Packing order:
       `[ w_1 .. w_n, c(1,S), c(1,I), c(1,D), ..., c(n,S), c(n,I), c(n,D),
          r(1,S), r(1,I), r(1,D), ..., r(n,S), r(n,I), r(n,D) ]`
   - Bounds: `policy_param_bounds["RBF"]` (validated in `validate_policy_params()`)

   Use `parse_policy_params()` to populate internal arrays after setting
   `policy_params`.

2) **CSV row** (dict/Series; convenient for Pywr models):
   Expected per-basis columns (1-indexed):
     - `rbf{i}_center_storage`, `rbf{i}_center_inflow`, `rbf{i}_center_doy`
     - `rbf{i}_scale_storage`,  `rbf{i}_scale_inflow`,  `rbf{i}_scale_doy`
     - `rbf{i}_weight`
   Call `assign_policy_params(row, set_context_from_row=False/True)`.
   If `set_context_from_row=True`, the row must also provide
     - `S_cap` (or `Adjusted_CAP_MG` / `GRanD_CAP_MG`)
     - `I_min`, `I_max`
     - optional `R_min`, `R_max`
   which are forwarded to `set_context(...)` as:
     `x_min=(0.0, I_min, 1.0)`, `x_max=(S_cap, I_max, 366.0)`.

Key Behaviors & Guardrails
--------------------------
- **Weight normalization**: weights are normalized to sum to 1; if all zero,
  uniform weights are used.
- **Scale floor**: each `r_{ij}` is floored to 1e-6 to avoid division by zero.
- **Clamping**: `z` is clamped to `[0,1]` before scaling.
- **Constraints**: minimum/maximum release and water availability are enforced
  via `enforce_constraints(...)`; binding/violation tallies are accessible with
  `get_violation_summary()` from the base class.
- **Dimensions**: the number of bases `n` is `n_rbfs`; inputs `d` are
  `n_rbf_inputs` (typically 3).

API Summary
-----------
- `__init__(policy_params)`: optionally takes a flat vector and parses it.
- `validate_policy_params()`: checks vector length and bounds.
- `parse_policy_params()`: converts the flat vector into `(w, c, r)`.
- `assign_policy_params(row, set_context_from_row=False)`: CSV-row loader and
  optional context setter.
- `evaluate(X_norm) -> float`: returns `z ∈ [0,1]` for normalized inputs.
- `get_release(storage, inflow, day_of_year) -> float`: normalize → evaluate →
  scale → enforce; returns **MGD**.
- `plot(N=41)`: 3D surface slice of `z(S_norm, I_norm)` at `D_norm=0.5`.

Example (optimizer-vector path)
-------------------------------
>>> pol = RBF(policy_params=[
...   # weights (n=2)
...   0.6, 0.4,
...   # centers (2*3)
...   0.3, 0.2, 0.5,   0.8, 0.6, 0.5,
...   # scales  (2*3)
...   0.2, 0.3, 0.4,   0.25, 0.25, 0.35,
... ])
>>> pol.set_context(
...   release_min=10.0, release_max=1200.0,
...   storage_capacity=18000.0,
...   x_min=(0.0, 0.0, 1.0), x_max=(18000.0, 3000.0, 366.0),
... )
>>> r = pol.get_release(storage=9000.0, inflow=200.0, day_of_year=180)

References
----------
- 

Change Log
----------
- 2025-09-24 — Added comprehensive documentation: vector packing, CSV schema,
  normalization/constraints, guardrails, and example usage.
"""

import matplotlib.pyplot as plt
from typing import Sequence, Mapping, Any, Optional
import numpy as np

from pywrdrb.release_policies.abstract_policy import AbstractPolicy
from pywrdrb.release_policies.config import policy_n_params, policy_param_bounds, drbc_conservation_releases
from pywrdrb.release_policies.config import n_rbfs, n_rbf_inputs

class RBF(AbstractPolicy):
    """
    Radial Basis Function (RBF) policy class for reservoir operation.
    
    Uses Gaussian RBFs to determine scaled releases based on 
    input variables (x).
    
    One RBF is used for each input variable.
    
    Policy parameters are defined as:
    - c_ij: center of the ith RBF for the jth input (mean)
    - r_ij: width of the ith RBF for the jth input (standard deviation)
    - w_i: weight of the ith RBF (contribution to the output)
    
    All parameters are defined in the range [0, 1].
    
    The RBF function is defined as:
    z = sum(
        w_i * (
            sum(
                exp(-((x_j - c_ij) / r_ij)^2)
            ) for j in range(n_inputs)
        )
    ) for i in range(n_RBFs)
    
    
    Final release is computed as (subject to constraints):
    release = z * max_release
    
    More info on the formulation (for a different problem)
    can be found in Hadjimichael, Reed and Quinn (2020) 
    https://doi-org.proxy.library.cornell.edu/10.1155/2020/4170453
    """
    
    def __init__(self,
                 policy_params):
        
        super().__init__(policy_params=policy_params)
        self.nRBFs = int(n_rbfs)
        self.n_inputs = int(n_rbf_inputs)   # should be 3 for [S, I, D]
        self.n_params = policy_n_params["RBF"]
        self.param_bounds = policy_param_bounds["RBF"]

        #self.policy_id = policy_id

        # Parsed parameter arrays (filled by parse_policy_params)
        self.w = None                 # (nRBFs,)
        self.c = None                 # (nRBFs, n_inputs)
        self.r = None                 # (nRBFs, n_inputs)

        if policy_params is not None:
            self.parse_policy_params()

    # ---------- optimizer-path: flat vector ----------
    def validate_policy_params(self):
        """
        Validates the policy parameters
        """
        
        # Check if the number of parameters is correct
        assert len(self.policy_params) == self.n_params, \
            f"RBF policy expected {self.n_params} parameters, got {len(self.policy_params)}."
        
        # check parameter bounds
        for i, p in enumerate(self.policy_params):
            bounds = self.param_bounds[i]
            assert (p >= bounds[0]) and (p <= bounds[1]), \
                f"Parameter with index {i} is out of bounds {bounds}. Value: {p}."
        
        return
    
    def parse_policy_params(self):
        """
        Parses the policy parameters into RBF centers, widths, and weights.
        """
        
        # Validate the policy parameters
        self.validate_policy_params()
 
        ### Parse and assign
        # Given:
        # n RBF functions
        # d inputs (storage, inflow)
        # params = [ [w]*n, [c_ij]*n*d, [r_ij]*n*d ]
        
        w = self.policy_params[:self.nRBFs]
        
        start_idx = self.nRBFs
        end_idx = start_idx + (self.nRBFs)*self.n_inputs
        self.c = self.policy_params[start_idx:end_idx]
        
        start_idx = end_idx
        end_idx = start_idx + (self.nRBFs)*self.n_inputs
        self.r = self.policy_params[start_idx:end_idx]
        
        
        assert len(self.c) == self.nRBFs * self.n_inputs, \
            f"Expected {self.nRBFs * self.n_inputs} center parameters, got {len(self.c)}."
        assert len(self.r) == self.nRBFs * self.n_inputs, \
            f"Expected {self.nRBFs * self.n_inputs} radius parameters, got {len(self.r)}."
        assert len(w) == self.nRBFs, \
            f"Expected {self.nRBFs} weight parameters, got {len(w)}."
        
        
        # Make sure r > 0 to avoid division by zero
        for i in range(len(self.r)):
            self.r[i] = max(self.r[i], 1e-6)
        
        
        # Normalize the weights
        w_norm = []
        if np.sum(w) != 0:
            for w_i in w:
                w_norm.append(w_i / np.sum(w))
        else:
            w_norm = (1/self.nRBFs)*np.ones(len(w))
        self.w = w_norm
        
        return 

    # ---------- Pywr-path: CSV row params (dict/Series row) ----------
    def assign_policy_params(
        self,
        row: Mapping[str, Any],
        *,
        set_context_from_row: bool = False,
        max_rbfs: Optional[int] = None,
    ) -> None:
        """
        Define RBF centers, widths (scales), and weights from a pandas Series / dict-like row.

        Expected per-basis columns (1-indexed):
          - rbf{i}_center_storage, rbf{i}_center_inflow, rbf{i}_center_doy
          - rbf{i}_scale_storage,  rbf{i}_scale_inflow,  rbf{i}_scale_doy
          - rbf{i}_weight

        Optionally (if set_context_from_row=True), also expects:
          - S_cap (or Adjusted_CAP_MG / GRanD_CAP_MG)
          - I_min, I_max
          - R_min (optional), R_max (optional)
        """
        cap = int(max_rbfs if max_rbfs is not None else self.nRBFs)

        found = []
        i = 1
        while i <= 999:
            keys = [
                f"rbf{i}_center_storage", f"rbf{i}_center_inflow", f"rbf{i}_center_doy",
                f"rbf{i}_scale_storage",  f"rbf{i}_scale_inflow",  f"rbf{i}_scale_doy",
                f"rbf{i}_weight",
            ]
            if all(k in row for k in keys):
                found.append(i)
                if len(found) >= cap:
                    break
                i += 1
            else:
                break

        if len(found) == 0:
            raise KeyError("No RBF entries found (expected keys like 'rbf1_center_storage', etc.).")

        n, d = len(found), self.n_inputs
        centers = np.zeros((n, d), dtype=float)
        scales  = np.zeros((n, d), dtype=float)
        weights = np.zeros(n, dtype=float)

        for idx, i in enumerate(found):
            centers[idx, 0] = float(row[f"rbf{i}_center_storage"])
            centers[idx, 1] = float(row[f"rbf{i}_center_inflow"])
            centers[idx, 2] = float(row[f"rbf{i}_center_doy"])

            scales[idx, 0]  = float(row[f"rbf{i}_scale_storage"])
            scales[idx, 1]  = float(row[f"rbf{i}_scale_inflow"])
            scales[idx, 2]  = float(row[f"rbf{i}_scale_doy"])

            weights[idx]    = float(row[f"rbf{i}_weight"])

        scales = np.maximum(scales, 1e-6)

        w_sum = float(np.sum(weights))
        weights = (weights / w_sum) if w_sum > 0.0 else np.full(n, 1.0 / n, dtype=float)

        self.w, self.c, self.r = weights, centers, scales
        self.nRBFs = n  # reflect actual number loaded

        if set_context_from_row:
            S_cap = row.get("S_cap", row.get("Adjusted_CAP_MG", row.get("GRanD_CAP_MG", None)))
            if S_cap is None:
                raise KeyError("assign_policy_params: missing S_cap/Adjusted_CAP_MG/GRanD_CAP_MG")

            I_min = row.get("I_min")
            I_max = row.get("I_max")
            if I_min is None or I_max is None:
                raise KeyError("assign_policy_params: missing I_min / I_max")

            R_min = float(row.get("R_min", 0.0))
            R_max = float(row.get("R_max", 1e12))

            self.set_context(
                release_min=float(R_min),
                release_max=float(R_max),
                storage_capacity=float(S_cap),
                x_min=(0.0, float(I_min), 1.0),
                x_max=(float(S_cap), float(I_max), 366.0),
            )

    def evaluate(self, X_norm):
        """
        Evaluate the policy function.

        Args:
            X (list): A list of input values, including normalized:
                - Storage (S)
                - Inflow (I)
                - Day of year (D)

        Returns:
            float: The computed release.
        """
        
        # Make sure we got the right number of inputs
        assert len(X_norm) == self.n_inputs, \
            f"Expected {self.n_inputs} input variables; got {len(X_norm)}."

        # make sure X values are in [0, 1]
        assert all(0 <= x <= 1 for x in X_norm), \
            f"Input values must be in the range [0, 1]. Values: {X_norm}."

        # Calculate
        z = 0.0
        for i in range(self.nRBFs):
            sq_term = 0.0
            for j in range(self.n_inputs):
                idx = i * self.n_inputs + j
                sq_term += ((X_norm[j] - self.c[idx]) / self.r[idx]) ** 2
            z += self.w[i] * np.exp(-sq_term)
        
        # Impose bound limits
        z = max(0.0, min(1.0, z)) 
                       
        return z
    
    def get_release(self, storage: float, inflow: float, day_of_year: float) -> float:
        """Normalize via base class, scale, then enforce constraints with availability."""
        S_t = float(storage)
        I_t = float(inflow)
        D_t = float(day_of_year)

        forced = self._storage_safety_override(S_t, I_t)
        if forced is not None:
            return self.enforce_constraints(forced, available=S_t + I_t)
        
        X_norm = self._normalize(S_t, I_t, D_t)
        z = self.evaluate(X_norm)
        release = z * float(self.release_max)
        return self.enforce_constraints(release, available=S_t + I_t)

    # ---------- optional: surface plot ----------

    def plot_surfaces_for_different_weeks(self, fname=None, save=False, *, grid=30, weeks=None, n_weeks=5):
        """
        3D surfaces: X=inflow (normalized), Y=storage (normalized), Z=policy output,
        with multiple 'week' (D_norm) slices. Matches your legacy look.
        """
        import numpy as np
        import matplotlib.pyplot as plt

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
        ax.set_title('RBF: 3D Policy Output for Different Weeks')

        custom = [plt.Line2D([0],[0], linestyle="none", marker='s', markersize=10,
                            markerfacecolor=cmap(i / max(1, len(weeks)-1)), alpha=0.6)
                for i in range(len(weeks))]
        ax.legend(custom, [f'Week {w:.2f}' for w in weeks], loc='upper left', framealpha=0.9)

        if save:
            assert fname is not None, "Filename must be provided to save the plot."
            plt.savefig(fname, dpi=300)
        plt.show()

    def plot(self, N=41):
        return self.plot_surfaces_for_different_weeks(grid=N)

