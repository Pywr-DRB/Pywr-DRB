import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Mapping, Any, Optional

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

    # ---------- Convenience: MultiIndex DataFrame (reservoir_name, policy_id) ----------
    def assign_from_df(
        self,
        df,  # pandas DataFrame
        *,
        reservoir_name: Optional[str] = None,
        policy_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Load parameters from a MultiIndex DataFrame with index (reservoir_name, policy_id).
        Falls back to (reservoir_name, 'default') if the exact row is missing.
        """
        if df.index.nlevels < 2:
            raise ValueError("assign_from_df expects a MultiIndex with (reservoir_name, policy_id).")

        res = reservoir_name or self.reservoir_name
        pid = policy_id or self.policy_id or "default"
        if res is None:
            raise ValueError("reservoir_name not provided and self.reservoir_name is None.")

        key = (res, pid)
        if key not in df.index:
            fallback = (res, "default")
            if fallback in df.index:
                print(f"[RBF] policy_id '{pid}' not found for '{res}'. Falling back to 'default'.")
                key = fallback
            else:
                raise KeyError(f"Parameters not found for {res}/{pid} (and no 'default').")

        row = df.loc[key]
        self.assign_policy_params(row, **kwargs)

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
            f"Expected {self.n_inputs} input variables; got {len(X)}."

        # make sure X values are in [0, 1]
        assert all(0 <= x <= 1 for x in X_norm), \
            f"Input values must be in the range [0, 1]. Values: {X}."
        
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

        X_norm = self._normalize(S_t, I_t, D_t)
        z = self.evaluate(X_norm)
        release = z * float(self.release_max)
        return self.enforce_constraints(release, available=S_t + I_t)

    # ---------- quick 3D slice plot ----------
    def plot(self, N=41):
        xs = np.linspace(0.0, 1.0, N)
        ys = np.linspace(0.0, 1.0, N)
        Z  = np.zeros((N, N))
        for i, s in enumerate(xs):
            for j, q in enumerate(ys):
                Z[i, j] = self.evaluate([s, q, 0.5])  # hold day mid-season

        X, Y = np.meshgrid(xs, ys)
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z.T, alpha=0.7)
        ax.set_xlabel("S_norm"); ax.set_ylabel("I_norm"); ax.set_zlabel("z")
        ax.set_title("RBF policy surface (D_norm=0.5)")
        plt.tight_layout()
        plt.show()