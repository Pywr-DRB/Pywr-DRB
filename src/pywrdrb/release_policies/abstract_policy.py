"""
Abstract base class for parameterized reservoir release policies.

Overview
--------
`AbstractPolicy` defines the minimal interface and shared utilities for
storage–inflow–season based operating rules used by the Reservoir simulator.
Concrete policies (e.g., STARFIT, RBF, PWL) inherit from this class and provide
their own parameter validation, parsing, and `get_release(...)` logic.

This base class standardizes:
- **Context ingestion** via `set_context(...)` (release bounds, capacity,
  and input scaling ranges).
- **Input normalization** of (S, I, D) → [0,1]^n with `_normalize_vector(...)`.
- **Constraint enforcement** with `enforce_constraints(...)`, applying policy
  bounds and physical availability in a predictable order.
- **Diagnostics**: lightweight violation tracking and a printable context.

Responsibilities & Contracts
----------------------------
Concrete subclasses **must** implement:
- `validate_policy_params(self)`: assert/raise if `self.policy_params`
  violates shape/value rules for the policy.
- `parse_policy_params(self)`: transform `self.policy_params` into any
  internal structures (e.g., breakpoints, weights) needed at runtime.
- `get_release(self, storage, inflow, day_of_year) -> float`:
  compute a raw target release, then **always call**
  `self.enforce_constraints(target, available=S+I)` before returning (MGD).
- `plot(self)`: optional visualization interface (signature standardized here).

Common optional pattern (not required by this base class):
- `evaluate(self, Xn) -> float`: map **normalized** inputs in [0,1]^n to a
  non-dimensional release index `z ∈ [0,1]`. If present, `get_release(...)`
  typically sets `release = z * release_max` and then calls
  `enforce_constraints(...)`.

Context & Normalization
-----------------------
Call `set_context(...)` **once** before using the policy:
- `release_min` (float, MGD): conservation minimum
- `release_max` (float, MGD): policy/ops maximum
- `storage_capacity` (float, MG): reservoir capacity
- `x_min`, `x_max` (array-like, len = n_inputs): per-input scaling bounds

Assumptions and checks:
- Default `n_inputs = 3` → **[storage (MG), inflow (MGD), day-of-year (1–366)]**.
  Subclasses may override `n_inputs` **before** calling `set_context(...)`.
- `x_min`/`x_max` must match length `n_inputs` and satisfy `x_max > x_min` element-wise.
- For the storage dimension, **invariant**: `x_max[0] == storage_capacity` (enforced with
  a tight tolerance).
- Convenience copies `I_min = x_min[1]`, `I_max = x_max[1]` are exposed for downstream use.

Normalization:
- `_normalize_vector(X)` maps raw inputs `X` to `[0,1]^n` using min–max scaling and
  clamps each component into [0,1].
- `_normalize(S, I, D)` is a convenience wrapper for the canonical 3-input case.

Constraint Enforcement
----------------------
`enforce_constraints(release, available, track=True)` enforces limits **in this order**:
1. **Policy bounds**: clamp to `[release_min, release_max]` (if provided).
2. **Physics**: clamp to `available = S + I` (cannot exceed mass-balance availability).

The method also logs which bounds were binding or violated:
- `"min_binding"`: raised to `release_min`
- `"max_binding"`: capped by `release_max`
- `"availability_binding"`: capped by availability
- `"min_violation"`: **could not** meet `release_min` because `available < release_min`

Diagnostics & Utilities
-----------------------
- `debug` flag (bool): subclasses and calling code can set this to enable step prints.
- `reset_violation_log()` clears per-timestep flags; call once per simulation run.
- `get_violation_summary()` returns counts per flag for quick reporting.
- `get_context()` returns a dict with the active context (useful for audits).
- A one-line `[CTX] ...` summary is printed from `set_context(...)` **once** per instance.

Units (DRB convention)
----------------------
- Storage **S**: MG
- Flow / Release **I, R**: MGD
- Day-of-year **D**: integer in `[1, 366]`

Subclassing Guide (Typical Pattern)
-----------------------------------
```python
class MyPolicy(AbstractPolicy):
    def validate_policy_params(self):
        p = np.asarray(self.policy_params, float)
        if p.shape != (K,):  # your K
            raise ValueError("Expected K parameters.")
        # additional value/range checks…

    def parse_policy_params(self):
        p = np.asarray(self.policy_params, float)
        # precompute structures (e.g., centers, slopes)
        self._parsed = {...}

    def get_release(self, storage, inflow, day_of_year):
        S = float(storage); I = float(inflow); D = float(day_of_year)
        Xn = self._normalize(S, I, D)          # [0,1]^n
        # If you expose evaluate(Xn), produce z in [0,1]
        z = self.evaluate(Xn)                   # or inline your mapping
        r_target = float(z) * float(self.release_max)
        return self.enforce_constraints(r_target, available=S + I)

    def evaluate(self, Xn):
        # map normalized inputs Xn to z in [0,1]
        ...

    def plot(self):
        # optional visualization
        ...
```

"""

from abc import ABC, abstractmethod
import numpy as np

class AbstractPolicy(ABC):
    """
    Minimal base for parameterized reservoir policies.

    Shared stuff:
      - set_context(...) to pass limits & scaling ranges
      - _normalize_vector(X) using your simple loop
      - enforce_constraints(release, available)
      - validate_policy_params / parse_policy_params contracts
    Concrete policies keep get_release(...) and evaluate(...).
    """

    def __init__(self, policy_params):
        self.policy_params = policy_params

        # simple debug toggle used by Reservoir.run() print guard
        self.debug = False  # set True from outside when needed

        self._violations = {
            "min_binding": [],       # r was raised to release_min
            "max_binding": [],       # r was capped by release_max
            "availability_binding": [],  # r was capped by available
            "min_violation": [],     # could not meet release_min because avail < release_min
        }

        # context placeholders (filled by set_context)
        self.release_min = None
        self.release_max = None
        self.storage_capacity = None
        self.x_min = None      # array-like length = n_inputs
        self.x_max = None      # array-like length = n_inputs
        self.low_storage_threshold = None

        # convenience copies of inflow bounds
        self.I_min = None
        self.I_max = None

        # one-time context print flag
        self._ctx_printed = False   

        # number of inputs (policies may overwrite)
        self.n_inputs = 3  # default [S, I, D]

    # ---------- context ----------
    def set_context(self, *, release_min, release_max, storage_capacity, x_min, x_max, low_storage_threshold=None):
        """Host provides operating envelope & scaling."""
        self.release_min = float(release_min)
        self.release_max = float(release_max)
        self.storage_capacity = float(storage_capacity)

        self.x_min = np.asarray(x_min, dtype=float)
        self.x_max = np.asarray(x_max, dtype=float)

        # allow subclasses to choose n_inputs; otherwise infer
        if getattr(self, "n_inputs", None) is None:
            self.n_inputs = int(len(self.x_min))
        else:
            self.n_inputs = int(self.n_inputs)

        # sanity checks
        if self.x_min.shape != self.x_max.shape:
            raise ValueError("x_min and x_max must have same shape.")
        if len(self.x_min) != self.n_inputs:
            raise ValueError(f"x_min/x_max length ({len(self.x_min)}) must equal n_inputs ({self.n_inputs}).")
        if not np.all(self.x_max > self.x_min):
            raise ValueError("x_max must be > x_min element-wise.")

        if low_storage_threshold is None:
            self.low_storage_threshold = 0.05 * self.storage_capacity
        else:
            self.low_storage_threshold = float(low_storage_threshold)

        # cache inflow bounds; assert storage dim matches cap
        self.I_min = float(self.x_min[1])
        self.I_max = float(self.x_max[1])
        # Optional invariant: x_max[0] should equal storage_capacity
        if not np.isclose(self.x_max[0], self.storage_capacity, rtol=0, atol=1e-6):
            raise ValueError(
                f"x_max[0] ({self.x_max[0]}) must equal storage_capacity ({self.storage_capacity})."
            )

        # one-line context log (printed once per policy instance)
        if not self._ctx_printed:
            name = getattr(self, "reservoir_name", getattr(self, "name", "unknown"))
            print(
                f"[CTX] {name}: S_cap={self.storage_capacity:.2f}, "
                f"I∈[{self.I_min:.2f}, {self.I_max:.2f}], "
                f"R∈[{self.release_min:.2f}, {self.release_max:.2f}]"
            )
            self._ctx_printed = True

    def reset_violation_log(self):
        for k in self._violations:
            self._violations[k].clear()

    def get_violation_summary(self):
        # count TRUE flags per key
        return {k: int(np.sum(v)) for k, v in self._violations.items()}
    
    # convenience for debug or tests
    def get_context(self):
        return {
            "release_min": self.release_min,
            "release_max": self.release_max,
            "storage_capacity": self.storage_capacity,
            "x_min": tuple(np.asarray(self.x_min, float)),
            "x_max": tuple(np.asarray(self.x_max, float)),
            "I_min": self.I_min,
            "I_max": self.I_max,
            "low_storage_threshold": self.low_storage_threshold,
        }
    
    # ---------- normalization ----------
    def _normalize_vector(self, X):
        """
        Normalize a raw vector X (len = n_inputs) to [0,1]^n using min-max.
        Uses your exact loop logic.
        """
        if self.x_min is None or self.x_max is None:
            raise RuntimeError("set_context(...) must be called before normalization.")

        X = np.asarray(X, dtype=float)
        if len(X) != self.n_inputs:
            raise ValueError(f"Expected X of length {self.n_inputs}, got {len(X)}.")

        X_norm = np.zeros(self.n_inputs, dtype=float)
        for i in range(self.n_inputs):
            denom = (self.x_max[i] - self.x_min[i]) or 1.0
            X_norm[i] = (X[i] - self.x_min[i]) / denom
            X_norm[i] = max(0.0, min(1.0, X_norm[i]))  # clamp to [0,1]
        return X_norm

    def _normalize(self, S, I, D):
        return self._normalize_vector([S, I, D])

    # ---------- constraints ----------
    def enforce_constraints(self, release, available, *, track=True, eps=1e-9):
        avail = max(0.0, float(available))
        r_raw = float(release)

        # Policy bounds first (intent)
        r_bound = r_raw
        max_binding = False
        if self.release_max is not None and r_bound > float(self.release_max):
            r_bound = float(self.release_max)
            max_binding = True

        min_binding = False
        if self.release_min is not None and r_bound < float(self.release_min):
            r_bound = float(self.release_min)
            min_binding = True

        # Physics last (cannot be undone)
        availability_binding = r_bound > avail + eps
        r_final = min(r_bound, avail)

        # Could we meet the conservation minimum?
        min_violation = False
        if self.release_min is not None:
            min_violation = (avail + eps) < float(self.release_min)

        if track:
            self._violations["min_binding"].append(min_binding)
            self._violations["max_binding"].append(max_binding)
            self._violations["availability_binding"].append(availability_binding)
            self._violations["min_violation"].append(min_violation)

        # Optional hard guard:
        assert r_final <= avail + 1e-9, "Release exceeds physically available water."

        return r_final
    
    def _storage_safety_override(self, storage, inflow, *, eps=1e-6):
        """
        Implements simple guardrails suggested by Trevor / Sai Veena:
          - If storage >= capacity (≈ normalized S >= 1), force max release.
          - If storage <= low_storage_threshold, force min release.
        Returns:
          float | None  -> the forced release if guard triggered, else None.
        """
        S = float(storage)
        if self.storage_capacity is None:
            return None  # context not set; skip

        # Full (or numerically full): dump using max release
        if S >= self.storage_capacity - eps:
            return float(self.release_max) if self.release_max is not None else None

        # Very low storage: preserve water with min release
        if self.low_storage_threshold is not None and S <= self.low_storage_threshold + eps:
            return float(self.release_min) if self.release_min is not None else None

        return None
    # ---------- policy contracts ----------
    @abstractmethod
    def validate_policy_params(self):
        pass

    @abstractmethod
    def parse_policy_params(self):
        pass

    @abstractmethod
    def get_release(self, storage, inflow, day_of_year):
        """
        Concrete policies implement this end-to-end. Typical pattern:

            S = float(storage); I = float(inflow); D = float(day_of_year)
            Xn = self._normalize(S, I, D)      # or self._normalize_vector([S,I,D])
            z  = self.evaluate(Xn)             # z in [0,1]
            release = float(z) * self.release_max
            return self.enforce_constraints(release, available=S+I)
        """
        pass

    @abstractmethod
    def plot(self):
        pass
