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

        # context placeholders (filled by set_context)
        self.release_min = None
        self.release_max = None
        self.storage_capacity = None
        self.x_min = None      # array-like length = n_inputs
        self.x_max = None      # array-like length = n_inputs

        # number of inputs (policies may overwrite)
        self.n_inputs = 3  # default [S, I, D]

    # ---------- context ----------
    def set_context(self, *, release_min, release_max, storage_capacity, x_min, x_max):
        """Host (Reservoir/Pywr) provides operating envelope & scaling."""
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

    # optional tiny helper for [S,I,D]
    def _normalize(self, S, I, D):
        return self._normalize_vector([S, I, D])

    # ---------- constraints ----------
    def enforce_constraints(self, release, available):
        """
        Clamp by user min/max, then by mass-balance availability (S + I),
        and ensure non-negative (or at least release_min if provided).
        """
        r = float(release)
        if self.release_min is not None:
            r = max(self.release_min, r)
        if self.release_max is not None:
            r = min(self.release_max, r)
        r = min(r, float(available))
        r = max(r, self.release_min if self.release_min is not None else 0.0)
        return r

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
