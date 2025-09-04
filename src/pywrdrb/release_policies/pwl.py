import numpy as np
import matplotlib.pyplot as plt

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
                 release_max,
                 release_min,
                 storage_capacity,
                 n_rbfs,
                 n_pwl_inputs,
                 policy_n_params,
                 policy_param_bounds,
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
        self.n_segments = n_segments
        self.n_inputs = n_pwl_inputs
        self.param_bounds = policy_param_bounds["PWL"]
        self.n_params = policy_n_params["PWL"]
        
        # X (input) max and min values
        # used to normalize the input data
        # X = [storage, inflow, day_of_year]
        # self.x_min = np.array([0.0, 
        #                        self.Reservoir.inflow_min,
        #                        1.0])
        
        # self.x_max = np.array([self.Reservoir.capacity, 
        #                        self.Reservoir.inflow_max,
        #                        366.0])
        
        self.policy_params = policy_params
        self.parse_policy_params()

        
    def validate_policy_params(self):
        """
        Validates the policy parameters.
        """
        # Check if the number of parameters is correct
        assert len(self.policy_params) == self.n_params, \
            f"PiecewiseLinear policy expected {self.n_params} parameters, got {len(self.policy_params)}."
        
        # check parameter bounds
        for i, p in enumerate(self.policy_params):
            bounds = self.param_bounds[i]
            assert (p >= bounds[0]) and (p <= bounds[1]), \
                f"Parameter with index {i} is out of bounds {bounds}. Value: {p}."
            
        return
        
    def parse_policy_params(self):
        """
        Parses policy parameters into segment boundaries and slopes.
        """

        # Validate the policy parameters
        self.validate_policy_params()

        def parse_segment_params(segment_params, M = self.n_segments):
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
            
            x_bounds = [0.0] + list(segment_params[:M - 1]) + [1.0]
            theta_vals = segment_params[M - 1:]
            slopes = [np.tan(theta) for theta in theta_vals]

            intercepts = [0.0]
            for i in range(1, M):
                dx = x_bounds[i] - x_bounds[i - 1]
                b = intercepts[i - 1] + slopes[i - 1] * dx
                intercepts.append(b)
            return x_bounds, slopes, intercepts


        ### Params contains [storage_params, inflow_params, day_of_year_params]
        # split params in thirds

        n_param_subset = len(self.policy_params) // 3
        
        s_params = self.policy_params[:n_param_subset]
        i_params = self.policy_params[n_param_subset:(2 * n_param_subset)]
        d_params = self.policy_params[(2 * n_param_subset):]

        # Calculate and store the segment boundaries, slopes, and intercepts
        # for storage, inflow, and day of year functions
        (self.storage_bounds,
        self.storage_slopes,
        self.storage_intercepts) = parse_segment_params(s_params)

        (self.inflow_bounds,
        self.inflow_slopes,
        self.inflow_intercepts) = parse_segment_params(i_params)

        (self.day_bounds,
        self.day_slopes,
        self.day_intercepts) = parse_segment_params(d_params)

    def assign_policy_params(self, pwl_params):
        """
        Load PWL parameters for (reservoir_name, policy_id) and build the three
        1-D piecewise linear mappings (storage, inflow, season).

        Expects `pwl_params` indexed by ['reservoir','policy_id'] with columns:
        storage_x1, storage_x2, storage_theta1..3,
        inflow_x1,  inflow_x2,  inflow_theta1..3,
        season_x1,  season_x2,  season_theta1..3,
        GRanD_CAP_MG, GRanD_MEANFLOW_MGD,
        Adjusted_CAP_MG, Adjusted_MEANFLOW_MGD,
        Max_release, Release_max, Release_min

        Notes:
        - Thetas are radians; slopes = tan(theta).
        - DRBC overrides take precedence for R_min / R_max.
        - If `self.M` is unset, defaults to 3.
        """
        # Ensure expected index
        if not isinstance(pwl_params.index, pd.MultiIndex) or \
        set(pwl_params.index.names) != {"reservoir", "policy_id"}:
            pwl_params = pwl_params.set_index(["reservoir", "policy_id"])

        # Resolve key (with fallback to policy_id='default')
        key = (self.reservoir_name, self.policy_id)
        if key not in pwl_params.index:
            fallback_key = (self.reservoir_name, "default")
            if fallback_key in pwl_params.index:
                if hasattr(self.model, "logger"):
                    self.model.logger.warning(
                        f"[PWL] policy_id='{self.policy_id}' not found for {self.reservoir_name}; "
                        f"falling back to 'default'."
                    )
                key = fallback_key
            else:
                raise KeyError(
                    f"PWL parameters not found for reservoir='{self.reservoir_name}' "
                    f"with policy_id='{self.policy_id}' or 'default'."
                )

        row = pwl_params.loc[key]

        if hasattr(self.model, "logger"):
            self.model.logger.info(f"[PWL] Loaded parameters for {self.reservoir_name}, policy_id={key[1]}")

        # Number of segments
        if getattr(self, "M", None) is None:
            self.M = 3  # matches your config

        # ---- 15 PWL params in Storage → Inflow → Season order ----
        required_cols = [
            "storage_x1","storage_x2","storage_theta1","storage_theta2","storage_theta3",
            "inflow_x1","inflow_x2","inflow_theta1","inflow_theta2","inflow_theta3",
            "season_x1","season_x2","season_theta1","season_theta2","season_theta3",
        ]
        missing = [c for c in required_cols if c not in row.index]
        if missing:
            raise KeyError(f"Missing PWL columns in CSV: {missing}")

        s_params = [
            float(row["storage_x1"]),
            float(row["storage_x2"]),
            float(row["storage_theta1"]),
            float(row["storage_theta2"]),
            float(row["storage_theta3"]),
        ]
        i_params = [
            float(row["inflow_x1"]),
            float(row["inflow_x2"]),
            float(row["inflow_theta1"]),
            float(row["inflow_theta2"]),
            float(row["inflow_theta3"]),
        ]
        d_params = [
            float(row["season_x1"]),
            float(row["season_x2"]),
            float(row["season_theta1"]),
            float(row["season_theta2"]),
            float(row["season_theta3"]),
        ]

        # Build PWLs (same logic as standalone)
        (self.s_bounds, self.s_slopes, self.s_intercepts) = self._parse_segment_params(s_params, M=self.M)
        (self.i_bounds, self.i_slopes, self.i_intercepts) = self._parse_segment_params(i_params, M=self.M)
        (self.d_bounds, self.d_slopes, self.d_intercepts) = self._parse_segment_params(d_params, M=self.M)

        # Optional: quick hardening identical to standalone expectations
        self._validate_axis(self.s_bounds, self.s_slopes)
        self._validate_axis(self.i_bounds, self.i_slopes)
        self._validate_axis(self.d_bounds, self.d_slopes)

        # ---- Capacity & mean inflow (normalization / factor scaling) ----
        self.S_cap = float(row["Adjusted_CAP_MG"]) if pd.notnull(row.get("Adjusted_CAP_MG", np.nan)) \
                    else float(row["GRanD_CAP_MG"])
        self.I_bar = float(row["Adjusted_MEANFLOW_MGD"]) if pd.notnull(row.get("Adjusted_MEANFLOW_MGD", np.nan)) \
                    else float(row["GRanD_MEANFLOW_MGD"])

        # ---- Release limits (DRBC overrides take precedence) ----
        # R_min
        if self.reservoir_name in drbc_conservation_releases:
            self.R_min = float(drbc_conservation_releases[self.reservoir_name])
        else:
            self.R_min = float((row["Release_min"] + 1.0) * self.I_bar) \
                        if pd.notnull(row.get("Release_min", np.nan)) else 0.0

        # R_max
        if self.reservoir_name in max_discharges:
            self.R_max = float(max_discharges[self.reservoir_name])
        else:
            if pd.notnull(row.get("Max_release", np.nan)) and float(row["Max_release"]) > 0.0:
                # absolute cap provided
                self.R_max = float(row["Max_release"])
            elif pd.notnull(row.get("Release_max", np.nan)):
                # factor × mean-flow style cap
                self.R_max = float((row["Release_max"] + 1.0) * self.I_bar)
            else:
                # fallback
                self.R_max = float(getattr(self.node, "max_flow", 1e12))

        # Optional flag to remove R_max entirely
        if getattr(self, "remove_R_max", False):
            self.R_max = 999999.0

        #TODO: require this to be loaded from the shared config file
        # Require exact inflow bounds from CSV
        for col in ("I_min", "I_max"):
            if col not in row.index or pd.isnull(row[col]):
                raise KeyError("pwl.csv must include columns I_min and I_max for exact normalization.")

        self.I_min = float(row["I_min"])
        self.I_max = float(row["I_max"])
        if not (self.I_max > self.I_min):
            raise ValueError(f"I_max ({self.I_max}) must be > I_min ({self.I_min}).")

        # EXACT same min–max ranges as the standalone
        self.x_min = np.array([0.0, self.I_min, 1.0], dtype=float)
        self.x_max = np.array([float(self.S_cap), float(self.I_max), 366.0], dtype=float)


    def evaluate(self, X):
        """
        Evaluate PWL on normalized [S_norm, I_norm, D_norm] -> z in [0,1].
        
        Args:
            X (list): A list of input values, including normalized:
                - Storage (S)
                - Inflow (I)
                - Day of year (D)
        
        Returns:
            float: The computed release.
        """
        # Separate inputs [storage, inflow, day_of_year]
        S, I, D = X
        
        assert I is not None, "Inflow input required but not provided."
        assert S is not None, "Storage input required but not provided."
        assert D is not None, "Day of year input required but not provided."

        
        def segment_eval(x, bounds, slopes, intercepts):
            """
            Resolves the piecewise linear function for a given x.
            
            Uses function:
            f(x) = m_i * (x - x_i) + b_i
            
            where:
                - m_i is the slope of the segment
                - x_i is the lower bound of the segment
                - b_i is the intercept of the segment
            
            
            Args:
                x (float): The input value.
                bounds (list): The segment boundaries.
                slopes (list): The slopes of the segments.
                intercepts (list): The intercepts of the segments.
            
            Returns:
                float: The evaluated value at x.
            """
            for i in range(self.n_segments):
                if bounds[i] <= x < bounds[i + 1]:
                    dx = x - bounds[i]
                    return slopes[i] * dx + intercepts[i]
            if x >= bounds[-1]:
                dx = x - bounds[-2]
                return slopes[-1] * dx + intercepts[-1]
            raise ValueError(f"Value {x} outside bounds: {bounds}")

        zS = segment_eval(S, self.storage_bounds, self.storage_slopes, self.storage_intercepts)
        zS = max(0.0, min(1.0, zS)) 
        
        zI = segment_eval(I, self.inflow_bounds, self.inflow_slopes, self.inflow_intercepts)
        zI = max(0.0, min(1.0, zI))

        zD = segment_eval(D, self.day_bounds, self.day_slopes, self.day_intercepts)
        zD = max(0.0, min(1.0, zD))

        # Compute the final release value
        z = (zS + zI + zD) / 3.0

        # Impose bound limits
        z = max(0.0, min(1.0, z)) 
        return z


    def get_release(self, 
                    inflow, 
                    storage,
                    day_of_year):
        """
        Computes the reservoir release for a given timestep based on the 
        current storage level.

        Args:
            inflow (float): Current inflow (MGD).
            storage (float): Current storage (MG).
            day_of_year (float): Current day of the year (1-366).

        Returns:
            float: The computed release.
        """
        
       # Get state variables
        I_t = float(inflow)
        S_t = float(storage)
        day_of_year = float(day_of_year)

        # inputs  = [storage, inflow, day_of_year]
        X = np.array([S_t, I_t, day_of_year])

        # Normalize X
        #TODO: Check the normalization logic
        X_norm = np.zeros(self.n_inputs)
        for i in range(self.n_inputs):
            X_norm[i] = (X[i] - self.x_min[i]) / (self.x_max[i] - self.x_min[i])        
            X_norm[i] = max(0.0, min(1.0, X_norm[i])) # enforce bounds [0, 1]
        
        # Compute release
        release  = self.evaluate(X_norm) * self.Reservoir.release_max

        # Enforce constraints (defined in AbstractPolicy)
        release = self.enforce_constraints(release)
        release = min(release, S_t + I_t)
        release = max(release, self.Reservoir.release_min)
        
        return release

    def plot(self, 
             fname=None,
             save=False):
        """
        Plot the piecewise linear policy function.

        Args:
            fname (str): Filename for saving the plot.
            save (bool): Whether to save the plot as a file.
        """
        self.plot_surfaces_for_different_weeks(fname=fname, save=save)
        # self.plot_storage_policy(fname=fname, save=save)
        
        