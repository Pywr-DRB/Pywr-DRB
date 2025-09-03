"""
Piecewise Linear custom Pywr parameters for reservoir release.

Overview
--------
Custom Pywr parameter that computes reservoir releases based on a surrogate operating policy.

- Breaks the state space into linear segments, specifying reservoir release as a function of storage or inflow using connected line segments.
- Designed for the Delaware River Basin (DRB) Pywr models.
- Supports scenario-based sensitivity analysis using parameters from CSV or HDF5.
- Enforces policy constraints (e.g., R_min, R_max) and physical bounds.

Inputs:
- Normalized storage (S/S_cap)
- Normalized inflow 
- Seasonal indicator (STARFIT: day of year, RBF: normalized week, PWL: storage/inflow breakpoints)


Key Steps
---------
1. **Parameter Setup**: Initializes the parameter with scenario-aware storage for releases.
2. **Policy Parameter Assignment**: Loads and assigns piecewise linear policy parameters.
3. **Policy Evaluation**: Evaluates the release based on current storage, inflow,
    and policy parameters, ensuring compliance with maximum and minimum release limits.
4. **Value Calculation**: Computes the release value for a given timestep and scenario,
    considering inflow and storage conditions.
5. **Parameter Loading**: Supports loading parameters from sensitivity analysis scenarios
    or default values, ensuring flexibility in policy evaluation.

Technical Notes
---------------
This parameter is designed to be used in conjunction with the Pywr framework, specifically
for the Delaware River Basin (DRB) models. It integrates with the STARFIT framework
for evaluating reservoir releases based on piecewise linear policies. The parameter
supports scenario-based sensitivity analysis, allowing for dynamic policy evaluation
based on different operational conditions.


Links
-----


Change Log
----------
Marilyn Smith, [Date], Initial implementation of PWLReservoirRelease parameter.
"""
import numpy as np
import pandas as pd
from pywr.parameters import Parameter, load_parameter

from pywrdrb.parameters.lower_basin_ffmp import conservation_releases, max_discharges
from pywrdrb.path_manager import get_pn_object

pn = get_pn_object()

class PWLReservoirRelease(Parameter):
    """
    Piecewise Linear reservoir release parameter for DRB Pywr models.

    Implements scenario-aware parameter loading and piecewise linear policy 
    evaluation for release decisions, consistent with STARFIT integration.

     - Inputs: storage, inflow, day-of-year
      - Min–max normalization to [0,1]
      - Three 1-D PWL transforms (angles -> slopes via tan)
      - z = (zS + zI + zD)/3 in [0,1]
      - release = z * R_max, then clamp by R_min and mass-balance
    """
    _default_params_cache = None

    def __init__(
        self,
        model,
        reservoir_name,
        storage_node,
        flow_parameter,
        run_sensitivity_analysis,
        sensitivity_analysis_scenarios,
        policy_id="default",
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.node = storage_node
        self.reservoir_name = reservoir_name
        self.inflow = flow_parameter
        self.children.add(flow_parameter)

        self.run_sensitivity_analysis = run_sensitivity_analysis
        self.sensitivity_analysis_scenarios = sensitivity_analysis_scenarios
        self.policy_id = policy_id
        self.parameters_loaded = False

        # Modifications to
        self.remove_R_max = False

        # Core sizes
        self.M = 3           # n_segments; keep in sync with config if you change it
        self.n_inputs = 3    # [storage, inflow, day]

        # Flags
        self.remove_R_max = False
        self.parameters_loaded = False

        # Placeholders filled by assign_policy_params
        self.S_cap = None
        self.I_bar = None
        self.I_min = None
        self.I_max = None
        self.R_min = None
        self.R_max = None

        # PWL structs
        self.s_bounds = self.s_slopes = self.s_intercepts = None
        self.i_bounds = self.i_slopes = self.i_intercepts = None
        self.d_bounds = self.d_slopes = self.d_intercepts = None

    @classmethod
    def load_default_params(cls):
        """
        Load default piecewise linear parameters from a CSV file.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by reservoir name containing PWL calibration parameters.

        """
        if cls._default_params_cache is None:
            cls._default_params_cache = pd.read_csv(
                pn.operational_constants.get_str("pwl.csv"), 
                sep=",", 
                index_col=["reservoir", "policy_id"]
            )
        return cls._default_params_cache

    def _parse_segment_params(self, segment_params, M=3):
        """segment_params: length 2M-1 → [x1..x_{M-1}, theta1..thetaM]"""
        x_bounds = [0.0] + [float(v) for v in segment_params[:M-1]] + [1.0]
        theta_vals = [float(v) for v in segment_params[M-1:]]
        slopes = [np.tan(theta) for theta in theta_vals]

        intercepts = [0.0]
        for i in range(1, M):
            dx = x_bounds[i] - x_bounds[i-1]
            intercepts.append(intercepts[i-1] + slopes[i-1]*dx)
        return x_bounds, slopes, intercepts

    def _validate_axis(self, bounds, slopes):
        if len(bounds) != len(slopes) + 1:
            raise ValueError("bounds must have one more element than slopes")
        if not all(0.0 <= b <= 1.0 for b in bounds):
            raise ValueError("bounds must be in [0,1]")
        if not all(b2 > b1 for b1, b2 in zip(bounds[:-1], bounds[1:])):
            raise ValueError("bounds must be strictly increasing")
        for m in slopes:
            if not np.isfinite(m):
                raise ValueError("non-finite slope in PWL")

    # ---------- Parameter assignment ----------
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
        if self.reservoir_name in conservation_releases:
            self.R_min = float(conservation_releases[self.reservoir_name])
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

        # --- In assign_policy_params(...) after S_cap/I_bar/R_min/R_max ---

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


    def evaluate_policy(self, X_norm):
        """Evaluate PWL on normalized [S_norm, I_norm, D_norm] -> z in [0,1]."""
        S, I, D = X_norm

        def seg(x, bounds, slopes, intercepts):
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
            for i in range(self.M):
                if bounds[i] <= x < bounds[i + 1]:
                    return slopes[i] * (x - bounds[i]) + intercepts[i]
            return slopes[-1] * (x - bounds[-2]) + intercepts[-1]

        zS = max(0.0, min(1.0, seg(S, self.s_bounds, self.s_slopes, self.s_intercepts)))
        zI = max(0.0, min(1.0, seg(I, self.i_bounds, self.i_slopes, self.i_intercepts)))
        zD = max(0.0, min(1.0, seg(D, self.d_bounds, self.d_slopes, self.d_intercepts)))
        return max(0.0, min(1.0, (zS + zI + zD) / 3.0))

    def value(self, timestep, scenario_index):
        if not self.parameters_loaded:
            if self.run_sensitivity_analysis:
                self.sample_scenario_index = self.sensitivity_analysis_scenarios[
                    scenario_index.indices[0]
                ]
                all_params = self.load_sensitivity_samples(
                    self.sample_scenario_index, "pwl"
                )
                self.policy_params = all_params.loc[self.policy_id]
            else:
                self.pwl_params = self.load_default_params()
            
            self.assign_policy_params(self.pwl_params)
            self.parameters_loaded = True

        # Get state variables 
        S_t = self.node.volume[scenario_index.indices]
        I_t = self.inflow.get_value(scenario_index)
        d = self.model.timestep_times[timestep]
        day_of_year = float(getattr(d, "dayofyear", d.timetuple().tm_yday))

        # make sure I_t and S_t are float
        I_t = float(I_t)
        S_t = float(S_t)
        D_t = float(day_of_year)

        # inputs  = [storage, inflow, day_of_year]
        X = np.array([S_t, I_t, D_t])

        # Normalize X
        X_norm = np.zeros(self.n_inputs)
        for i in range(self.n_inputs):
            X_norm[i] = (X[i] - self.x_min[i]) / (self.x_max[i] - self.x_min[i])        
            X_norm[i] = max(0.0, min(1.0, X_norm[i])) # enforce bounds [0, 1]
        
        # Compute release
        release = self.evaluate_policy(X_norm) * self.R_max

        # Enforce constraints 
        release = min(release, S_t + I_t)  # can't release more than available
        release = max(self.R_min, release)

        return release

    @classmethod
    def load(cls, model, data):
        reservoir_name = data.pop("node")
        storage_node = model.nodes[f"reservoir_{reservoir_name}"]
        flow_parameter = load_parameter(model, f"flow_{reservoir_name}")
        run_sensitivity_analysis = data.pop("run_sensitivity_analysis")
        sensitivity_analysis_scenarios = data.pop("sensitivity_analysis_scenarios")
        policy_id = data.pop("policy_id")
        return cls(
            model,
            reservoir_name,
            storage_node,
            flow_parameter,
            run_sensitivity_analysis,
            sensitivity_analysis_scenarios,
            policy_id,
            **data,
        )

# Register with Pywr
PWLReservoirRelease.register()
