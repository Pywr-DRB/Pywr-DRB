"""
Parametric (STARFIT/RBF/PWL) Pywr parameter for reservoir release

Overview
--------
`ParametricReservoirRelease` is a custom Pywr `Parameter` that computes a
reservoir’s **daily release (MGD)** from storage, inflow, and seasonality using
one of three policy families: **STARFIT**, **RBF**, or **PWL**. The class
provides a thin, robust integration layer that:

1) Resolves the **operating context** (storage capacity, release bounds,
   inflow scaling bounds) via `get_policy_context(...)`, with YAML overrides.
2) Loads a **policy parameter vector** either **inline** (from YAML) or from an
   **operational-constants CSV** (resolved through `path_manager`).
3) Injects that vector into the chosen policy and performs policy-specific
   validation/parse steps when available.
4) Returns `policy.get_release(storage, inflow, day_of_year)` at each timestep.

This parameter is used to evaluate calibrated operating rules and to explore
alternative policies (e.g., DRBC policy testing, robustness/sensitivity runs)
within the PywrDRB modeling framework.

Key Steps
---------
1. **Initialize policy**  
   Construct the chosen policy (`STARFIT`, `RBF`, or `PWL`) without params.

2. **Attach operating context**  
   Call `get_policy_context(reservoir_name, ...)` to obtain canonical
   bounds/scaling and pass them to `policy.set_context(**ctx)`. Optional YAML
   overrides (`R_min`, `R_max`, `S_cap`, `I_min`, `I_max`) take precedence.

3. **Load parameters**
   - **Inline**: if `params_inline` is provided, it is used directly (exact order
     must match the policy’s expected vector).
   - **CSV**: otherwise, load from the policy’s CSV:
       * `starfit.csv` for STARFIT  
       * `rbf.csv` for RBF  
       * `pwl.csv` for PWL  
     The CSV is resolved with `pn.operational_constants.get_str(...)`. The row
     is selected by `(reservoir, policy_id)` with a fallback to
     `(reservoir, "default")`.

4. **Assign parameters to policy**
   - If the policy exposes `assign_policy_params(row, ...)`, use it.
   - Else, extract the family-specific columns listed in `_VARIABLE_NAMES[...]`,
     build the flat vector, assign it to `policy.policy_params`, and call
     `validate_policy_params()` / `parse_policy_params()` if present.

5. **STARFIT normalization (if available)**
   If the CSV row includes `Adjusted_MEANFLOW_MGD` (preferred) or
   `GRanD_MEANFLOW_MGD`, set `policy.I_bar` accordingly to enable STARFIT’s
   internal normalization.

6. **Return release each timestep**
   At `value(timestep, scenario_index)`, read storage (MG) and inflow (MGD),
   compute `dayofyear`, and return the policy’s release (MGD).

Technical Notes
---------------
- **Units (DRB convention)**:
  - Storage (S): **MG** from `storage_node.volume`
  - Flow (I, release): **MGD**
  - Time index: `dayofyear` ∈ [1, 366]
- **Context precedence**: YAML overrides → `get_policy_context(...)` defaults.
- **CSV schema**:
  - Expected columns depend on policy family; see `_VARIABLE_NAMES[...]`.
  - MultiIndex by `["reservoir", "policy_id"]` is supported for row selection.
- **Scenario safety**: storage access is robust to Pywr scenario indexing.
- **Sensitivity/robustness**: `run_sensitivity_analysis` and
  `sensitivity_analysis_scenarios` are stored for orchestration by higher-level
  tooling; this class does not execute sweeps itself.

Inputs
------
Required (from YAML / loader):
- `node` (str): reservoir name key (used to find `reservoir_<name>` storage node)
- `flow_parameter_name` (str): name of an existing Pywr flow `Parameter` (MGD)
- `policy_type` (str): one of `STARFIT`, `RBF`, `PWL`

Optional:
- `policy_id` (str): CSV row selector (default `"default"`)
- `params_inline` (list[float]): vector in exact policy order (bypasses CSV)
- Context overrides: `R_min`, `R_max` (MGD); `S_cap` (MG); `I_min`, `I_max` (MGD)
- Sensitivity flags: `run_sensitivity_analysis`, `sensitivity_analysis_scenarios`

Outputs
-------
- Pywr `Parameter.value(...)` returns **release in MGD** for the current timestep.

Failure Modes & Diagnostics
---------------------------
- `ValueError`: unknown `policy_type`
- `FileNotFoundError`: missing policy CSV (if `params_inline` is not provided)
- `KeyError`: required CSV columns absent or row not found for
  `(reservoir, policy_id)` nor `(reservoir, "default")`
- Parameter order mismatch (when using `params_inline`): ensure the vector
  matches the policy’s expected order (see `_VARIABLE_NAMES[...]` or the policy’s
  own documentation).

Examples
--------
YAML
~~~~
parameters:
  fewalter_release:
    type: ParametricReservoirRelease
    node: fewalter
    flow_parameter_name: flow_fewalter
    policy_type: STARFIT
    policy_id: calibrated_v1
    # Optional inline vector (bypasses CSV)
    # params_inline: [15.08, 5.0, 20.0, 0.0, -15.0, 9.0, 1.6, 14.2, -1.0, -30.0,
    #                 0.2118, -0.0357, 0.1302, -0.0248, -0.123, 0.183, 0.732]
    # Optional context overrides (units MG / MGD)
    R_min: 32.3
    R_max: 4900.0
    S_cap: 35800.0
    I_min: 0.0
    I_max: 20000.0
    run_sensitivity_analysis: false
    sensitivity_analysis_scenarios: []

Python
~~~~~~
param = ParametricReservoirRelease(
    model=model,
    reservoir_name="fewalter",
    storage_node=model.nodes["reservoir_fewalter"],
    flow_parameter=load_parameter(model, "flow_fewalter"),
    run_sensitivity_analysis=False,
    sensitivity_analysis_scenarios=[],
    policy_type="STARFIT",
    policy_id="default",
    # params_inline=[...],  # optional: bypass CSV
    R_min=32.3, R_max=4900.0, S_cap=35800.0, I_min=0.0, I_max=20000.0,
)

Links
-----
- Turner, S.W.D., Steyaert, J.C., Condon, L., & Voisin, N. (2021).
  Water storage and release policies for all large reservoirs of the
  conterminous United States. *Environmental Modelling & Software*, 145, 105201.
  https://doi.org/10.1016/j.envsoft.2021.105201

Change Log
----------
- Marilyn Smith — 2025-09-24 Initial version of parametric release parameter
"""

import pandas as pd
import os

from pywr.parameters import Parameter, load_parameter

from pywrdrb.path_manager import get_pn_object
from pywrdrb.release_policies.config import get_policy_context, n_segments, n_rbfs
from pywrdrb.release_policies import RBF, PWL, STARFIT

pn = get_pn_object()

_POLICY_LABEL = {"PWL": "PWL", "RBF": "RBF", "STARFIT": "STARFIT"}

_POLICY_FILENAME = {
    "RBF": "rbf.csv",
    "PWL": "pwl.csv",
    # Dedicated STARFIT optimization defaults CSV (from CEE pipeline handoff).
    "STARFIT": "starfit.csv",
}


def _pwl_csv_column_names() -> list[str]:
    """Names expected in pwl.csv for current n_segments (three axes: storage, inflow, season)."""
    names: list[str] = []
    for prefix in ("storage", "inflow", "season"):
        for k in range(1, n_segments):
            names.append(f"{prefix}_x{k}")
        for k in range(1, n_segments + 1):
            names.append(f"{prefix}_theta{k}")
    return names


def _rbf_csv_column_names() -> list[str]:
    """Names expected in rbf.csv for current n_rbfs (order matches RBF.assign_policy_params keys)."""
    cols: list[str] = []
    for i in range(1, n_rbfs + 1):
        cols.extend(
            [
                f"rbf{i}_center_storage",
                f"rbf{i}_center_inflow",
                f"rbf{i}_center_doy",
                f"rbf{i}_scale_storage",
                f"rbf{i}_scale_inflow",
                f"rbf{i}_scale_doy",
                f"rbf{i}_weight",
            ]
        )
    return cols


_VARIABLE_NAMES = {
    "STARFIT": [
        "NORhi_mu","NORhi_min","NORhi_max","NORhi_alpha","NORhi_beta",
        "NORlo_mu","NORlo_min","NORlo_max","NORlo_alpha","NORlo_beta",
        "Release_alpha1","Release_alpha2","Release_beta1","Release_beta2",
        "Release_c","Release_p1","Release_p2"
    ],
    "RBF": _rbf_csv_column_names(),
    "PWL": _pwl_csv_column_names(),
}

class ParametricReservoirRelease(Parameter):
    """Parametric reservoir release parameter that delegates to RBF/PWL/STARFIT policies."""

    def __init__(
        self,
        model,
        reservoir_name,
        storage_node,
        flow_parameter,
        run_sensitivity_analysis,
        sensitivity_analysis_scenarios,
        policy_type,
        policy_id="default",
        params_inline=None,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        
        self.node = storage_node
        self.inflow = flow_parameter
        self.children.add(flow_parameter)

        self.reservoir_name = reservoir_name
        self.policy_id = policy_id
        self.policy_type = policy_type
        self.params_inline = params_inline

        self.run_sensitivity_analysis = run_sensitivity_analysis
        self.sensitivity_analysis_scenarios = sensitivity_analysis_scenarios

        # Optional context overrides (if provided)
        self.R_min = kwargs.pop("R_min", None)
        self.R_max = kwargs.pop("R_max", None)
        self.S_cap = kwargs.pop("S_cap", None)
        self.I_min = kwargs.pop("I_min", None)
        self.I_max = kwargs.pop("I_max", None)

        self.policy = None

    # ---------- helpers ----------
    def _init_policy(self) -> None:
        if self.policy_type == "PWL":
            self.policy = PWL(policy_params=None)
        elif self.policy_type == "RBF":
            self.policy = RBF(policy_params=None)
        elif self.policy_type == "STARFIT":
            self.policy = STARFIT(policy_params=None, reservoir_name=self.reservoir_name)
        else:
            raise ValueError(f"Invalid policy type: {self.policy_type}")

    def _policy_csv_path(self) -> str:
        try:
            fname = _POLICY_FILENAME[self.policy_type]
        except KeyError:
            raise ValueError(f"Unknown policy_type '{self.policy_type}'. Expected one of {list(_POLICY_FILENAME)}.")
        path = pn.operational_constants.get_str(fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Operational-constants CSV not found: {path}")
        return path

    def _select_row(self, df: pd.DataFrame, csv_path: str) -> pd.Series:
        # Handle MultiIndex (['reservoir','policy_id'])
        if isinstance(df.index, pd.MultiIndex) and set(df.index.names) >= {"reservoir", "policy_id"}:
            key = (self.reservoir_name, self.policy_id)
            if key not in df.index:
                key = (self.reservoir_name, "default")
                if key not in df.index:
                    raise KeyError(f"Params not found in {csv_path} for {self.reservoir_name}/{self.policy_id} (and no 'default').")
            return df.loc[key]

        # Handle standard CSVs where reservoir/policy_id are regular columns.
        required_cols = {"reservoir", "policy_id"}
        if not required_cols.issubset(df.columns):
            raise KeyError(
                f"{csv_path} must contain columns {sorted(required_cols)} "
                f"or have a MultiIndex with names ['reservoir', 'policy_id']."
            )

        exact = df[
            (df["reservoir"].astype(str) == str(self.reservoir_name))
            & (df["policy_id"].astype(str) == str(self.policy_id))
        ]
        if not exact.empty:
            return exact.iloc[0]

        fallback = df[
            (df["reservoir"].astype(str) == str(self.reservoir_name))
            & (df["policy_id"].astype(str) == "default")
        ]
        if not fallback.empty:
            return fallback.iloc[0]

        raise KeyError(
            f"Params not found in {csv_path} for {self.reservoir_name}/{self.policy_id} "
            "(and no 'default')."
        )

    def _apply_row_to_policy(self, row: pd.Series) -> None:
        """
        Prefer a policy's own assigner if it exists (RBF/PWL/STARFIT policy classes can expose
        `assign_policy_params`). Otherwise, fall back to vector injection using _VARIABLE_NAMES.
        """
        # If policy provides a row-based assigner, use it.
        if hasattr(self.policy, "assign_policy_params"):
            self.policy.assign_policy_params(row, set_context_from_row=False)
        else:
            family_label = _POLICY_LABEL[self.policy_type]
            cols = _VARIABLE_NAMES[family_label]
            vec = pd.Series(row)[cols].astype(float).tolist()

            # Generic injection path: set vector, then validate/parse if available
            if hasattr(self.policy, "policy_params"):
                self.policy.policy_params = vec
            if hasattr(self.policy, "validate_policy_params"):
                self.policy.validate_policy_params()
            if hasattr(self.policy, "parse_policy_params"):
                self.policy.parse_policy_params()
            
            # NOTE: Do not set I_bar here. STARFIT loads I_bar internally from istarf_capacity.csv.

    def _load_params_from_csv(self) -> None:
        csv_path = self._policy_csv_path()
        df = pd.read_csv(csv_path)

        # sanity: vector columns exist
        family_label = _POLICY_LABEL[self.policy_type]
        cols = _VARIABLE_NAMES[family_label]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"CSV {csv_path} missing expected columns for {self.policy_type}: {missing}")

        row = self._select_row(df, csv_path)
        self._apply_row_to_policy(row)


    # ---------- Pywr lifecycle ----------
    def setup(self):
        super().setup()
        self._init_policy()

        # Set context ONCE from model metadata/overrides
        ctx = get_policy_context(
            self.reservoir_name,
            release_min_override=self.R_min,
            release_max_override=self.R_max,
            capacity_override=self.S_cap,
            inflow_bounds_override=(self.I_min, self.I_max) if (self.I_min is not None and self.I_max is not None) else None,
        )
        self.policy.set_context(**ctx)

        # Prefer inline vector over CSV
        if self.params_inline is not None:
            # Direct-injection path: treat like optimizer vector
            if hasattr(self.policy, "policy_params"):
                self.policy.policy_params = list(map(float, self.params_inline))
            if hasattr(self.policy, "validate_policy_params"):
                self.policy.validate_policy_params()
            if hasattr(self.policy, "parse_policy_params"):
                self.policy.parse_policy_params()
        else: 
            # Load optimizer vector from CSV and assign to policy (no extra context set)
            self._load_params_from_csv()

        # Align with STARFITReservoirRelease: below NOR_lo, ramp release via S_hat/NORlo (not R_min-only).
        if self.policy_type == "STARFIT" and self.policy is not None:
            self.policy.linear_below_NOR = True

    def value(self, timestep, scenario_index):
        # scenario-safe storage
        S_t = float(self.node.volume[scenario_index.indices]) if hasattr(scenario_index, "indices") else float(self.node.volume)
        # robust inflow access
        I_t = float(self.inflow.value(timestep, scenario_index)) if hasattr(self.inflow, "value") \
              else float(self.inflow.get_value(scenario_index))
        D_t = float(timestep.dayofyear)
        return float(self.policy.get_release(S_t, I_t, D_t))

    # ---------- YAML loader ----------
    @classmethod
    def load(cls, model, data):
        reservoir_name = data.pop("node")
        storage_node = model.nodes[f"reservoir_{reservoir_name}"]
        flow_param_name = data.pop("flow_parameter_name", f"flow_{reservoir_name}")
        flow_parameter = load_parameter(model, flow_param_name)
        run_sensitivity_analysis = data.pop("run_sensitivity_analysis")
        sensitivity_analysis_scenarios = data.pop("sensitivity_analysis_scenarios")
        policy_id = data.pop("policy_id", "default")
        policy_type = data.pop("policy_type")
        params_inline = data.pop("params_inline", None) 

        return cls(
            model,
            reservoir_name,
            storage_node,
            flow_parameter,
            run_sensitivity_analysis,
            sensitivity_analysis_scenarios,
            policy_type=policy_type,
            policy_id=policy_id,
            params_inline=params_inline,
            **data,
        )

# Register
ParametricReservoirRelease.register()
