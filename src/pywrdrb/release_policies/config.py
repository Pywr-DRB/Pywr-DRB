"""
Contains configuration required for optimization.
"""
from __future__ import annotations

import os
from copy import deepcopy

import numpy as np
from pywrdrb.utils.constants import cfs_to_mgd, ACRE_FEET_TO_MG

### Random ##################
SEED = 71
# Borg / MOEA result CSV filename seeds (``MMBorg_*_nfe*_seed{N}*.csv``):
# - MRF-filtered bundles use ``..._seed{SEED}_mrffiltered_*`` (same as the MOEA run).
# - Unfiltered (full-objective) bundles omit ``_mrffiltered_``; set ``BORG_SEED_FULL`` when that
#   export used a different seed than ``SEED`` (e.g. separate Slurm batch).
BORG_SEED_FULL = 72
DEBUG = True

### Directories ###########
# Get the directory of this file
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# Other directories relative to this file
DATA_DIR = os.path.join(CONFIG_DIR, "../obs_data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
PUB_RECON_DIR = os.path.join(DATA_DIR, "pub_reconstruction")
OUTPUT_DIR = os.path.join(CONFIG_DIR, "../outputs")
FIG_DIR = os.path.join(CONFIG_DIR, "../figures")

### Constants ###############
cfs_to_mgd = cfs_to_mgd
ACRE_FEET_TO_MG = ACRE_FEET_TO_MG  # Acre-feet to million gallons

### MOEA Settings ##########
# Higher NFE helps fill the objective space and reach extreme trade-offs (thin Pareto sets
# often look "similar" with too few evaluations). Scale up further (e.g. 2e5) if runs are affordable.
NFE = 100000
ISLANDS = 4

# === Objectives ===============================================================
RELEASE_METRICS = [
    "neg_nse",          # release shape/timing (minimize -NSE)
    "Q20_abs_pbias",    # Low flow release percent bias
]

STORAGE_METRICS = [
    "neg_nse",          # storage vs obs: minimize -NSE
    "Q80_abs_pbias",    # high-flow storage: |PBIAS| at Q80
]

METRICS = RELEASE_METRICS + STORAGE_METRICS

# Pretty names for obj* columns. Names say "NSE" for humans, but **on-disk values** follow
# ``METRICS`` (e.g. ``neg_nse`` = minimize −NSE), not raw NSE. Do not infer CSV sign from
# these labels alone.
OBJ_LABELS = {
    "obj1": "Release NSE",
    "obj2": "Q20 Abs % Bias (Release)",
    "obj3": "Storage NSE",
    "obj4": "Q80 Abs % Bias (Storage)",
}

OBJ_FILTER_BOUNDS = {
    "Release NSE": (-1.0, 1.0),
    "Q20 Abs % Bias (Release)": (0.0, 50.0),
    "Storage NSE": (-1.0, 1.0),
    "Q80 Abs % Bias (Storage)": (0.0, 50.0),
}

# --- ε-dominance (Borg / ε-MOEA) -----------------------------------------------
# Each objective is minimized in the form stored in METRICS. Below: what that number is,
# and how you might set ε_j (grid width in the same units as the objective passed to the MOEA).
#
# Release NSE & Storage NSE (keys "neg_nse" in METRICS):
#   Optimizer minimizes  neg_nse = −NSE.
#   Nash–Sutcliffe:  NSE = 1 − Σ(y_sim − y_obs)² / Σ(y_obs − ȳ_obs)²
#   Perfect match → NSE = 1 → neg_nse = −1.  Worse models → neg_nse increases toward 0 or above.
#   Example ε scale:  ε ← f × (NSE_hi − NSE_lo)  with NSE in [−1, 1] from OBJ_FILTER_BOUNDS → span 2;
#   e.g. f = 0.015 gives ε ≈ 0.03 on the neg_nse axis.
#
# Q20 / Q80 absolute percent bias (release & storage):
#   Optimizer minimizes  |PBIAS| (%) in the low-flow (Q20) or high-flow (Q80) FDC regime.
#   Common whole-series percent bias:  PBIAS = 100 × Σ(y_sim − y_obs) / Σ(y_obs)
#   Q20 / Q80 variants restrict y to the duration-curve slice near the 20th or 80th percentile;
#   use the same definition as your evaluation code.  Ideal |PBIAS| = 0.
#   Example ε scale:  ε ← f × (bias_hi − bias_lo)  from OBJ_FILTER_BOUNDS → span 50;
#   e.g. f = 0.015 gives ε ≈ 0.75.
#
# General tuning: smaller ε → finer Pareto grid (more distinct boxes); larger ε → coarser grid.
# -----------------------------------------------------------------------------

EPSILONS_BY_OBJECTIVE = {
    "Release NSE": 0.03,
    "Q20 Abs % Bias (Release)": 0.75,
    "Storage NSE": 0.03,
    "Q80 Abs % Bias (Storage)": 0.75,
}

# List order must match METRICS (via obj1 … objN in OBJ_LABELS).
EPSILONS = [
    EPSILONS_BY_OBJECTIVE[OBJ_LABELS[f"obj{i}"]]
    for i in range(1, len(METRICS) + 1)
]

# Column names in each Borg ``MMBorg_*.csv`` row for objectives (``obj1`` … ``objN``).
# Order matches :data:`EPSILONS`, :data:`METRICS`, and :data:`OBJ_LABELS`.
MOEA_OBJECTIVE_CSV_KEYS = tuple(f"obj{i}" for i in range(1, len(OBJ_LABELS) + 1))

# === Objective senses & baseline aliases (needed by selection/plotting) ======
# **Not** the sign stored in raw ``MMBorg_*.csv`` ``obj*`` columns. Those follow ``METRICS``
# (all minimized: ``neg_nse``, |PBIAS|, …). :data:`SENSES_ALL` describes the *physical*
# preference (e.g. higher NSE is better) after any sign convention used in figures /
# ``load_results``. Post-processing ε-Pareto sorts on raw Borg rows should **minimize** every
# ``obj*`` — the default in ``pareto.eps_sort`` — and must **not** use ``maximize=`` for NSE
# columns that are already ``neg_nse`` on disk.
SENSES_ALL = {
    "Release NSE": "max",
    "Q20 Abs % Bias (Release)": "min",
    "Storage NSE": "max",
    "Q80 Abs % Bias (Storage)": "min",
}

BASELINE_ALIASES = {
    "Release NSE":              "neg_nse",                 # your baseline CSV keys
    "Q20 Abs % Bias (Release)": "Q20_abs_pbias",
    "Storage NSE":              "neg_nse",
    "Q80 Abs % Bias (Storage)": "Q80_abs_pbias",
}

BASELINE_VALUE_COL = "pywr_baseline"

# Symmetric inertia settings by reservoir (release + storage)
# scale ∈ {"range","max","value"}; for "value", provide scale_value (S0)
INERTIA_BY_RESERVOIR = {
    "prompton": {
        "release": {"scale": "range", "tau": 0.025, "scale_value": None},
        "storage": {"scale": "range", "tau": 0.020, "scale_value": None},
    },
    "fewalter": {
        "release": {"scale": "value", "tau": 0.020, "scale_value": None},  # was "max"
        "storage": {"scale": "value", "tau": 0.038, "scale_value": 1790.0},
    },
    "blueMarsh": {
        "release": {"scale": "value", "tau": 0.018, "scale_value": None},  # was "max"
        "storage": {"scale": "value", "tau": 0.035, "scale_value": 2116.0175},
    },
    "beltzvilleCombined": {
        "release": {"scale": "value", "tau": 0.030, "scale_value": None},  # was "max"
        "storage": {"scale": "value", "tau": 0.025, "scale_value": 2415.8529},
    },
}


### Reservoirs ###############
reservoir_options = [
    'beltzvilleCombined',
    'fewalter',
    'prompton',
    'blueMarsh', 
]

### Polcy Settings ###############

policy_type_options = [
    "STARFIT",
    "RBF",
    "PWL",
]

# === Baseline/validation settings used by plotting ===
BASELINE_DIR_NAME   = "baseline_pywr"   # subfolder under FIG_DIR that holds baseline CSVs
BASELINE_INFLOW_TAG = "inflow_pub"      # which inflow series name the baseline used
VAL_START           = "2019-01-01"      # validation window (string or pandas-parseable)
VAL_END             = "2024-12-31"


## RBF
n_rbfs = 3              # Number of radial basis functions (RBFs) used in the policy
n_rbf_inputs = 3         # Number of input variables (inflow, storage, day_of_year)
n_rbf_params = n_rbfs * (2 * n_rbf_inputs + 1)
rbf_param_bounds = [[0.0, 1.0]] * n_rbf_params

## STARFIT
n_starfit_params = 17         # Number of parameters in STARFIT policy
STARFIT_PARAM_NAMES = (
    "NORhi_mu",
    "NORhi_min",
    "NORhi_max",
    "NORhi_alpha",
    "NORhi_beta",
    "NORlo_mu",
    "NORlo_min",
    "NORlo_max",
    "NORlo_alpha",
    "NORlo_beta",
    "Release_alpha1",
    "Release_alpha2",
    "Release_beta1",
    "Release_beta2",
    "Release_c",
    "Release_p1",
    "Release_p2",
)
assert len(STARFIT_PARAM_NAMES) == n_starfit_params
STARFIT_PARAM_NAME_TO_IDX = {name: i for i, name in enumerate(STARFIT_PARAM_NAMES)}
n_starfit_inputs = 3         # Number of input variables (inflow, storage, week_of_year)

# Starfit parameter bounds (global default; per-reservoir overrides below)
starfit_param_bounds_default = [
    [0.0, 100.0],         # NORhi_mu
    [0.0, 100.0],         # NORhi_min
    [0.07, 100.0],        # NORhi_max
    [-10.95, 79.63],      # NORhi_alpha
    [-44.29, 5.21],       # NORhi_beta

    [0.0, 100.0],         # NORlo_mu
    [0.0, 100.0],         # NORlo_min
    [1.76, 100.0],        # NORlo_max
    [-14.41, 11.16],      # NORlo_alpha
    [-45.21, 5.72],       # NORlo_beta

    [-4.088, 240.9161],   # Release_alpha1
    [-0.5901, 84.7844],   # Release_alpha2
    [-1.2104, 83.9024],   # Release_beta1
    [-52.3545, 0.4454],   # Release_beta2

    [-1.414, 63.516],     # Release_c
    [0.0, 97.625],        # Release_p1
    [0.0, 0.957],         # Release_p2
]

starfit_param_bounds = starfit_param_bounds_default

# Per-reservoir: only listed keys replace entries from ``starfit_param_bounds_default``.
# Use ``STARFIT_PARAM_NAMES`` keys; values are ``[lower, upper]`` in the same units as
# ``starfit_param_bounds_default`` (NOR min/max: storage % on 0–100 scale, matching CSV/Pywr).
#
# Example — tighten only NOR band for one reservoir while leaving release terms global:
#   "someReservoir": {"NORhi_min": [5.0, 30.0], "NORhi_max": [5.0, 30.0], ...}
STARFIT_PARAM_BOUNDS_OVERRIDES_BY_RESERVOIR = {
    "blueMarsh": {
        "NORhi_min": [10.0,25.0],
        "NORhi_max": [10.0, 25.0],
        "NORlo_min": [10.0, 25.0],
        "NORlo_max": [10.0, 25.0],
    },
    "fewalter": {
        "NORhi_min": [1.0, 30.0],
        "NORhi_max": [1.0, 30.0],
        "NORlo_min": [1.0, 30.0],
        "NORlo_max": [1.0, 30.0],
    },
    "prompton": {
        "NORhi_min": [1.0, 10.0],
        "NORhi_max": [1.0, 10.0],
        "NORlo_min": [1.0, 10.0],
        "NORlo_max": [1.0, 10.0],
    },
}


def build_starfit_param_bounds_for_reservoir(reservoir_name: str) -> list[list[float]]:
    """Return STARFIT bounds for ``reservoir_name``: defaults merged with per-reservoir overrides."""
    overrides = STARFIT_PARAM_BOUNDS_OVERRIDES_BY_RESERVOIR.get(reservoir_name)
    if not overrides:
        raise KeyError(
            f"No STARFIT bounds overrides for '{reservoir_name}'. "
            f"Known: {sorted(STARFIT_PARAM_BOUNDS_OVERRIDES_BY_RESERVOIR)}"
        )
    b = deepcopy(starfit_param_bounds_default)
    for pname, pair in overrides.items():
        if pname not in STARFIT_PARAM_NAME_TO_IDX:
            raise KeyError(
                f"Unknown STARFIT param '{pname}' in overrides for '{reservoir_name}'. "
                f"Valid names: {STARFIT_PARAM_NAMES}"
            )
        if len(pair) != 2:
            raise ValueError(
                f"Override for '{reservoir_name}'.'{pname}' must be [lo, hi]; got {pair!r}"
            )
        i = STARFIT_PARAM_NAME_TO_IDX[pname]
        b[i] = [float(pair[0]), float(pair[1])]
    return b


def get_starfit_param_bounds(reservoir_name: str) -> list[list[float]]:
    """STARFIT bounds for Borg / validation: merged overrides when configured, else global default."""
    if reservoir_name in STARFIT_PARAM_BOUNDS_OVERRIDES_BY_RESERVOIR:
        return build_starfit_param_bounds_for_reservoir(reservoir_name)
    return deepcopy(starfit_param_bounds_default)

# ---------------- Piecewise Linear (PWL) ----------------
# Each input block uses: [x1..x_{M-1}, theta1..thetaM]
# - x_i are internal breakpoints in (0, 1), enforced strictly increasing in the policy code.
# - theta_j are line angles (radians) converted to slopes via tan(theta_j).
#   Using a moderate angle range avoids extreme slopes.


def make_pwl_bounds(n_segments: int, n_inputs: int, *, eps: float = 1e-3):
    """Return (n_params, bounds) for PWL with M segments and I inputs."""
    per_block = 2 * n_segments - 1
    n_params  = per_block * n_inputs
    # Breakpoints in (eps, 1-eps); ANGLES kept safely away from ±pi/2
    bounds_one = (
        [[eps, 1.0 - eps]] * (n_segments - 1) +     # x1..x_{M-1}
        [[-np.pi/3, np.pi/3]] * n_segments          # theta1..thetaM  (SAFE)
    )
    bounds = []
    for _ in range(n_inputs):
        bounds += bounds_one
    return n_params, bounds

# Wire it in
n_segments     = 4
n_pwl_inputs   = 3
n_pwl_params, pwl_param_bounds = make_pwl_bounds(n_segments, n_pwl_inputs)

# --- Keep the legacy for reference (but commented) ---
# single_input_pwl_param_bounds = [
#     [[i/(n_segments-1), (i+1)/(n_segments-1)] for i in range(n_segments-1)] +
#     [[-np.pi/2, np.pi/2]] * n_segments
# ][0]
# pwl_param_bounds = []
# for _ in range(n_pwl_inputs):
#     pwl_param_bounds += single_input_pwl_param_bounds


## Dictionaries of configurations
policy_n_params = {
    "STARFIT": n_starfit_params,
    "RBF": n_rbf_params,                  
    "PWL": n_pwl_params,  
}

policy_param_bounds = {
    "STARFIT": starfit_param_bounds_default,
    "RBF": rbf_param_bounds,
    "PWL": pwl_param_bounds,
}

#### RESERVOIR CONSTRAINTS ##############

# Storage capacities (MG)
# NOTE: For Beltzville, OBS storage max is 17,736 MG while we currently use 13,500 MG (crest).
reservoir_capacity = {
    "prompton": 27956.02,
    "beltzvilleCombined": 17736.09, #48317.0588,   # OBS max 17736.09 #Old 13500.0
    "fewalter": 35800.0,
    "blueMarsh": 42320.35,
}

### Low Storage (Deadpool) Fractions ###########################################
# Fractions are computed as (dead storage / total capacity).
# All cited from USGS NWIS "REMARKS" sections unless noted otherwise.
# These represent the approximate inactive (dead) storage proportions.
#
# ┌──────────────────────┬──────────────┬────────────────────────────┬────────────────────────────────────────────────────────────────────┐
# │ Reservoir            │ Dead Storage │ Fraction of Capacity (≈)   │ Primary Citation                                                   │
# ├──────────────────────┼──────────────┼────────────────────────────┼────────────────────────────────────────────────────────────────────┤
# │ Blue Marsh Lake      │ 2,560 ac-ft  │ 0.00192                    │ USGS WYS 01470870 “Blue Marsh Lake near Bernville, PA.”           │
# │                      │              │                            │ Records Oct 2010–present: “Dead storage is 2,560 acre-ft.”         │
# │                      │              │                            │ https://waterdata.usgs.gov/nwis/uv?site_no=01470870&legacy=1       │
# ├──────────────────────┼──────────────┼────────────────────────────┼────────────────────────────────────────────────────────────────────┤
# │ Beltzville Lake      │ 1,390 ac-ft  │ 0.00106                    │ USGS NWIS 01449790 “Beltzville Lake near Parryville, PA.”         │
# │                      │              │                            │ “Dead storage is 1,390 acre-ft.”                                   │
# │                      │              │                            │ https://waterdata.usgs.gov/nwis/uv?site_no=01449790&legacy=1       │
# ├──────────────────────┼──────────────┼────────────────────────────┼────────────────────────────────────────────────────────────────────┤
# │ F.E. Walter Reservoir│ 2,000 ac-ft  │ 0.00159                    │ USGS NWIS 01447780 “Francis E. Walter Reservoir near White Haven.”│
# │                      │              │                            │ “Dead storage is 2,000 acre-ft.”                                   │
# │                      │              │                            │ https://waterdata.usgs.gov/nwis/uv?site_no=01447780&legacy=1       │
# ├──────────────────────┼──────────────┼────────────────────────────┼────────────────────────────────────────────────────────────────────┤
# │ Prompton Reservoir   │ —            │ 0.035 (fallback)           │ USGS 01428900 lists minimum conservation pool 3,420 ac-ft at elev 1125 ft │
# │                      │              │                            │ but no explicit “dead storage” value.                              │
# │                      │              │                            │ https://waterdata.usgs.gov/nwis/uv?site_no=01428900&legacy=1       │
# └──────────────────────┴──────────────┴────────────────────────────┴────────────────────────────────────────────────────────────────────┘

LOW_STORAGE_FRACTION_BY_RES = {
    "blueMarsh": 0.00192,
    "beltzvilleCombined": 0.00106,
    "fewalter": 0.00159,
    "prompton": 0.035,   # fallback (no published deadpool)
}


# Inflow bounds used for normalization (MGD)
inflow_bounds_by_reservoir = {
    "prompton":           {"I_min": 0.0, "I_max": 7500.00},   # I_max = 1740.00 × 1.5 = 2610.00
    "beltzvilleCombined": {"I_min": 0.0, "I_max": 3002.50},   # I_max = 1440.00 × 1.5 = 2160.00
    "fewalter":           {"I_min": 0.0, "I_max": 30000.00},  # I_max = 7690.00 × 1.5 = 11535.00
    "blueMarsh": {"I_min": 0.0, "I_max": 7500.00},  
}

# Conservation minimums (MGD) from DRBC Water Code
drbc_conservation_releases = {
    "blueMarsh": 50 * cfs_to_mgd,
    "beltzvilleCombined": 35 * cfs_to_mgd,  # ~22.61 MGD
    #"fewalter": 50 * cfs_to_mgd,            # ~32.30 MGD
}

# Release maxima (MGD) updated from your OBS maxima
release_max_by_reservoir = {
    "prompton":           2610.00,  # R_max = 1740.00 × 1.5 = 2610.00, 231.60651
    "beltzvilleCombined": 969.5,  # R_max = 1440.00 × 1.5 = 2160.00
    "fewalter":           1292.6,  # R_max = 7690.00 × 1.5 = 11535.00, 1292.6
    "blueMarsh":          969.5,
}

# promton observed minimum reported (~5.75 MGD).
release_min_by_reservoir = {
    "prompton": 5.75,  # from CTX print
}

# --- Build BASE_POLICY_CONTEXT directly from the dicts above ---

def _rmin(name: str) -> float:
    # Prefer DRBC conservation if present, else any explicit per-reservoir min, else 0.
    if name in drbc_conservation_releases:
        return float(drbc_conservation_releases[name])
    return float(release_min_by_reservoir.get(name, 0.0))

def _rmax(name: str) -> float:
    return float(release_max_by_reservoir[name])

def _icap(name: str) -> float:
    return float(reservoir_capacity[name])

def _ibounds(name: str) -> tuple[float, float]:
    b = inflow_bounds_by_reservoir[name]
    return (float(b["I_min"]), float(b["I_max"]))


BASE_POLICY_CONTEXT_BY_RESERVOIR = {
    name: {
        "release_min": _rmin(name),
        "release_max": _rmax(name),
        "storage_capacity": _icap(name),
        "x_min": (0.0, _ibounds(name)[0], 1.0),
        "x_max": (_icap(name), _ibounds(name)[1], 366.0),
        "low_storage_threshold": LOW_STORAGE_FRACTION_BY_RES[name] * _icap(name),
    }
    for name in reservoir_options
}

def get_policy_context(
    reservoir_name: str,
    *,
    release_min_override: float | None = None,
    release_max_override: float | None = None,
    capacity_override: float | None = None,
    inflow_bounds_override: tuple[float, float] | None = None,
    low_storage_threshold_override: float | None = None,
) -> dict:
    from copy import deepcopy
    try:
        base = BASE_POLICY_CONTEXT_BY_RESERVOIR[reservoir_name]
    except KeyError as e:
        available = ", ".join(sorted(BASE_POLICY_CONTEXT_BY_RESERVOIR))
        raise KeyError(f"Unknown reservoir '{reservoir_name}'. Known: {available}") from e

    ctx = deepcopy(base)

    R_min = float(ctx["release_min"])
    R_max = float(ctx["release_max"])
    S_cap = float(ctx["storage_capacity"])
    _, I_min_base, _ = ctx["x_min"]
    _, I_max_base, _ = ctx["x_max"]
    S_low = float(base["low_storage_threshold"])

    if release_min_override is not None:
        R_min = float(release_min_override)
    if release_max_override is not None:
        R_max = float(release_max_override)
    if capacity_override is not None:
        S_cap = float(capacity_override)
    if inflow_bounds_override is not None:
        I_min, I_max = map(float, inflow_bounds_override)
    else:
        I_min, I_max = float(I_min_base), float(I_max_base)
    if low_storage_threshold_override is not None:
        S_low = float(low_storage_threshold_override)
    else:
        # keep S_low consistent with possibly overridden capacity
        S_low = max(0.0, LOW_STORAGE_FRACTION_BY_RES[reservoir_name] * S_cap) if capacity_override is not None else S_low

    if not (I_max > I_min):
        raise ValueError(f"{reservoir_name}: I_max ({I_max}) must be > I_min ({I_min}).")
    if not (R_max >= R_min >= 0.0):
        raise ValueError(f"{reservoir_name}: invalid release bounds [{R_min}, {R_max}].")

    return {
        "release_min": R_min,
        "release_max": R_max,
        "storage_capacity": S_cap,
        "x_min": (0.0, I_min, 1.0),
        "x_max": (S_cap, I_max, 366.0),
        "low_storage_threshold": S_low,
    }

# Optional: precompute
POLICY_CONTEXT_BY_RESERVOIR = {r: get_policy_context(r) for r in reservoir_options}



def parse_params_inline(policy_type: str, params: str | list[float]) -> list[float]:
    """Accepts comma-separated string or list of floats."""
    if params is None:
        raise ValueError("parse_params_inline: 'params' is None.")
    if isinstance(params, str):
        # allow whitespace, mixed commas
        vec = [float(x.strip()) for x in params.replace("，", ",").split(",") if x.strip() != ""]
    elif isinstance(params, (list, tuple)):
        vec = [float(x) for x in params]
    else:
        raise TypeError(f"Unsupported params type: {type(params)}")

    # optional: quick length checks to fail fast (kept permissive if you change bounds later
    expected = policy_n_params.get(policy_type)
    if expected is not None and len(vec) != expected:
        raise ValueError(f"{policy_type} expects {expected} params; got {len(vec)}.")
    return vec