"""
Contains configuration required for simulation.
"""
import os
import numpy as np
from copy import deepcopy
from typing import Tuple
from pywrdrb.utils.constants import cfs_to_mgd, ACRE_FEET_TO_MG

### Random ##################
SEED = 71
DEBUG = True

### Directories ###########
# Get the directory of this file
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# Other directories relative to this file
DATA_DIR = os.path.join(CONFIG_DIR, "../obs_data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(CONFIG_DIR, "../outputs")
FIG_DIR = os.path.join(CONFIG_DIR, "../figures")

### Constants ###############
cfs_to_mgd = cfs_to_mgd
ACRE_FEET_TO_MG = ACRE_FEET_TO_MG  # Acre-feet to million gallons

### MOEA Settings ##########
NFE = 30000
ISLANDS = 4

RELEASE_METRICS = [
    'neg_nse',           # Negative Nash Sutcliffe Efficiency
    'Q20_abs_pbias',         # Absolute Percent Bias
    'Q80_abs_pbias',         # Absolute Percent Bias
]

STORAGE_METRICS = [
    'neg_nse',           # Negative Nash Sutcliffe Efficiency
]

METRICS = RELEASE_METRICS + STORAGE_METRICS

EPSILONS = [0.01] * len(METRICS) # Epsilon values for Borg

OBJ_LABELS = {
        "obj1": "Release NSE",
        "obj2": "Release q20 Abs % Bias",
        "obj3": "Release q80 Abs % Bias",
        "obj4": "Storage NSE",
}

# Used to filter pareto front
# obj : (min, max)
OBJ_FILTER_BOUNDS = {
    "Release NSE": (-3, 1.0),
    "Release q20 Abs % Bias": (0, 70.0),
    "Release q80 Abs % Bias": (0, 70.0),
    "Storage NSE": (-5, 1.0),
}

### Reservoirs ###############
reservoir_options = [
    'beltzvilleCombined',
    'fewalter',
    'prompton',
    'blueMarsh', # keep if/when ready 
]

### Polcy Settings ###############

policy_type_options = [
    "STARFIT",
    "RBF",
    "PWL",
]

## RBF
n_rbfs = 2              # Number of radial basis functions (RBFs) used in the policy
n_rbf_inputs = 3         # Number of input variables (inflow, storage, day_of_year)
n_rbf_params = n_rbfs * (2 * n_rbf_inputs + 1)
rbf_param_bounds = [[0.0, 1.0]] * n_rbf_params


## STARFIT
n_starfit_params = 17         # Number of parameters in STARFIT policy
# param order = [ NORhi_mu, NORhi_min, NORhi_max, NORhi_alpha, NORhi_beta,
#                  NORlo_mu, NORlo_min, NORlo_max, NORlo_alpha, NORlo_beta,
#                  Release_alpha1, Release_alpha2, Release_beta1, Release_beta2,
#                  Release_c, Release_p1, Release_p2]
n_starfit_inputs = 3         # Number of input variables (inflow, storage, week_of_year)

starfit_param_bounds = [
    [0.0, 100.0],         # NORhi_mu
    [0, 79.24],         # NORhi_min
    [0.07, 100],        # NORhi_max
    [-10.95, 79.63],      # NORhi_alpha
    [-44.29, 5.21],       # NORhi_beta

    [0.0, 100],         # NORlo_mu
    [0.0, 100],         # NORlo_min
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

## Piecewise Linear
n_segments = 3         # linear segments 
n_pwl_inputs = 3         # Number of input variables (inflow, storage, week_of_year)
n_pwl_params = (2 * n_segments - 1) * n_pwl_inputs   # n_params =  (2 * n_segments - 1) * n_predictors

# param order = [segment_breakpoints, slopes] 
# Segment breakpoints (x_i) in [0.0, 1.0]
# Segment slopes (θ_i) in [0.0, π/3]
single_input_pwl_param_bounds = [
    [[i/(n_segments-1), (i+1)/(n_segments-1)] for i in range(n_segments-1)] +
    [[-np.pi/2, np.pi/2]] * n_segments  
][0]

# repeat parameter bounds for each input
pwl_param_bounds = []
for _ in range(n_pwl_inputs):
    pwl_param_bounds += single_input_pwl_param_bounds


## Dictionaries of configurations
policy_n_params = {
    "STARFIT": n_starfit_params,
    "RBF": n_rbf_params,                  
    "PWL": n_pwl_params,  
}

policy_param_bounds = {
    "STARFIT": starfit_param_bounds,
    "RBF": rbf_param_bounds,
    "PWL": pwl_param_bounds,
}


#### RESERVOIR CONSTRAINTS ##############

# --- Single source of truth (all units = MG or MGD) ---

# Storage capacities (MG)
# NOTE: For Beltzville, your OBS storage max is 17,736 MG while you currently use 13,500 MG (crest).
# If you want to model up to observed operating range, bump cap to >= 17,736.
# I rounded to 18,000 to keep it simple.
reservoir_capacity = {
    "prompton": 27956.02,
    "beltzvilleCombined": 18000.0,   # was 13500; >= OBS max 17736.09
    "fewalter": 35800.0,
    "blueMarsh": 42320.35,
}

LOW_STORAGE_FRACTION = 0.05

# Inflow bounds used for normalization (MGD)
inflow_bounds_by_reservoir = {
    "prompton":           {"I_min": 0.0, "I_max": 7500.00},   # I_max = 1740.00 × 1.5 = 2610.00
    "beltzvilleCombined": {"I_min": 0.0, "I_max": 3002.50},   # I_max = 1440.00 × 1.5 = 2160.00
    "fewalter":           {"I_min": 0.0, "I_max": 20000.00},  # I_max = 7690.00 × 1.5 = 11535.00
    "blueMarsh": {"I_min": 0.0, "I_max": 7500.00},  # keep if/when ready
}

# Conservation minimums (MGD) from DRBC Water Code
drbc_conservation_releases = {
    "blueMarsh": 50 * cfs_to_mgd,
    "beltzvilleCombined": 35 * cfs_to_mgd,  # ~22.61 MGD
    "fewalter": 50 * cfs_to_mgd,            # ~32.30 MGD
}

# Release maxima (MGD) updated from your OBS maxima
release_max_by_reservoir = {
    "prompton":           3000.00,  # R_max = 1740.00 × 1.5 = 2610.00
    "beltzvilleCombined": 3000.00,  # R_max = 1440.00 × 1.5 = 2160.00
    "fewalter":           11535.00,  # R_max = 7690.00 × 1.5 = 11535.00
    "blueMarsh":          7500.00,
}

# Optional: if a reservoir isn’t in DRBC table, you can define its min here.
# For the three you’re running, DRBC mins cover beltzvilleCombined and fewalter;
# for promton we’ll use the small observed minimum you reported (~5.75 MGD).
release_min_by_reservoir = {
    "prompton": 5.75,  # from your CTX print
    # beltzvilleCombined, fewalter come from drbc_conservation_releases
}

# --- Build BASE_POLICY_CONTEXT directly from the dicts above (no drift!) ---

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
        "low_storage_threshold": LOW_STORAGE_FRACTION * _icap(name),
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
        S_low = max(0.0, LOW_STORAGE_FRACTION * S_cap) if capacity_override is not None else S_low

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

    # optional: quick length checks to fail fast (kept permissive if you change bounds later)
    from pywrdrb.release_policies.config import policy_n_params
    expected = policy_n_params.get(policy_type)
    if expected is not None and len(vec) != expected:
        raise ValueError(f"{policy_type} expects {expected} params; got {len(vec)}.")
    return vec