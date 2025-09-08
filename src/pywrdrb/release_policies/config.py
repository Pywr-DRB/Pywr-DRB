"""
Contains configuration specifications for the project.
"""
import os
import numpy as np
from methods.utils.release_constraints import get_release_minmax_release_dict


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
cfs_to_mgd = 0.645932368556
ACRE_FEET_TO_MG = 0.325851  # Acre-feet to million gallons


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
] #blueMarsh not ready 


### Polcy Settings ###############

policy_type_options = [
    "STARFIT",
    "RBF",
    "PiecewiseLinear",
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


starfit_param_bounds_old = [
    [0.0, 100.0],         # NORhi_mu
    [0.0, 79.24],         # NORhi_min
    [0.07, 100.0],        # NORhi_max
    [-2, 2],              # NORhi_alpha
    [-4, 5.21],       # NORhi_beta

    [0.0, 40],         # NORlo_mu
    [0.0, 40],         # NORlo_min
    [0, 40],        # NORlo_max
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

#this is a larger subset to see if I get better solutions
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
    "PiecewiseLinear": n_pwl_params,  
}

policy_param_bounds = {
    "STARFIT": starfit_param_bounds,
    "RBF": rbf_param_bounds,
    "PiecewiseLinear": pwl_param_bounds,
}


#### RESERVOIR CONSTRAINTS ##############

# Reservoir capacities (in Million Gallons - MG) from ISARF CONUS dataset
reservoir_capacity = {
    "prompton": 27956.02,              # 27,956.02 MG
    "beltzvilleCombined": 13500,     # 13,500 MG (approximate to spillway crest)
    "fewalter": 35800,               # 35,800 MG
    "blueMarsh": 42320.35              # 42,320.35 MG
}

# Conservation releases at lower reservoirs
# Specified in the DRBC Water Code Table 4
drbc_conservation_releases = {
    "blueMarsh": 50 * cfs_to_mgd,
    "beltzvilleCombined": 35 * cfs_to_mgd,
    "fewalter": 50 * cfs_to_mgd,
}

# Observed min/max releases
obs_release_min, obs_release_max = get_release_minmax_release_dict()


# Assign min/max releases for each reservoir
reservoir_min_release = {}
reservoir_max_release = {}
for r in reservoir_options:
    reservoir_max_release[r] = obs_release_max[r]
    
    # Set min (conservation) releases from WaterCode
    if r in drbc_conservation_releases:
        reservoir_min_release[r] = drbc_conservation_releases[r]
    else:
        reservoir_min_release[r] = obs_release_min[r]
    


inflow_bounds_by_reservoir = {
    # "reservoir_name": {"I_min": <float>, "I_max": <float>}
    "blueMarsh": {"I_min": 0.0, "I_max": 692.06},
    "beltzvilleCombined": {"I_min": 0.0, "I_max": 22300.0},
    "prompton": {"I_min": 0.0, "I_max": 900.585},
    "fewalter": {"I_min": 0.0, "I_max": 3652.15 }
}

def get_policy_context(
    reservoir_name: str,
    *,
    release_min_override: float | None = None,
    release_max_override: float | None = None,
    capacity_override: float | None = None,
    inflow_bounds_override: tuple[float, float] | None = None,
) -> dict:
    """
    Assemble kwargs for AbstractPolicy.set_context(...) for a given reservoir.

    Returns a dict with:
      - release_min, release_max, storage_capacity
      - x_min = (0.0, I_min, 1.0)
      - x_max = (S_cap, I_max, 366.0)
    """

    # Base values from config tables
    try:
        S_cap_cfg = float(reservoir_capacity[reservoir_name])
        R_min_cfg = float(reservoir_min_release[reservoir_name])
        R_max_cfg = float(reservoir_max_release[reservoir_name])
        I_min_cfg = float(inflow_bounds_by_reservoir[reservoir_name]["I_min"])
        I_max_cfg = float(inflow_bounds_by_reservoir[reservoir_name]["I_max"])
    except KeyError as e:
        raise KeyError(f"Missing config for {reservoir_name}: {e}")

    # Apply optional overrides (e.g., from CLI or CSV)
    S_cap = float(capacity_override) if capacity_override is not None else S_cap_cfg
    R_min = float(release_min_override) if release_min_override is not None else R_min_cfg
    R_max = float(release_max_override) if release_max_override is not None else R_max_cfg
    if inflow_bounds_override is not None:
        I_min, I_max = map(float, inflow_bounds_override)
    else:
        I_min, I_max = I_min_cfg, I_max_cfg

    # Basic sanity
    if not (I_max > I_min):
        raise ValueError(f"{reservoir_name}: I_max ({I_max}) must be > I_min ({I_min}).")
    if not (R_max >= R_min >= 0.0):
        raise ValueError(f"{reservoir_name}: release bounds invalid: [{R_min}, {R_max}].")

    return {
        "release_min": R_min,
        "release_max": R_max,
        "storage_capacity": S_cap,
        "x_min": (0.0, I_min, 1.0),
        "x_max": (S_cap, I_max, 366.0),
    }

# Optional: precompute for convenience
POLICY_CONTEXT_BY_RESERVOIR = {r: get_policy_context(r) for r in reservoir_options}
