"""
Helpers for manual STARFIT parameter tuning of lower basin reservoirs.

Workflow: load default params -> apply overrides -> write custom CSV ->
static diagnostics -> offline quick sim -> full pywrdrb run -> compare vs obs.

Notes
-----
- Reservoir rows in the parameter CSV use the "modified_" prefix for
  blueMarsh, beltzvilleCombined, and fewalter; helpers map names transparently.
- R_max/R_min for the DRBC lower basin reservoirs (blueMarsh,
  beltzvilleCombined, fewalter, nockamixon) are hard-coded in
  pywrdrb.parameters.lower_basin_ffmp (max_discharges, conservation_releases);
  CSV Release_max/Release_min have no effect there (prompton's do apply).
"""
import os
import warnings

import numpy as np
import pandas as pd

import pywrdrb
from pywrdrb.path_manager import get_pn_object
from pywrdrb.utils.lists import modified_starfit_reservoir_list
from pywrdrb.parameters.lower_basin_ffmp import conservation_releases, max_discharges
from pywrdrb.pre.generate_presimulated_releases import STARFITOfflineSimulator

pn = get_pn_object()

RESERVOIRS = ["fewalter", "beltzvilleCombined", "blueMarsh", "prompton"]

# Reservoirs whose R_max/R_min come from DRBC dicts, not the CSV
DRBC_FIXED_RELEASE_BOUNDS = sorted(set(max_discharges) | set(conservation_releases))

TUNABLE_COLS = [
    "NORhi_mu", "NORhi_alpha", "NORhi_beta", "NORhi_min", "NORhi_max",
    "NORlo_mu", "NORlo_alpha", "NORlo_beta", "NORlo_min", "NORlo_max",
    "Release_alpha1", "Release_alpha2", "Release_beta1", "Release_beta2",
    "Release_c", "Release_p1", "Release_p2",
    "Release_max", "Release_min",
    "Adjusted_CAP_MG", "Adjusted_MEANFLOW_MGD",
]


# --------------------------------------------------------------------------
# Parameter table handling
# --------------------------------------------------------------------------
def get_starfit_row_name(reservoir):
    """CSV row name for a reservoir ("modified_" prefix where applicable)."""
    if reservoir in modified_starfit_reservoir_list:
        return f"modified_{reservoir}"
    return reservoir


def load_default_params():
    """Load the full default istarf_conus.csv, indexed by reservoir row name."""
    return pd.read_csv(
        pn.operational_constants.get_str("istarf_conus.csv"), index_col=0
    )


def get_reservoir_params(params_df, reservoir):
    """Return the parameter row (pd.Series) for a reservoir."""
    return params_df.loc[get_starfit_row_name(reservoir)]


def apply_overrides(base_df, overrides):
    """
    Return a copy of base_df with parameter overrides applied.

    Parameters
    ----------
    base_df : pd.DataFrame
        Full parameter table (e.g. from load_default_params()).
    overrides : dict
        {reservoir_name: {param_column: value}}. Reservoir names are plain
        pywrdrb names; mapping to "modified_" rows is handled here.
    """
    df = base_df.copy(deep=True)
    for reservoir, mods in overrides.items():
        row = get_starfit_row_name(reservoir)
        if row not in df.index:
            raise KeyError(f"Row '{row}' not found in parameter table.")
        for col, value in mods.items():
            if col not in TUNABLE_COLS:
                raise KeyError(
                    f"'{col}' is not a tunable STARFIT column. "
                    f"Options: {TUNABLE_COLS}"
                )
            if (
                col in ("Release_max", "Release_min")
                and reservoir in DRBC_FIXED_RELEASE_BOUNDS
            ):
                warnings.warn(
                    f"{col} for '{reservoir}' is fixed by DRBC rules "
                    "(lower_basin_ffmp.py) - this override has no effect."
                )
            df.loc[row, col] = value
    return df


def write_custom_csv(params_df, output_dir, tag="tuned"):
    """Write a full parameter table to output_dir; returns the absolute path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.abspath(os.path.join(output_dir, f"custom_starfit_{tag}.csv"))
    out = params_df.copy()
    out.index.name = "reservoir"
    out.to_csv(path)
    return path


# --------------------------------------------------------------------------
# Static STARFIT curves (replicates parameters/starfit.py seasonal lookups)
# --------------------------------------------------------------------------
def compute_nor_curves(params):
    """
    Compute NOR bounds vs day-of-year for a parameter row.

    Returns
    -------
    pd.DataFrame
        Indexed by day-of-year (1-366) with columns NORhi, NORlo (fractions).
    """
    doy = np.arange(1, 367)
    c = (np.pi / 365) * doy
    sin_2c, cos_2c = np.sin(2 * c), np.cos(2 * c)

    norhi = np.clip(
        params["NORhi_mu"] + params["NORhi_alpha"] * sin_2c + params["NORhi_beta"] * cos_2c,
        params["NORhi_min"],
        params["NORhi_max"],
    ) / 100
    norlo = np.clip(
        params["NORlo_mu"] + params["NORlo_alpha"] * sin_2c + params["NORlo_beta"] * cos_2c,
        params["NORlo_min"],
        params["NORlo_max"],
    ) / 100
    return pd.DataFrame({"NORhi": norhi, "NORlo": norlo}, index=pd.Index(doy, name="doy"))


def compute_harmonic_release(params):
    """
    Seasonal (harmonic) release target vs day-of-year, in MGD:
    I_bar * (harmonic + Release_c + 1), i.e. the in-NOR release under
    average inflow (I_hat=0) and A_t=0 conditions.

    Returns
    -------
    pd.Series indexed by day-of-year (1-366).
    """
    doy = np.arange(1, 367)
    c = (np.pi / 365) * doy
    harmonic = (
        params["Release_alpha1"] * np.sin(2 * c)
        + params["Release_alpha2"] * np.sin(4 * c)
        + params["Release_beta1"] * np.cos(2 * c)
        + params["Release_beta2"] * np.cos(4 * c)
    )
    release = params["Adjusted_MEANFLOW_MGD"] * (harmonic + params["Release_c"] + 1)
    return pd.Series(release, index=pd.Index(doy, name="doy"), name="release_mgd")


def get_effective_rmin_rmax(params, reservoir):
    """
    Effective (R_min, R_max) in MGD, honoring the DRBC hard-coded overrides.
    """
    I_bar = params["Adjusted_MEANFLOW_MGD"]
    if reservoir in max_discharges:
        R_max = max_discharges[reservoir]
    else:
        R_max = (params["Release_max"] + 1) * I_bar
    if reservoir in conservation_releases:
        R_min = conservation_releases[reservoir]
    else:
        R_min = (params["Release_min"] + 1) * I_bar
    return R_min, R_max


# --------------------------------------------------------------------------
# Data loading (all timeseries via pywrdrb.Data)
# --------------------------------------------------------------------------
def load_observations():
    """Load observed storage/flow data. Returns a pywrdrb.Data object."""
    data = pywrdrb.Data(print_status=False)
    data.load_observations(
        results_sets=["res_storage", "reservoir_downstream_gage", "major_flow"]
    )
    return data


def get_obs_storage(obs_data, reservoir, start=None, end=None):
    """Observed storage (MG) series for a reservoir."""
    df = obs_data.res_storage["obs"][0]
    s = df[reservoir].copy()
    s.index = pd.to_datetime(s.index)
    return s.loc[start:end].dropna()


def get_obs_downstream_flow(obs_data, reservoir, start=None, end=None):
    """Observed downstream gage flow (MGD); None if no gage (e.g. prompton)."""
    df = obs_data.reservoir_downstream_gage["obs"][0]
    if reservoir not in df.columns:
        return None
    s = df[reservoir].copy()
    s.index = pd.to_datetime(s.index)
    return s.loc[start:end].dropna()


def load_sim_results(output_filenames):
    """
    Load pywrdrb output HDF5 files. Returns a pywrdrb.Data object with
    res_storage, res_release, reservoir_downstream_gage, and all (raw keys,
    including starfit_release_{name} parameter traces).
    """
    data = pywrdrb.Data(print_status=False)
    data.load_output(
        output_filenames=[str(f) for f in output_filenames],
        results_sets=["res_storage", "res_release", "reservoir_downstream_gage", "all"],
    )
    return data


def get_sim_storage(sim_data, run_label, reservoir, start=None, end=None):
    """Simulated storage (MG) series for a run/reservoir."""
    s = sim_data.res_storage[run_label][0][reservoir].copy()
    s.index = pd.to_datetime(s.index)
    return s.loc[start:end]


def get_sim_release(sim_data, run_label, reservoir, start=None, end=None):
    """Simulated total release (outflow + spill, MGD) for a run/reservoir."""
    s = sim_data.res_release[run_label][0][reservoir].copy()
    s.index = pd.to_datetime(s.index)
    return s.loc[start:end]


def get_starfit_release_trace(sim_data, run_label, reservoir, start=None, end=None):
    """Recorded starfit_release_{name} parameter trace (MGD) for a run."""
    df = sim_data.all[run_label][0]
    col = f"starfit_release_{reservoir}"
    if col not in df.columns:
        return None
    s = df[col].copy()
    s.index = pd.to_datetime(s.index)
    return s.loc[start:end]


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def run_offline_sim(inflow_type, start, end, csv_path=None,
                    reservoirs=RESERVOIRS, initial_volume_frac=0.8):
    """
    Fast offline STARFIT simulation (seconds; no Pywr run).

    Pure STARFIT behavior only: beltzvilleCombined and blueMarsh FFMP /
    Trenton-contribution logic is NOT represented - treat as a preview.

    Returns
    -------
    dict of {reservoir: pd.DataFrame with columns [release, storage]}
        storage is end-of-day (MG), release in MGD.
    """
    sim = STARFITOfflineSimulator(
        initial_volume_frac=initial_volume_frac,
        starfit_params_filename=csv_path,
    )
    inflow_file = pn.sc.get(f"flows/{inflow_type}") / "catchment_inflow_mgd.csv"
    inflows = pd.read_csv(str(inflow_file), index_col=0, parse_dates=True)
    inflows = inflows.loc[start:end]
    doy = inflows.index.dayofyear.values

    results = {}
    for res in reservoirs:
        releases, storage = sim.simulate_reservoir(
            res, inflows[res].values.astype(float), doy
        )
        results[res] = pd.DataFrame(
            {"release": releases, "storage": storage[1:]}, index=inflows.index
        )
    return results


def run_pywrdrb_model(run_label, inflow_type, start, end, output_dir,
                      csv_path=None, flow_prediction_mode="perfect_foresight"):
    """
    Build and run a pywrdrb model; returns the output HDF5 path.

    Always runs fresh, overwriting any existing {run_label}.json/.hdf5.
    flow_prediction_mode sets how the model forecasts Montague/Trenton flows
    for the FFMP release logic; "perfect_foresight" uses forecasts built from
    the actual future inflows (with STARFIT releases pre-simulated using the
    default parameters).
    """
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{run_label}.json")
    h5_path = os.path.join(output_dir, f"{run_label}.hdf5")

    options = {"flow_prediction_mode": flow_prediction_mode}
    if csv_path is not None:
        options["starfit_params_filename"] = str(csv_path)
    mb = pywrdrb.ModelBuilder(
        inflow_type=inflow_type, start_date=start, end_date=end, options=options
    )
    mb.make_model()
    mb.write_model(json_path)

    model = pywrdrb.Model.load(json_path)
    pywrdrb.OutputRecorder(
        model=model,
        output_filename=h5_path,
        parameters=[p for p in model.parameters if p.name],
    )
    model.run()
    print(f"Run complete: {h5_path}")
    return h5_path
