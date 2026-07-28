"""
Fast offline smoke tests for STARFITOfflineSimulator (new in v2.2).

Runs a short synthetic-inflow simulation for one reservoir, and verifies that
a custom parameter CSV (via starfit_params_filename) changes the result.
"""
import numpy as np
import pandas as pd
import pytest

from pywrdrb.pre import STARFITOfflineSimulator
from pywrdrb.path_manager import get_pn_object

RESERVOIR = "prompton"
N_DAYS = 90


def _synthetic_inputs():
    inflows = np.full(N_DAYS, 60.0)  # ~ prompton mean inflow, MGD
    day_of_year = (np.arange(N_DAYS) % 365) + 1
    return inflows, day_of_year


def test_simulate_reservoir_shapes_and_bounds():
    # start inside the (low) flood-control NOR so release rules are active
    sim = STARFITOfflineSimulator(initial_volume_frac=0.05)
    inflows, doy = _synthetic_inputs()
    releases, storage = sim.simulate_reservoir(RESERVOIR, inflows, doy)

    assert releases.shape == (N_DAYS,)
    assert storage.shape == (N_DAYS + 1,)
    assert np.all(np.isfinite(releases)) and np.all(np.isfinite(storage))
    assert np.all(releases >= 0.0)

    cap = sim._get_reservoir_params(RESERVOIR)["S_cap"]
    assert np.all(storage >= 0.0)
    assert np.all(storage <= cap + 1e-6)


def test_custom_params_filename_changes_releases(tmp_path):
    pn = get_pn_object()
    df = pd.read_csv(
        pn.operational_constants.get_str("istarf_conus.csv"), index_col=0
    )
    df.loc[RESERVOIR, "Release_c"] = df.loc[RESERVOIR, "Release_c"] + 0.25
    df.index.name = "reservoir"
    csv_path = tmp_path / "custom_starfit.csv"
    df.to_csv(csv_path)

    inflows, doy = _synthetic_inputs()
    default_releases, _ = STARFITOfflineSimulator(
        initial_volume_frac=0.05
    ).simulate_reservoir(RESERVOIR, inflows, doy)
    custom_releases, _ = STARFITOfflineSimulator(
        initial_volume_frac=0.05, starfit_params_filename=str(csv_path)
    ).simulate_reservoir(RESERVOIR, inflows, doy)

    assert not np.allclose(default_releases, custom_releases), (
        "Custom starfit_params_filename had no effect on simulated releases"
    )
