"""
Tests for the canonical scalar release-step helper.

``compute_starfit_release_step`` is the documented source of truth for the
STARFIT release math (including the DRB modification that R_min is enforced
in all storage conditions). The online ``STARFITReservoirRelease.value`` and
the offline ``STARFITOfflineSimulator.simulate_reservoir`` both inline this
body for speed. These tests verify:

  * the three target-release branches (in-NOR, above-NOR, below-NOR);
  * the available-water constraint actually clamps;
  * the helper produces non-negative output even with degenerate inputs;
  * the offline simulator produces bit-identical numbers to a reference
    computation that calls the helper directly, proving the inlined copies
    have not drifted from the canonical equation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pywrdrb.parameters.starfit import compute_starfit_release_step
from pywrdrb.path_manager import get_pn_object
from pywrdrb.pre import STARFITOfflineSimulator
from pywrdrb.utils.lists import starfit_reservoir_list


def _baseline_kwargs():
    """A plausible set of inputs we can perturb to hit each target branch."""
    return dict(
        I_t=50.0, S_t=400.0,
        S_cap=1000.0, I_bar=40.0,
        R_min=5.0, R_max=200.0,
        Release_c=0.0, Release_p1=0.1, Release_p2=0.1,
        harmonic_release=0.0,
        NORhi=0.6, NORlo=0.3,
        inv_S_cap=1.0 / 1000.0, inv_I_bar=1.0 / 40.0,
        linear_below_NOR=False,
    )


def test_in_nor_branch_uses_seasonal_target():
    """S_hat in [NORlo, NORhi] picks the I_bar*(seasonal+epsilon+1) target,
    capped at R_max and bounded by available water."""
    kw = _baseline_kwargs()
    # S_t/S_cap = 0.4 — comfortably inside [0.3, 0.6].
    rel = compute_starfit_release_step(**kw)
    # Target = I_bar * (0 + epsilon + 1) where epsilon = 0 + 0.1*A + 0.1*I_hat
    # A = (0.4 - 0.3)/0.6 = 1/6; I_hat = (50-40)/40 = 0.25; eps = 1/60 + 0.025
    # Mirror the helper's exact multiplication order to avoid FP rounding diffs.
    A = (0.4 - kw["NORlo"]) / kw["NORhi"]
    I_hat = (kw["I_t"] - kw["I_bar"]) * kw["inv_I_bar"]
    epsilon = kw["Release_c"] + kw["Release_p1"] * A + kw["Release_p2"] * I_hat
    expected_target = min(
        kw["I_bar"] * (kw["harmonic_release"] + epsilon + 1), kw["R_max"]
    )
    expected = min(expected_target, kw["I_t"] + kw["S_t"])
    assert rel == pytest.approx(expected, rel=1e-15)


def test_above_nor_branch_releases_excess_storage():
    """S_hat > NORhi triggers the spill-style release; the result is bounded
    by R_max and never negative."""
    kw = _baseline_kwargs()
    kw["S_t"] = 900.0  # S_hat = 0.9 > 0.6 (NORhi)
    rel = compute_starfit_release_step(**kw)
    # target = (S_cap*(S_hat-NORhi) + I_t*7)/7 = (1000*0.3 + 50*7)/7 = (300+350)/7 = 92.857...
    expected_target = (kw["S_cap"] * (0.9 - kw["NORhi"]) + kw["I_t"] * 7) / 7
    assert rel == min(expected_target, kw["R_max"])


def test_below_nor_clamps_to_R_min_when_linear_disabled():
    """The default DRB STARFIT setup has linear_below_NOR=False; below NOR
    the release is exactly R_min (subject to available-water clamp)."""
    kw = _baseline_kwargs()
    kw["S_t"] = 100.0  # S_hat = 0.1 < 0.3 (NORlo)
    rel = compute_starfit_release_step(**kw)
    assert rel == kw["R_min"]  # R_min < I_t + S_t so no available-water clamp


def test_available_water_constraint_clamps_target():
    """If target_release > I_t + S_t the helper must clamp to available water."""
    kw = _baseline_kwargs()
    kw["I_t"] = 0.0
    kw["S_t"] = 1.0  # available_water = 1.0
    kw["R_min"] = 50.0  # target wants 50 but only 1 available
    kw["S_cap"] = 1000.0
    rel = compute_starfit_release_step(**kw)
    assert rel == 1.0  # available_water


def test_returns_non_negative_for_extreme_inputs():
    """min_required = available_water - S_cap can go negative for low storage;
    the final max(0.0, ...) guarantees a non-negative release."""
    kw = _baseline_kwargs()
    kw["I_t"] = 0.0
    kw["S_t"] = 0.0
    kw["S_cap"] = 1000.0
    kw["R_min"] = 0.0
    rel = compute_starfit_release_step(**kw)
    assert rel >= 0.0


def test_offline_simulator_matches_helper_release_math():
    """STARFITOfflineSimulator.simulate_reservoir inlines the same release
    math as compute_starfit_release_step, then applies the model-matching
    consumption/net-inflow storage balance. A hand-rolled reference that
    calls the helper for the release step and replicates the storage
    balance must reproduce the simulator bit-identically. This guards
    against the two implementations drifting apart."""
    sim = STARFITOfflineSimulator(initial_volume_frac=0.8)
    sim.load_parameters()

    # 30 days of fabricated inflows for a representative reservoir.
    n_days = 30
    rng = np.random.default_rng(seed=42)
    inflows = rng.uniform(20, 100, size=n_days).astype(np.float64)
    day_of_year = np.arange(1, n_days + 1, dtype=np.int64)

    reservoir = "wallenpaupack"
    actual_releases, actual_storage = sim.simulate_reservoir(
        reservoir, inflows, day_of_year,
    )

    # Reference: build params and the seasonal arrays the same way, then
    # walk the loop calling the helper for the release step and mirroring
    # the simulator's consumption timing (CU_ratio * withdrawal_{t-1},
    # withdrawal limited by the day's inflow) and net-inflow release bound.
    params = sim._get_reservoir_params(reservoir)
    harmonic, norhi, norlo = sim._precompute_seasonal_arrays(params)
    wd, cu_ratio = sim._get_withdrawal_params(reservoir)
    inv_S_cap = 1.0 / params["S_cap"]
    inv_I_bar = 1.0 / params["I_bar"]

    expected_releases = np.empty(n_days)
    expected_storage = np.empty(n_days + 1)
    expected_storage[0] = params["S_cap"] * sim.initial_volume_frac
    withdrawal_prev = 0.0
    for t in range(n_days):
        I_t = inflows[t]
        S_t = expected_storage[t]
        day_idx = day_of_year[t] - 1
        rel = compute_starfit_release_step(
            I_t=I_t, S_t=S_t,
            S_cap=params["S_cap"], I_bar=params["I_bar"],
            R_min=params["R_min"], R_max=params["R_max"],
            Release_c=params["Release_c"],
            Release_p1=params["Release_p1"],
            Release_p2=params["Release_p2"],
            harmonic_release=harmonic[day_idx],
            NORhi=norhi[day_idx], NORlo=norlo[day_idx],
            inv_S_cap=inv_S_cap, inv_I_bar=inv_I_bar,
            linear_below_NOR=False,
        )
        withdrawal_t = min(wd, I_t)
        consumption_t = min(cu_ratio * withdrawal_prev, withdrawal_t)
        withdrawal_prev = withdrawal_t
        net_inflow = I_t - consumption_t
        rel = min(rel, S_t + net_inflow)
        expected_releases[t] = rel
        expected_storage[t + 1] = S_t + net_inflow - rel

    # Bit-identical: the helper and the simulator share one release equation.
    np.testing.assert_array_equal(actual_releases, expected_releases)
    np.testing.assert_array_equal(actual_storage, expected_storage)


def test_simulate_all_unchanged_against_real_inflows():
    """Run the offline simulator against a real DRB inflow CSV and verify
    the output is non-negative and finite — a smoke test that the refactored
    loop still runs end-to-end against canonical data."""
    pn = get_pn_object()
    inflow_csv = (
        pn.sc.get("flows/nhmv10_withObsScaled") / "catchment_inflow_mgd.csv"
    )
    inflows_df = pd.read_csv(str(inflow_csv), index_col=0, parse_dates=True)
    inflows_df.index = pd.DatetimeIndex(inflows_df.index)
    inflows_df = inflows_df.iloc[:60]  # keep the test fast

    sim = STARFITOfflineSimulator(initial_volume_frac=0.8)
    sim.load_parameters()
    df = sim.simulate_all(inflows_df, reservoir_list=starfit_reservoir_list)
    assert df.shape == (60, len(starfit_reservoir_list))
    assert np.all(np.isfinite(df.values))
    assert (df.values >= 0).all()
