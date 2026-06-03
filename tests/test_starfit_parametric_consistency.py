import math

import pandas as pd

from pywrdrb.path_manager import get_pn_object
from pywrdrb.release_policies import STARFIT
from pywrdrb.release_policies.config import STARFIT_PARAM_NAMES, get_policy_context


def _legacy_starfit_release(row, ctx, storage_mg, inflow_mgd, day_of_year):
    """Reference STARFIT release math from legacy STARFITReservoirRelease implementation."""
    s_cap = float(ctx["storage_capacity"])
    r_min = float(ctx["release_min"])
    r_max = float(ctx["release_max"])
    i_bar = float(row["Adjusted_MEANFLOW_MGD"])

    # state normalization
    s_hat = float(storage_mg) / s_cap
    i_hat = (float(inflow_mgd) - i_bar) / i_bar

    c = math.pi * float(day_of_year) / 365.0
    sin_2c, sin_4c = math.sin(2.0 * c), math.sin(4.0 * c)
    cos_2c, cos_4c = math.cos(2.0 * c), math.cos(4.0 * c)

    nor_hi = (
        float(row["NORhi_mu"])
        + float(row["NORhi_alpha"]) * sin_2c
        + float(row["NORhi_beta"]) * cos_2c
    )
    nor_hi = min(max(nor_hi, float(row["NORhi_min"])), float(row["NORhi_max"])) / 100.0

    nor_lo = (
        float(row["NORlo_mu"])
        + float(row["NORlo_alpha"]) * sin_2c
        + float(row["NORlo_beta"]) * cos_2c
    )
    nor_lo = min(max(nor_lo, float(row["NORlo_min"])), float(row["NORlo_max"])) / 100.0

    harmonic = (
        float(row["Release_alpha1"]) * sin_2c
        + float(row["Release_alpha2"]) * sin_4c
        + float(row["Release_beta1"]) * cos_2c
        + float(row["Release_beta2"]) * cos_4c
    )
    a_t = (s_hat - nor_lo) / (nor_hi + 1e-6)
    epsilon = float(row["Release_c"]) + float(row["Release_p1"]) * a_t + float(row["Release_p2"]) * i_hat

    if nor_lo <= s_hat <= nor_hi:
        target = min(i_bar * (harmonic + epsilon + 1.0), r_max)
    elif s_hat > nor_hi:
        target = min((s_cap * (s_hat - nor_hi) + inflow_mgd * 7.0) / 7.0, r_max)
    else:
        target = max((i_bar * (harmonic + epsilon + 1.0)) * (s_hat / nor_lo), r_min)

    available_water = float(inflow_mgd) + float(storage_mg)
    min_required = available_water - s_cap
    release = max(min(target, available_water), min_required)
    return max(r_min, release)


def test_starfit_parametric_inline_matches_legacy_math_for_selected_reservoirs():
    pn = get_pn_object()
    istarf_df = pd.read_csv(pn.operational_constants.get_str("istarf_conus.csv"))
    inline_df = pd.read_csv("tests/data/starfit_parametric_inline_test_params.csv")

    reservoirs = ["fewalter", "blueMarsh"]
    for reservoir in reservoirs:
        istarf_row = istarf_df[
            (istarf_df["reservoir"] == reservoir) & (istarf_df["policy_id"] == "default")
        ].iloc[0]
        inline_row = inline_df[
            (inline_df["reservoir"] == reservoir) & (inline_df["policy_id"] == "default")
        ].iloc[0]

        # Explicitly verify inline CSV injects the same STARFIT parameter set as ISTARF defaults.
        inline_params = [float(inline_row[name]) for name in STARFIT_PARAM_NAMES]
        istarf_params = [float(istarf_row[name]) for name in STARFIT_PARAM_NAMES]
        for injected, expected in zip(inline_params, istarf_params):
            assert abs(injected - expected) < 1e-10

        ctx = get_policy_context(reservoir)

        # Parametric-style inline injection path:
        # policy_params assigned directly, then parsed (no row-based assignment).
        policy = STARFIT(policy_params=None, reservoir_name=reservoir)
        policy.set_context(**ctx)
        policy.policy_params = inline_params
        policy.parse_policy_params()
        policy.linear_below_NOR = True

        test_points = [
            (0.15 * ctx["storage_capacity"], 120.0, 30.0),
            (0.45 * ctx["storage_capacity"], 300.0, 120.0),
            (0.75 * ctx["storage_capacity"], 700.0, 210.0),
            (0.95 * ctx["storage_capacity"], 1200.0, 330.0),
        ]

        for storage_mg, inflow_mgd, day in test_points:
            expected = _legacy_starfit_release(istarf_row, ctx, storage_mg, inflow_mgd, day)
            actual = policy.get_release(storage_mg, inflow_mgd, day)
            assert abs(actual - expected) < 1e-6
