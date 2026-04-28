import pandas as pd
import pytest

from pywrdrb.parameters.parametric_release import (
    ParametricReservoirRelease,
    _POLICY_FILENAME,
)


def _build_instance(reservoir_name="fewalter", policy_id="candidate_a"):
    # Bypass __init__ because the full constructor depends on a live Pywr model.
    inst = ParametricReservoirRelease.__new__(ParametricReservoirRelease)
    inst.reservoir_name = reservoir_name
    inst.policy_id = policy_id
    return inst


def test_starfit_filename_points_to_starfit_csv():
    assert _POLICY_FILENAME["STARFIT"] == "starfit.csv"


def test_select_row_uses_exact_match_for_standard_dataframe():
    df = pd.DataFrame(
        [
            {"reservoir": "fewalter", "policy_id": "default", "NORhi_mu": 1.0},
            {"reservoir": "fewalter", "policy_id": "candidate_a", "NORhi_mu": 2.0},
        ]
    )
    inst = _build_instance(reservoir_name="fewalter", policy_id="candidate_a")
    row = inst._select_row(df, "dummy.csv")
    assert row["NORhi_mu"] == 2.0


def test_select_row_falls_back_to_default_for_standard_dataframe():
    df = pd.DataFrame(
        [
            {"reservoir": "fewalter", "policy_id": "default", "NORhi_mu": 1.0},
            {"reservoir": "prompton", "policy_id": "candidate_a", "NORhi_mu": 3.0},
        ]
    )
    inst = _build_instance(reservoir_name="fewalter", policy_id="candidate_a")
    row = inst._select_row(df, "dummy.csv")
    assert row["policy_id"] == "default"
    assert row["NORhi_mu"] == 1.0


def test_select_row_raises_when_no_match_or_default():
    df = pd.DataFrame(
        [{"reservoir": "prompton", "policy_id": "candidate_a", "NORhi_mu": 3.0}]
    )
    inst = _build_instance(reservoir_name="fewalter", policy_id="candidate_a")
    with pytest.raises(KeyError):
        inst._select_row(df, "dummy.csv")
