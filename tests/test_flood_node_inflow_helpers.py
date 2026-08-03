"""
Unit tests for the pure helpers in pywrdrb.pre.flood_node_inflows.

These tests verify the per-trace math (drainage-area redistribution, mass
balance, flood-node addition) without any I/O. The helpers are shared between
the single-trace and ensemble preprocessors, so correctness here implies
correctness for both.

The design guarantee under test is STRICT MASS-CONSERVING REDISTRIBUTION: each
flood node takes a fixed drainage-area fraction of its downstream node's
already-marginal inflow, and the same amount is removed from that downstream
node. Net basin flow must be unchanged.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from pywrdrb.pre.flood_node_inflows import (
    DRAINAGE_AREAS,
    DRAINAGE_AREA_SOURCES,
    FLOOD_NODE_IDS,
    _FLOOD_DOWNSTREAM_MAP,
    add_flood_nodes_to_inflows,
    compute_bridgeville_inflow,
    compute_fishs_eddy_inflow,
    compute_hale_eddy_inflow,
    flood_node_inflow_fractions,
    incremental_drainage_areas,
    subtract_flood_inflows_from_downstream,
    validate_drainage_areas,
)


# Drainage areas as published by USGS NWIS / NYCDEP. Hard-coded here so that an
# edit to the source table has to be made deliberately in two places.
EXPECTED_AREAS = {
    "cannonsville": 455.0,
    "pepacton": 372.0,
    "neversink": 92.5,
    "01425000": 456.0,
    "01417000": 372.0,
    "01436000": 92.6,
    "01426500": 595.0,
    "01421000": 784.0,
    "01436690": 171.0,
    "delLordville": 1590.0,
    "delMontague": 3480.0,
}

# The pre-2026-07-31 table, retained so the regression stays named and testable.
HISTORICAL_BAD_AREAS = {
    "01425000": 515.0,
    "01417000": 705.0,
    "01436000": 150.0,
    "01436690": 160.0,
    "delLordville": 1595.0,
}


def _synthetic_inflows():
    """A small MARGINAL inflow frame with all donor nodes the helpers expect."""
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "cannonsville": [100.0, 110.0, 120.0, 130.0],
            "01425000": [10.0, 11.0, 12.0, 13.0],
            "pepacton": [80.0, 85.0, 90.0, 95.0],
            "01417000": [15.0, 16.0, 17.0, 18.0],
            "neversink": [40.0, 42.0, 44.0, 46.0],
            "01436000": [5.0, 6.0, 7.0, 8.0],
            "delLordville": [300.0, 310.0, 320.0, 330.0],
            "delMontague": [200.0, 210.0, 220.0, 230.0],
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# A. Drainage-area table integrity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("node,expected", sorted(EXPECTED_AREAS.items()))
def test_drainage_areas_match_published_values(node, expected):
    assert DRAINAGE_AREAS[node] == pytest.approx(expected)


def test_incremental_drainage_areas():
    inc = incremental_drainage_areas()
    assert inc["01426500"] == pytest.approx(139.0)
    assert inc["01421000"] == pytest.approx(412.0)
    assert inc["01436690"] == pytest.approx(78.4)


def test_drainage_area_closure():
    """Increments + residual exactly tile delLordville's marginal catchment.

    This is an algebraic identity given the definitions, so it documents the
    tiling structure rather than constraining the table; the literal values are
    what this test actually pins.
    """
    inc = incremental_drainage_areas()
    marginal = (
        DRAINAGE_AREAS["delLordville"]
        - DRAINAGE_AREAS["01425000"]
        - DRAINAGE_AREAS["01417000"]
    )
    residual = (
        DRAINAGE_AREAS["delLordville"]
        - DRAINAGE_AREAS["01426500"]
        - DRAINAGE_AREAS["01421000"]
    )
    assert marginal == pytest.approx(762.0)
    assert residual == pytest.approx(211.0)
    assert inc["01426500"] + inc["01421000"] + residual == pytest.approx(marginal)


@pytest.mark.parametrize(
    "gage,reservoir",
    [("01425000", "cannonsville"), ("01417000", "pepacton"), ("01436000", "neversink")],
)
def test_at_dam_gage_areas_match_reservoir_catchments(gage, reservoir):
    """Release gages sit at the dam, so their marginal catchment is ~0 sq mi.

    This is the check that catches an inflated below-dam drainage area, which is
    how the flood-node increments came to be mis-sized.
    """
    rel = abs(DRAINAGE_AREAS[gage] - DRAINAGE_AREAS[reservoir]) / DRAINAGE_AREAS[reservoir]
    assert rel <= 0.10


def test_validate_drainage_areas_accepts_current_table():
    validate_drainage_areas()


def test_validate_drainage_areas_rejects_historical_bad_table():
    """The pre-fix table must be rejected by name."""
    bad = dict(DRAINAGE_AREAS)
    bad.update(HISTORICAL_BAD_AREAS)
    with pytest.raises(ValueError, match="sit at the dam"):
        validate_drainage_areas(bad)


def test_validate_drainage_areas_rejects_negative_lordville_residual():
    """Lordville must be larger than the two flood gauges it contains."""
    bad = dict(DRAINAGE_AREAS)
    bad["delLordville"] = 1300.0  # < 595 + 784, so the residual goes negative
    with pytest.raises(ValueError, match="Lordville residual"):
        validate_drainage_areas(bad)


def test_validate_drainage_areas_rejects_non_conserving_fractions():
    """If a donor's fractions summed to >= 1, redistribution would create water."""
    bad = dict(DRAINAGE_AREAS)
    bad["delLordville"] = 1450.0  # marginal 622 < 139 + 412 + residual demand
    bad["01426500"] = 900.0       # inflates the Hale Eddy increment to 444
    with pytest.raises(ValueError):
        validate_drainage_areas(bad)


def test_every_area_has_a_source():
    assert set(DRAINAGE_AREAS) == set(DRAINAGE_AREA_SOURCES)
    for node, source in DRAINAGE_AREA_SOURCES.items():
        assert source.strip(), f"empty source for {node}"


def test_prompton_is_not_treated_as_upstream_of_lordville():
    """prompton is on the Lackawaxen and enters BELOW Lordville.

    It was previously subtracted when computing the Lordville flood-node
    increments, which was wrong on top of being a double subtraction.
    """
    df = _synthetic_inflows()
    with_prompton = df.copy()
    with_prompton["prompton"] = 999.0
    pd.testing.assert_series_equal(
        compute_hale_eddy_inflow(df),
        compute_hale_eddy_inflow(with_prompton),
        check_names=False,
    )


# ---------------------------------------------------------------------------
# B. Estimator formulas
# ---------------------------------------------------------------------------

def test_hale_eddy_is_da_fraction_of_lordville_marginal():
    df = _synthetic_inflows()
    expected = df["delLordville"] * (139.0 / 762.0)
    pd.testing.assert_series_equal(
        compute_hale_eddy_inflow(df), expected, check_names=False
    )


def test_fishs_eddy_is_da_fraction_of_lordville_marginal():
    df = _synthetic_inflows()
    expected = df["delLordville"] * (412.0 / 762.0)
    pd.testing.assert_series_equal(
        compute_fishs_eddy_inflow(df), expected, check_names=False
    )


def test_bridgeville_is_da_fraction_of_montague_marginal():
    df = _synthetic_inflows()
    expected = df["delMontague"] * flood_node_inflow_fractions()["01436690"]
    pd.testing.assert_series_equal(
        compute_bridgeville_inflow(df), expected, check_names=False
    )


def test_lordville_helpers_ignore_upstream_columns():
    """Direct regression test for the double-subtraction defect.

    ``catchment_inflow_mgd.csv`` is already marginal, so the upstream columns
    must not influence the result at all. The pre-fix implementation read them
    via ``.get(..., 0)`` and returned a very different (near-zero) series.
    """
    df = _synthetic_inflows()
    stripped = df.drop(columns=["cannonsville", "pepacton", "01425000", "01417000"])
    for fn in (compute_hale_eddy_inflow, compute_fishs_eddy_inflow):
        pd.testing.assert_series_equal(fn(df), fn(stripped), check_names=False)


def test_bridgeville_ignores_the_at_dam_gage():
    """01436000 is no longer the donor; scaling it must not change the result."""
    df = _synthetic_inflows()
    inflated = df.copy()
    inflated["01436000"] = inflated["01436000"] * 100.0
    pd.testing.assert_series_equal(
        compute_bridgeville_inflow(df),
        compute_bridgeville_inflow(inflated),
        check_names=False,
    )


@pytest.mark.parametrize(
    "fn,donor",
    [
        (compute_hale_eddy_inflow, "delLordville"),
        (compute_fishs_eddy_inflow, "delLordville"),
        (compute_bridgeville_inflow, "delMontague"),
    ],
)
def test_helpers_raise_when_donor_missing(fn, donor):
    df = _synthetic_inflows().drop(columns=[donor])
    with pytest.raises(KeyError, match=donor):
        fn(df)


# ---------------------------------------------------------------------------
# C. Structural invariants: mass conservation
# ---------------------------------------------------------------------------

def test_no_clamping_and_exact_mass_conservation():
    """The constraint: redistribution must not change net basin flow at all."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-01-01", periods=500, freq="D")
    df = pd.DataFrame(
        {
            "delLordville": rng.gamma(2.0, 400.0, size=len(idx)),
            "delMontague": rng.gamma(2.0, 800.0, size=len(idx)),
            "neversink": rng.gamma(2.0, 90.0, size=len(idx)),
            "01436000": rng.gamma(1.0, 5.0, size=len(idx)),
        },
        index=idx,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any clamp warning fails the test
        out = add_flood_nodes_to_inflows(df)

    fr = flood_node_inflow_fractions()
    lordville_retained = 1.0 - fr["01426500"] - fr["01421000"]
    montague_retained = 1.0 - fr["01436690"]

    pd.testing.assert_series_equal(
        out["delLordville"], df["delLordville"] * lordville_retained, check_names=False
    )
    pd.testing.assert_series_equal(
        out["delMontague"], df["delMontague"] * montague_retained, check_names=False
    )

    # Per-node balance, and the total over every column.
    np.testing.assert_allclose(
        (out["delLordville"] + out["01426500"] + out["01421000"]).values,
        df["delLordville"].values,
        rtol=0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        (out["delMontague"] + out["01436690"]).values,
        df["delMontague"].values,
        rtol=0,
        atol=1e-9,
    )
    assert out.to_numpy().sum() == pytest.approx(df.to_numpy().sum(), rel=1e-12)


def test_increment_positive_wherever_donor_positive():
    """Under the pre-fix math this failed on ~92% of real days."""
    df = _synthetic_inflows()
    out = add_flood_nodes_to_inflows(df)
    for fid in FLOOD_NODE_IDS:
        donor = _FLOOD_DOWNSTREAM_MAP[fid]
        assert (out.loc[df[donor] > 0, fid] > 0).all()


def test_flood_node_share_of_donor_is_order_one():
    """A drainage-area-agnostic magnitude guard; the pre-fix code gave ~0.004."""
    df = _synthetic_inflows()
    out = add_flood_nodes_to_inflows(df)
    for fid in FLOOD_NODE_IDS:
        donor = _FLOOD_DOWNSTREAM_MAP[fid]
        share = (out[fid] / df[donor]).mean()
        assert 0.02 < share < 0.75, f"{fid} share of {donor} is {share:.4f}"


def test_subtract_preserves_lordville_mass_balance():
    df = _synthetic_inflows()
    augmented = add_flood_nodes_to_inflows(df)
    rhs = augmented["delLordville"] + augmented["01426500"] + augmented["01421000"]
    pd.testing.assert_series_equal(df["delLordville"], rhs, check_names=False)


def test_subtract_preserves_montague_mass_balance():
    df = _synthetic_inflows()
    augmented = add_flood_nodes_to_inflows(df)
    rhs = augmented["delMontague"] + augmented["01436690"]
    pd.testing.assert_series_equal(df["delMontague"], rhs, check_names=False)


def test_add_flood_nodes_adds_three_new_columns_in_order():
    df = _synthetic_inflows()
    out = add_flood_nodes_to_inflows(df)

    assert list(out.columns[-3:]) == list(FLOOD_NODE_IDS)
    for col in df.columns:
        assert col in out.columns
    assert out.shape[1] == df.shape[1] + 3


def test_add_flood_nodes_does_not_mutate_input():
    df = _synthetic_inflows()
    df_copy = df.copy()
    _ = add_flood_nodes_to_inflows(df)
    pd.testing.assert_frame_equal(df, df_copy)


def test_subtract_warns_when_clamping_negatives():
    """The clamp is unreachable via add_flood_nodes_to_inflows, but must still work."""
    df = _synthetic_inflows()
    df["01426500"] = df["delLordville"] * 2.0
    df["01421000"] = 0.0
    df["01436690"] = 0.0

    with pytest.warns(UserWarning, match="went negative"):
        out = subtract_flood_inflows_from_downstream(df)

    assert (out["delLordville"] >= 0).all()


# ---------------------------------------------------------------------------
# D. Cross-module consistency
# ---------------------------------------------------------------------------

def test_flood_node_ids_match_lists_module():
    from pywrdrb.flood_thresholds import flood_stage_thresholds
    from pywrdrb.utils.lists import flood_monitoring_nodes, majorflow_list

    assert FLOOD_NODE_IDS == tuple(flood_monitoring_nodes)
    for fid in FLOOD_NODE_IDS:
        assert fid in majorflow_list
        assert fid in flood_stage_thresholds


def test_flood_downstream_map_matches_flood_topology():
    """_FLOOD_DOWNSTREAM_MAP must not drift from the model topology."""
    from pywrdrb.pywr_drb_node_data import TopologyDictionaries

    immediate_downstream = TopologyDictionaries.get(include_flood_nodes=True)[0]
    for fid, downstream in _FLOOD_DOWNSTREAM_MAP.items():
        assert immediate_downstream[fid] == downstream
