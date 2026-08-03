"""
Physical-plausibility guards for flood-node catchment inflows, on real data.

The synthetic-frame tests in ``test_flood_node_inflow_helpers.py`` pin the
formulas. These tests instead ask whether the resulting series could plausibly
be a real catchment's runoff. That is the check that was missing when the
flood-node inflows silently ran at ~2% of physical magnitude: the formulas were
self-consistent, but the numbers were absurd.

Skipped automatically when the required inflow dataset is not present.
"""
import numpy as np
import pandas as pd
import pytest

from pywrdrb.pre.flood_node_inflows import (
    FLOOD_NODE_IDS,
    _FLOOD_DOWNSTREAM_MAP,
    add_flood_nodes_to_inflows,
    incremental_drainage_areas,
)

# Inflow datasets to check. Any dataset shipping catchment_inflow_mgd.csv works;
# these are the two that are actually run with enable_nyc_flood_operations=True.
INFLOW_TYPES = ["pub_nhmv10_BC_withObsScaled", "nhmv10"]

# Humid-temperate catchments in the upper Delaware basin yield roughly
# 1-2 MGD/sq mi on average. This band is deliberately wide: it is an
# order-of-magnitude guard, not a calibration target.
MIN_YIELD_MGD_PER_SQMI = 0.5
MAX_YIELD_MGD_PER_SQMI = 4.0

ANALYSIS_START = "2000-01-01"
ANALYSIS_END = "2023-12-31"


def _load_marginal_inflows(inflow_type):
    """Load a dataset's marginal catchment inflows, or skip if unavailable."""
    from pywrdrb.path_manager import get_pn_object

    pn = get_pn_object()
    try:
        flows_dir = str(pn.sc.get(f"flows/{inflow_type}"))
    except Exception:  # pragma: no cover - depends on local data install
        pytest.skip(f"inflow type {inflow_type} is not registered")

    path = pd.io.common.os.path.join(flows_dir, "catchment_inflow_mgd.csv")
    if not pd.io.common.os.path.exists(path):
        pytest.skip(f"{path} not present")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.loc[ANALYSIS_START:ANALYSIS_END]


@pytest.fixture(scope="module", params=INFLOW_TYPES)
def augmented(request):
    original = _load_marginal_inflows(request.param)
    return request.param, original, add_flood_nodes_to_inflows(original)


@pytest.mark.parametrize("flood_node", FLOOD_NODE_IDS)
def test_unit_area_yield_in_physical_range(augmented, flood_node):
    """Mean runoff per unit area must be hydrologically plausible.

    Pre-fix values were ~0.02 MGD/sq mi at Hale Eddy and Fishs Eddy and
    ~0.004 at Bridgeville, i.e. two orders of magnitude too low.
    """
    inflow_type, _, out = augmented
    da = incremental_drainage_areas()[flood_node]
    yield_ = out[flood_node].mean() / da
    assert MIN_YIELD_MGD_PER_SQMI <= yield_ <= MAX_YIELD_MGD_PER_SQMI, (
        f"{inflow_type}/{flood_node}: {yield_:.3f} MGD/sq mi over {da} sq mi "
        f"is outside [{MIN_YIELD_MGD_PER_SQMI}, {MAX_YIELD_MGD_PER_SQMI}]"
    )


@pytest.mark.parametrize("flood_node", FLOOD_NODE_IDS)
def test_increment_nonzero_on_most_days(augmented, flood_node):
    """A real catchment does not stop producing runoff for most of the record.

    Pre-fix: 7.5% of days non-zero at Hale Eddy and Fishs Eddy, 21.6% at
    Bridgeville. Bridgeville keeps a looser bound because its donor
    (delMontague's marginal inflow) is itself zero on ~21% of days -- an
    upstream data characteristic this module does not attempt to correct.
    """
    inflow_type, _, out = augmented
    frac = float((out[flood_node] > 0).mean())
    floor = 0.75 if flood_node == "01436690" else 0.95
    assert frac >= floor, (
        f"{inflow_type}/{flood_node}: non-zero on only {frac:.1%} of days "
        f"(expected >= {floor:.0%})"
    )


def test_mass_is_conserved_exactly(augmented):
    """The hard constraint: net basin flow must not change.

    Checked per donor node and over the whole frame.
    """
    inflow_type, original, out = augmented

    for donor in ("delLordville", "delMontague"):
        moved = sum(
            out[fid] for fid in FLOOD_NODE_IDS if _FLOOD_DOWNSTREAM_MAP[fid] == donor
        )
        np.testing.assert_allclose(
            (out[donor] + moved).values,
            original[donor].values,
            rtol=0,
            atol=1e-8,
            err_msg=f"{inflow_type}: mass not conserved at {donor}",
        )

    total_before = original.to_numpy().sum()
    total_after = out.to_numpy().sum()
    assert total_after == pytest.approx(total_before, rel=1e-12), (
        f"{inflow_type}: basin total changed by {total_after - total_before:g} MGD"
    )


def test_no_clamping_occurs(augmented, recwarn):
    """Donor residuals must never go negative, so the clamp must never fire."""
    inflow_type, original, _ = augmented
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        add_flood_nodes_to_inflows(original)  # raises if any clamp warning fires


@pytest.mark.parametrize("flood_node", FLOOD_NODE_IDS)
def test_increment_is_a_minority_share_of_its_donor(augmented, flood_node):
    """Sanity bound on the redistribution: a flood node cannot take everything."""
    _, original, out = augmented
    donor = _FLOOD_DOWNSTREAM_MAP[flood_node]
    share = out[flood_node].sum() / original[donor].sum()
    assert 0.0 < share < 0.75
