"""
Regression test for the NYC reservoir `interpolatedvolume` parameters.

Over long/large synthetic ensembles a reservoir storage volume can land a hair below
zero (~ -2e-8 MG: an LP-solver / floating-point residual at "empty"), just past the
`-EPS` (= -1e-8) lower bound of these parameters' interpolation domain. With pywr's
default `interp1d(..., bounds_error=True)` this raises:

    ValueError: A value (...) in x_new is below the interpolation range's minimum value

`ModelBuilder` now emits `interp_kwargs={"bounds_error": False, "fill_value": <values>}`
for these parameters so such dust clamps to the empty/full edge value instead of crashing
(mirrors `pywrdrb/utils/rating_curves.py`). This test verifies:

1. The builder emits the clamping `interp_kwargs` on the relevant parameters.
2. Rebuilding the interpolator exactly as pywr does makes a sub-floor query succeed and
   clamp to the empty edge, while in-domain queries are unchanged.
3. (Negative control) the pywr default `bounds_error=True` would raise on the same query,
   i.e. this test fails without the fix.
"""

import numpy as np
import pytest
from scipy.interpolate import interp1d

import pywrdrb
from pywrdrb.utils.dates import model_date_ranges

# A tiny-negative "dust" volume observed in failing runs (just past the -1e-8 floor).
DUST_VOLUME = -2.0938330180797493e-08

# Representative parameters: one cost map and the two identity (volume->volume) maps.
PARAMS_UNDER_TEST = ["storage_cost_cannonsville", "volume_cannonsville", "volume_agg_nyc"]


@pytest.fixture(scope="module")
def model_parameters():
    """Build the model dict once and return its parameters block."""
    start, end = model_date_ranges["nhmv10"]
    mb = pywrdrb.ModelBuilder(inflow_type="nhmv10", start_date=start, end_date=end)
    mb.make_model()
    return mb.model_dict["parameters"]


def _build_interp(param):
    """Reconstruct the scipy interpolator exactly as pywr's
    AbstractInterpolatedParameter.setup() does, including the fill_value list->tuple
    conversion performed by its interp_kwargs setter."""
    volumes = np.asarray(param["volumes"], np.float64)
    values = np.asarray(param["values"], np.float64)
    interp_kwargs = dict(param["interp_kwargs"])
    fill_value = interp_kwargs.get("fill_value")
    if isinstance(fill_value, list):
        interp_kwargs["fill_value"] = tuple(fill_value)
    return interp1d(volumes, values, **interp_kwargs)


@pytest.mark.parametrize("param_name", PARAMS_UNDER_TEST)
def test_clamping_interp_kwargs_emitted(model_parameters, param_name):
    param = model_parameters[param_name]
    assert param["type"] == "interpolatedvolume"
    assert "interp_kwargs" in param, f"{param_name} missing interp_kwargs"
    interp_kwargs = param["interp_kwargs"]
    assert interp_kwargs["bounds_error"] is False
    fill_value = interp_kwargs["fill_value"]
    assert len(fill_value) == 2, "fill_value must give the (lower, upper) edge values"
    # fill_value must be exactly the value-array endpoints so out-of-range clamps to edge.
    assert list(fill_value) == [param["values"][0], param["values"][-1]]


@pytest.mark.parametrize("param_name", PARAMS_UNDER_TEST)
def test_subfloor_volume_clamps_to_empty_edge(model_parameters, param_name):
    param = model_parameters[param_name]
    interp = _build_interp(param)
    # The failing query no longer raises and clamps to the empty (lower) edge value.
    result = float(interp(DUST_VOLUME))
    assert result == pytest.approx(param["values"][0])


@pytest.mark.parametrize("param_name", PARAMS_UNDER_TEST)
def test_in_domain_values_unchanged(model_parameters, param_name):
    """Clamping must not perturb results inside the valid domain."""
    param = model_parameters[param_name]
    interp = _build_interp(param)
    volumes = param["volumes"]
    values = param["values"]

    # A reference interpolator with no out-of-range handling (default behavior) must
    # agree exactly with the clamping one for any in-domain query.
    ref = interp1d(np.asarray(volumes, np.float64), np.asarray(values, np.float64))

    # Sample a few strictly-in-domain volumes (avoid the exact edges).
    lo, hi = volumes[0], volumes[-1]
    for frac in (0.01, 0.25, 0.5, 0.75, 0.99):
        v = lo + frac * (hi - lo)
        assert float(interp(v)) == pytest.approx(float(ref(v)), rel=0, abs=0)


@pytest.mark.parametrize("param_name", PARAMS_UNDER_TEST)
def test_default_bounds_error_would_raise(model_parameters, param_name):
    """Negative control: without the fix (pywr default bounds_error=True) the same
    sub-floor query raises -- demonstrating the bug this change fixes."""
    param = model_parameters[param_name]
    volumes = np.asarray(param["volumes"], np.float64)
    values = np.asarray(param["values"], np.float64)
    buggy = interp1d(volumes, values)  # scipy default bounds_error=True
    with pytest.raises(ValueError):
        buggy(DUST_VOLUME)
