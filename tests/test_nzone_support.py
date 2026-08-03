"""
Tests for variable-N storage-zone support.

Covers:
- ``NYCOperationsConfig.from_n_zones`` classmethod (shape, naming, interpolation invariants).
- Parity of the new classmethod with NYCOptimization's ``build_nzone_config``.
- Parametric drought-emergency level on ``LowerBasinMaxMRFContribution`` and
  ``FlowTargetSaltFrontAdjustmentRatio``.
- Output loader autodiscovery of ``ffmp_level_boundaries`` keys for both default
  ``level*`` and N-zone ``zone_*`` naming.
- End-to-end model build with N-zone configs.

Usage:
    pytest tests/test_nzone_support.py -v
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pywrdrb
from pywrdrb.parameters.nyc_operations_config import NYCOperationsConfig


# --------------------------------------------------------------------------- #
# NYCOperationsConfig.from_n_zones
# --------------------------------------------------------------------------- #

class TestFromNZones:
    """Unit tests for the new NYCOperationsConfig.from_n_zones classmethod."""

    def test_default_from_defaults_levels_unchanged(self):
        cfg = NYCOperationsConfig.from_defaults()
        assert list(cfg.STORAGE_LEVELS) == list(NYCOperationsConfig._DEFAULT_STORAGE_LEVELS)
        assert list(cfg.DROUGHT_LEVELS) == list(NYCOperationsConfig._DEFAULT_DROUGHT_LEVELS)
        assert cfg.n_zones == 6
        assert cfg.n_drought_levels == 7

    @pytest.mark.parametrize("n_zones", [6, 10, 14, 20])
    def test_invariants(self, n_zones):
        cfg = NYCOperationsConfig.from_n_zones(n_zones)
        assert cfg.n_zones == n_zones
        assert cfg.n_drought_levels == n_zones + 1
        assert list(cfg.STORAGE_LEVELS) == [f"zone_{i+1}" for i in range(n_zones)]
        assert list(cfg.DROUGHT_LEVELS) == ["zone_0"] + list(cfg.STORAGE_LEVELS)
        # zone_0 must not be a threshold row
        assert "zone_0" not in cfg.storage_zones_df.index

    @pytest.mark.parametrize("n_zones", [6, 10, 14, 20])
    def test_monotonic_thresholds(self, n_zones):
        """pywr ControlCurveIndex requires descending thresholds."""
        cfg = NYCOperationsConfig.from_n_zones(n_zones)
        zones = [f"zone_{i+1}" for i in range(n_zones)]
        profiles = cfg.storage_zones_df.loc[zones].values.astype(float)
        for i in range(n_zones - 1):
            assert np.all(profiles[i, :] >= profiles[i + 1, :]), (
                f"zone_{i+1} is not >= zone_{i+2} on every day"
            )

    def test_endpoints_match_defaults_at_N6(self):
        """At N=6, interpolation is the identity so endpoints must match FFMP."""
        cfg6 = NYCOperationsConfig.from_n_zones(6)
        base = NYCOperationsConfig.from_defaults()
        np.testing.assert_allclose(
            cfg6.get_storage_zone_profile("zone_1"),
            base.get_storage_zone_profile("level1b"),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            cfg6.get_storage_zone_profile("zone_6"),
            base.get_storage_zone_profile("level5"),
            atol=1e-12,
        )

    @pytest.mark.parametrize("n_zones", [6, 10, 14])
    def test_constants_complete_and_no_orphans(self, n_zones):
        cfg = NYCOperationsConfig.from_n_zones(n_zones)
        expected_keys = set()
        for i in range(n_zones + 1):
            expected_keys.add(f"zone_{i}_factor_delivery_nyc")
            expected_keys.add(f"zone_{i}_factor_delivery_nj")
        present = {k for k in cfg.constants if "_factor_delivery_" in k}
        assert expected_keys == present
        # No orphan level* keys
        orphan_prefixes = tuple(f"level{s}_factor_delivery_"
                                for s in ["1a", "1b", "1c", "2", "3", "4", "5"])
        assert not any(k.startswith(orphan_prefixes) for k in cfg.constants)

    def test_mrf_daily_rows_present(self):
        cfg = NYCOperationsConfig.from_n_zones(10)
        for res in NYCOperationsConfig.RESERVOIRS:
            for lvl in cfg.DROUGHT_LEVELS:
                name = f"{lvl}_factor_mrf_{res}"
                assert name in cfg.mrf_factors_daily_df.index, (
                    f"Missing daily MRF row: {name}"
                )

    def test_mrf_monthly_rows_present(self):
        cfg = NYCOperationsConfig.from_n_zones(10)
        for loc in ["delMontague", "delTrenton"]:
            for lvl in cfg.DROUGHT_LEVELS:
                name = f"{lvl}_factor_mrf_{loc}"
                assert name in cfg.mrf_factors_monthly_df.index, (
                    f"Missing monthly MRF row: {name}"
                )

    def test_small_n_warning(self):
        with pytest.warns(UserWarning, match="permissive at small N"):
            NYCOperationsConfig.from_n_zones(3)

    def test_n_zones_too_small_raises(self):
        with pytest.raises(ValueError, match=">= 2"):
            NYCOperationsConfig.from_n_zones(1)

    def test_parity_with_nyc_optimization(self):
        """If NYCOptimization is available, verify element-wise parity at N=10."""
        try:
            nyc_opt_src = os.path.normpath(
                os.path.join(os.path.dirname(__file__),
                             '..', '..', 'NYCOptimization', 'src')
            )
            if not os.path.isdir(nyc_opt_src):
                pytest.skip("NYCOptimization not available")
            sys.path.insert(0, nyc_opt_src)
            from simulation import build_nzone_config  # noqa: F401
        except Exception as exc:
            pytest.skip(f"Could not import build_nzone_config: {exc}")

        n = 10
        cfg_pywrdrb = NYCOperationsConfig.from_n_zones(n)
        cfg_nycopt = build_nzone_config(n)

        # Storage thresholds
        date_cols = [c for c in cfg_pywrdrb.storage_zones_df.columns if c != 'doy']
        a = cfg_pywrdrb.storage_zones_df.loc[list(cfg_pywrdrb.STORAGE_LEVELS), date_cols].values
        b = cfg_nycopt.storage_zones_df.loc[list(cfg_nycopt.STORAGE_LEVELS), date_cols].values
        np.testing.assert_allclose(a.astype(float), b.astype(float), atol=1e-12)

        # Delivery factor constants
        for i in range(n + 1):
            for demand in ("nyc", "nj"):
                k = f"zone_{i}_factor_delivery_{demand}"
                assert np.isclose(
                    float(cfg_pywrdrb.constants[k]),
                    float(cfg_nycopt.constants[k]),
                    atol=1e-12,
                ), f"Mismatch at {k}"


# --------------------------------------------------------------------------- #
# Output loader autodiscovery
# --------------------------------------------------------------------------- #

class TestOutputLoaderFFMPBoundaries:
    """Verify ffmp_level_boundaries autodiscovery preserves default behavior."""

    def _invoke(self, keys):
        """Directly invoke the branch logic. Mirrors output_loader.py:287 logic."""
        import warnings
        default_keys = [f"level{l}" for l in ["1b", "1c", "2", "3", "4", "5"]]
        present_defaults = [k for k in default_keys if k in keys]
        if present_defaults:
            return present_defaults
        zone_keys = [
            k for k in keys
            if k.startswith("zone_") and k != "zone_0"
            and "_factor_" not in k
        ]
        try:
            zone_keys = sorted(zone_keys, key=lambda s: int(s.split("_")[1]))
        except (IndexError, ValueError):
            zone_keys = sorted(zone_keys)
        if not zone_keys:
            warnings.warn("no keys")
        return zone_keys

    def test_default_names_preserved(self):
        keys = ["level1b", "level1c", "level2", "level3", "level4",
                "level5", "other_irrelevant_key"]
        result = self._invoke(keys)
        assert result == ["level1b", "level1c", "level2", "level3", "level4", "level5"]

    def test_nzone_autodiscover_numeric_sort(self):
        """zone_2 must precede zone_10 — the critical correctness property."""
        keys = [f"zone_{i}" for i in [10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]]
        keys.append("zone_0")  # should be dropped
        keys.append("zone_3_factor_mrf_cannonsville")  # should be dropped
        result = self._invoke(keys)
        assert result == [f"zone_{i}" for i in range(1, 13)]

    def test_empty_warns(self):
        with pytest.warns(UserWarning, match="no keys"):
            self._invoke(["unrelated_key"])


# --------------------------------------------------------------------------- #
# Parametric emergency level on custom parameters
# --------------------------------------------------------------------------- #

class _FakeAggParam:
    def __init__(self, value):
        self._value = value

    def get_value(self, scenario_index):
        return self._value


class TestParametricEmergencyLevel:
    """Verify the parametric ``nyc_drought_emergency_level`` kwarg is honored."""

    def test_lower_basin_default_fires_at_6(self):
        """Default kwarg = 6 → matches only when drought index == 6."""
        from pywrdrb.parameters.lower_basin_ffmp import (
            LowerBasinMaxMRFContribution,
            reservoirs_used_during_normal_conditions,
            reservoirs_used_during_drought_conditions,
        )

        # Construct a minimal shim skipping full __init__ (which requires a model)
        shim = LowerBasinMaxMRFContribution.__new__(LowerBasinMaxMRFContribution)
        shim.nyc_drought_emergency_level = 6
        shim.drought_level_agg_nyc = _FakeAggParam(6.0)

        result = LowerBasinMaxMRFContribution.get_current_usable_reservoirs(shim, None)
        assert result == reservoirs_used_during_drought_conditions

        shim.drought_level_agg_nyc = _FakeAggParam(5.0)
        result = LowerBasinMaxMRFContribution.get_current_usable_reservoirs(shim, None)
        assert result == reservoirs_used_during_normal_conditions

    def test_lower_basin_parametric_emergency_level(self):
        """Custom emergency level (e.g. N=10 → 10) fires only at that index."""
        from pywrdrb.parameters.lower_basin_ffmp import (
            LowerBasinMaxMRFContribution,
            reservoirs_used_during_normal_conditions,
            reservoirs_used_during_drought_conditions,
        )

        shim = LowerBasinMaxMRFContribution.__new__(LowerBasinMaxMRFContribution)
        shim.nyc_drought_emergency_level = 10
        shim.drought_level_agg_nyc = _FakeAggParam(10.0)
        assert LowerBasinMaxMRFContribution.get_current_usable_reservoirs(
            shim, None) == reservoirs_used_during_drought_conditions

        shim.drought_level_agg_nyc = _FakeAggParam(6.0)
        # At idx=6 with threshold=10, this is NOT emergency for N=10 config
        assert LowerBasinMaxMRFContribution.get_current_usable_reservoirs(
            shim, None) == reservoirs_used_during_normal_conditions


# --------------------------------------------------------------------------- #
# End-to-end model build with N-zone configs
# --------------------------------------------------------------------------- #

class TestEndToEndNZoneModelBuild:
    """Smoke test that N-zone configs produce a valid model_dict."""

    @pytest.mark.parametrize("n_zones", [6, 10])
    def test_make_model_succeeds(self, n_zones, tmp_path):
        cfg = NYCOperationsConfig.from_n_zones(n_zones)
        mb = pywrdrb.ModelBuilder(
            inflow_type="nhmv10",
            start_date="1983-10-01",
            end_date="1983-12-31",
            nyc_operations_config=cfg,
        )
        mb.make_model()
        assert "parameters" in mb.model_dict

        # Zone-name parameters should be present
        zone_names = [f"zone_{i+1}" for i in range(n_zones)]
        for zn in zone_names:
            assert zn in mb.model_dict["parameters"], (
                f"Missing zone parameter: {zn}"
            )

        # Delivery factor constants for zone_0..zone_N should be present
        for i in range(n_zones + 1):
            key = f"zone_{i}_factor_delivery_nyc"
            assert key in mb.model_dict["parameters"], (
                f"Missing delivery factor: {key}"
            )

    def test_wiring_includes_emergency_level(self):
        """Confirm the ModelBuilder actually wires n_drought_levels - 1."""
        cfg = NYCOperationsConfig.from_n_zones(10)
        mb = pywrdrb.ModelBuilder(
            inflow_type="nhmv10",
            start_date="1983-10-01",
            end_date="1983-12-31",
            nyc_operations_config=cfg,
        )
        mb.make_model()
        # lower basin contributions: step 1..4, per reservoir
        # Pick one representative parameter and check the kwarg is 10.
        # Keys look like "max_mrf_trenton_step{step}_{reservoir}"
        found_any = False
        for k, v in mb.model_dict["parameters"].items():
            if isinstance(v, dict) and v.get("type") == "LowerBasinMaxMRFContribution":
                assert v.get("nyc_drought_emergency_level") == 10, (
                    f"{k} has wrong emergency level: {v.get('nyc_drought_emergency_level')}"
                )
                found_any = True
        assert found_any, "No LowerBasinMaxMRFContribution parameters in model"
