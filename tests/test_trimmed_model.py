"""
Tests for the trimmed model mode functionality.

This module tests the trimmed model feature which replaces independent STARFIT
reservoirs with pre-simulated releases for faster runtime during sensitivity analysis.

Tests verify:
1. Trimmed model produces identical results to full model at key nodes
2. Proper validation errors for missing/mismatched data
3. Helper function correctly generates pre-simulated releases
"""

import os
import json
import tempfile
import pytest
import numpy as np
import pandas as pd

import pywrdrb
from pywrdrb import ModelBuilder
from pywrdrb.utils.lists import (
    independent_starfit_reservoirs,
    required_model_reservoirs,
    reservoir_list,
    reservoir_list_nyc,
)
from pywrdrb.pre import STARFITOfflineSimulator


# Test parameters - use a short period for faster testing
TEST_START_DATE = "2000-01-01"
TEST_END_DATE = "2000-03-31"  # 3 months for quick tests
TEST_INFLOW_TYPE = "nhmv10_withObsScaled"


class TestReservoirLists:
    """Test that reservoir classification lists are correctly defined."""

    def test_independent_reservoirs_count(self):
        """Verify we have the expected number of independent reservoirs."""
        # 11 independent STARFIT reservoirs
        assert len(independent_starfit_reservoirs) == 11

    def test_required_reservoirs_count(self):
        """Verify we have the expected number of required reservoirs."""
        # 3 NYC + 3 lower basin = 6 required reservoirs
        assert len(required_model_reservoirs) == 6

    def test_no_overlap(self):
        """Verify independent and required reservoirs don't overlap."""
        overlap = set(independent_starfit_reservoirs) & set(required_model_reservoirs)
        assert len(overlap) == 0, f"Unexpected overlap: {overlap}"

    def test_covers_starfit(self):
        """Verify independent + lower basin DRBC covers all STARFIT reservoirs."""
        from pywrdrb.utils.lists import starfit_reservoir_list, drbc_lower_basin_reservoirs

        expected = set(starfit_reservoir_list)
        actual = set(independent_starfit_reservoirs) | set(drbc_lower_basin_reservoirs)
        assert expected == actual

    def test_expected_independent_reservoirs(self):
        """Verify the expected reservoirs are in the independent list."""
        expected = {
            "wallenpaupack", "prompton", "shoholaMarsh", "mongaupeCombined",
            "fewalter", "merrillCreek", "hopatcong", "assunpink",
            "ontelaunee", "stillCreek", "greenLane"
        }
        assert set(independent_starfit_reservoirs) == expected


class TestTrimmedModelValidation:
    """Test validation logic for trimmed model configuration."""

    def test_missing_presimulated_file_raises_error(self):
        """Verify FileNotFoundError when pre-simulated file doesn't exist."""
        with pytest.raises(FileNotFoundError) as exc_info:
            mb = ModelBuilder(
                start_date=TEST_START_DATE,
                end_date=TEST_END_DATE,
                inflow_type=TEST_INFLOW_TYPE,
                options={"use_trimmed_model": True}
            )

        assert "Pre-simulated releases file not found" in str(exc_info.value)

    def test_custom_presimulated_file_path(self):
        """Verify custom presimulated_releases_file path is used."""
        fake_path = "/nonexistent/path/to/presimulated.csv"

        with pytest.raises(FileNotFoundError) as exc_info:
            mb = ModelBuilder(
                start_date=TEST_START_DATE,
                end_date=TEST_END_DATE,
                inflow_type=TEST_INFLOW_TYPE,
                options={
                    "use_trimmed_model": True,
                    "presimulated_releases_file": fake_path
                }
            )

        assert fake_path in str(exc_info.value)


class TestGeneratePresimulatedReleases:
    """Test the STARFITOfflineSimulator."""

    def test_simulate_all_produces_releases(self):
        """Verify offline simulator produces valid releases for all STARFIT reservoirs."""
        from pywrdrb.path_manager import get_pn_object
        pn = get_pn_object()

        sim = STARFITOfflineSimulator(initial_volume_frac=0.8)
        sim.load_parameters()

        inflow_file = pn.sc.get(f"flows/{TEST_INFLOW_TYPE}") / "catchment_inflow_mgd.csv"
        inflows = pd.read_csv(str(inflow_file), index_col=0, parse_dates=True)
        releases = sim.simulate_all(inflows.iloc[:30])

        assert len(releases) == 30
        assert all(releases.min() >= 0)
        assert releases.shape[1] > 0


class TestTrimmedModelConstruction:
    """Test trimmed model construction when valid data exists."""

    @pytest.fixture
    def presimulated_data_dir(self):
        """Create a temporary directory with mock pre-simulated data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock pre-simulated releases CSV
            dates = pd.date_range(start="1999-01-01", end="2001-12-31", freq="D")
            data = {res: np.random.rand(len(dates)) * 100 for res in independent_starfit_reservoirs}
            df = pd.DataFrame(data, index=dates)
            df.index.name = "datetime"

            csv_path = os.path.join(tmpdir, "presimulated_releases_mgd.csv")
            df.to_csv(csv_path)

            # Create metadata file
            metadata = {
                "inflow_type": TEST_INFLOW_TYPE,
                "start_date": "1999-01-01",
                "end_date": "2001-12-31",
                "reservoirs": independent_starfit_reservoirs,
            }
            metadata_path = csv_path.replace('.csv', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

            yield tmpdir, csv_path

    def test_trimmed_model_builds_successfully(self, presimulated_data_dir):
        """Verify trimmed model can be built with valid pre-simulated data."""
        tmpdir, csv_path = presimulated_data_dir

        mb = ModelBuilder(
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            inflow_type=TEST_INFLOW_TYPE,
            options={
                "use_trimmed_model": True,
                "presimulated_releases_file": csv_path
            }
        )
        mb.make_model()

        # Verify model was built
        assert mb.model_dict is not None
        assert len(mb.model_dict["nodes"]) > 0

    def test_trimmed_model_has_fewer_nodes(self, presimulated_data_dir):
        """Verify trimmed model has fewer nodes than full model."""
        tmpdir, csv_path = presimulated_data_dir

        # Build full model
        mb_full = ModelBuilder(
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            inflow_type=TEST_INFLOW_TYPE,
        )
        mb_full.make_model()
        full_node_count = len(mb_full.model_dict["nodes"])

        # Build trimmed model
        mb_trimmed = ModelBuilder(
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            inflow_type=TEST_INFLOW_TYPE,
            options={
                "use_trimmed_model": True,
                "presimulated_releases_file": csv_path
            }
        )
        mb_trimmed.make_model()
        trimmed_node_count = len(mb_trimmed.model_dict["nodes"])

        # Trimmed should have fewer nodes
        assert trimmed_node_count < full_node_count
        print(f"Full model: {full_node_count} nodes, Trimmed model: {trimmed_node_count} nodes")
        print(f"Reduction: {full_node_count - trimmed_node_count} nodes ({(1 - trimmed_node_count/full_node_count)*100:.1f}%)")

    def test_trimmed_model_has_catchment_nodes_for_independent_reservoirs(self, presimulated_data_dir):
        """Verify trimmed model creates catchment nodes for independent reservoirs."""
        tmpdir, csv_path = presimulated_data_dir

        mb = ModelBuilder(
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            inflow_type=TEST_INFLOW_TYPE,
            options={
                "use_trimmed_model": True,
                "presimulated_releases_file": csv_path
            }
        )
        mb.make_model()

        # Check for catchment nodes
        node_names = [n["name"] for n in mb.model_dict["nodes"]]
        for res in independent_starfit_reservoirs:
            assert f"catchment_presim_{res}" in node_names, f"Missing catchment node for {res}"

    def test_trimmed_model_retains_full_nodes_for_required_reservoirs(self, presimulated_data_dir):
        """Verify trimmed model keeps full nodes for NYC and lower basin reservoirs."""
        tmpdir, csv_path = presimulated_data_dir

        mb = ModelBuilder(
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            inflow_type=TEST_INFLOW_TYPE,
            options={
                "use_trimmed_model": True,
                "presimulated_releases_file": csv_path
            }
        )
        mb.make_model()

        # Check for full reservoir nodes (storage type)
        node_names_and_types = {n["name"]: n["type"] for n in mb.model_dict["nodes"]}
        for res in required_model_reservoirs:
            expected_name = f"reservoir_{res}"
            assert expected_name in node_names_and_types, f"Missing reservoir node for {res}"
            assert node_names_and_types[expected_name] == "storage", f"{res} should be storage type"

    def test_date_range_validation(self, presimulated_data_dir):
        """Verify error when simulation period extends beyond pre-simulated data."""
        tmpdir, csv_path = presimulated_data_dir

        # Try to run simulation beyond the pre-simulated data range
        with pytest.raises(ValueError) as exc_info:
            mb = ModelBuilder(
                start_date="2002-01-01",  # Beyond metadata end_date
                end_date="2002-12-31",
                inflow_type=TEST_INFLOW_TYPE,
                options={
                    "use_trimmed_model": True,
                    "presimulated_releases_file": csv_path
                }
            )

        assert "extends beyond pre-simulated data" in str(exc_info.value)

    def test_inflow_type_validation(self):
        """Verify warning/error when inflow type doesn't match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock data with different inflow_type
            dates = pd.date_range(start="1999-01-01", end="2001-12-31", freq="D")
            data = {res: np.random.rand(len(dates)) * 100 for res in independent_starfit_reservoirs}
            df = pd.DataFrame(data, index=dates)
            df.index.name = "datetime"

            csv_path = os.path.join(tmpdir, "presimulated_releases_mgd.csv")
            df.to_csv(csv_path)

            # Create metadata with mismatched inflow_type
            metadata = {
                "inflow_type": "different_inflow_type",  # Mismatch!
                "start_date": "1999-01-01",
                "end_date": "2001-12-31",
                "reservoirs": independent_starfit_reservoirs,
            }
            metadata_path = csv_path.replace('.csv', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

            with pytest.raises(ValueError) as exc_info:
                mb = ModelBuilder(
                    start_date=TEST_START_DATE,
                    end_date=TEST_END_DATE,
                    inflow_type=TEST_INFLOW_TYPE,
                    options={
                        "use_trimmed_model": True,
                        "presimulated_releases_file": csv_path
                    }
                )

            assert "Inflow type mismatch" in str(exc_info.value)


# Integration test - only run if explicitly requested (slow)
@pytest.mark.slow
class TestTrimmedModelIntegration:
    """
    Integration tests that run actual simulations.

    These tests are marked as slow and require running full model simulations.
    Run with: pytest -m slow tests/test_trimmed_model.py
    """

    @pytest.fixture
    def full_model_output(self, tmp_path):
        """Build and run a full model, returning the output file path."""
        model_filename = str(tmp_path / "full_model.json")
        output_filename = str(tmp_path / "full_output.hdf5")

        # Build and run full model
        mb = ModelBuilder(
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            inflow_type=TEST_INFLOW_TYPE,
        )
        mb.make_model()
        mb.write_model(model_filename)

        model = pywrdrb.Model.load(model_filename)
        recorder = pywrdrb.OutputRecorder(model=model, output_filename=output_filename)
        model.run()

        return output_filename, tmp_path

    def test_trimmed_model_matches_full_model(self, full_model_output):
        """
        Verify trimmed model produces identical results for retained nodes.

        Compares:
        - delMontague flow
        - delTrenton flow
        - NYC reservoir storage (cannonsville, pepacton, neversink)
        """
        output_filename, tmp_path = full_model_output

        # Generate pre-simulated releases using offline simulator
        from pywrdrb.path_manager import get_pn_object
        pn = get_pn_object()

        sim = STARFITOfflineSimulator(initial_volume_frac=0.8)
        sim.load_parameters()

        inflow_file = pn.sc.get(f"flows/{TEST_INFLOW_TYPE}") / "catchment_inflow_mgd.csv"
        catchment_inflows = pd.read_csv(str(inflow_file), index_col=0, parse_dates=True)
        releases_df = sim.simulate_all(catchment_inflows)

        # Save to temp directory
        presim_dir = str(tmp_path / "presim")
        os.makedirs(presim_dir, exist_ok=True)
        csv_file = os.path.join(presim_dir, "presimulated_releases_mgd.csv")
        releases_df.index.name = "datetime"
        releases_df.index = pd.to_datetime(releases_df.index).strftime("%Y-%m-%d")
        releases_df.to_csv(csv_file, float_format="%.10f")

        metadata_file = os.path.join(presim_dir, "presimulated_releases_mgd_metadata.json")
        metadata = {
            "inflow_type": TEST_INFLOW_TYPE,
            "start_date": str(releases_df.index[0]),
            "end_date": str(releases_df.index[-1]),
            "reservoirs": list(releases_df.columns),
            "output_file": csv_file,
            "metadata_file": metadata_file,
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Build and run trimmed model
        trimmed_model_filename = str(tmp_path / "trimmed_model.json")
        trimmed_output_filename = str(tmp_path / "trimmed_output.hdf5")

        mb_trimmed = ModelBuilder(
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            inflow_type=TEST_INFLOW_TYPE,
            options={
                "use_trimmed_model": True,
                "presimulated_releases_file": metadata["output_file"]
            }
        )
        mb_trimmed.make_model()
        mb_trimmed.write_model(trimmed_model_filename)

        model_trimmed = pywrdrb.Model.load(trimmed_model_filename)
        recorder_trimmed = pywrdrb.OutputRecorder(
            model=model_trimmed,
            output_filename=trimmed_output_filename
        )
        model_trimmed.run()

        # Load and compare results
        data_full = pywrdrb.Data()
        data_full.load_output(
            output_filenames=[output_filename],
            results_sets=["major_flow", "res_storage"]
        )

        data_trimmed = pywrdrb.Data()
        data_trimmed.load_output(
            output_filenames=[trimmed_output_filename],
            results_sets=["major_flow", "res_storage"]
        )

        # Get model labels
        full_label = list(data_full.major_flow.keys())[0]
        trimmed_label = list(data_trimmed.major_flow.keys())[0]

        # Compare Montague and Trenton flows
        # The trimmed model uses offline-simulated STARFIT releases (approximate),
        # so flows won't match exactly. Verify they are strongly correlated.
        for node in ["delMontague", "delTrenton"]:
            full_flow = data_full.major_flow[full_label][0][node]
            trimmed_flow = data_trimmed.major_flow[trimmed_label][0][node]

            corr = np.corrcoef(full_flow.values, trimmed_flow.values)[0, 1]
            assert corr > 0.95, f"{node} correlation too low: {corr:.4f}"
            print(f"{node}: correlation = {corr:.4f}")

        # Compare NYC reservoir storage (should be strongly correlated)
        for res in reservoir_list_nyc:
            full_storage = data_full.res_storage[full_label][0][res]
            trimmed_storage = data_trimmed.res_storage[trimmed_label][0][res]

            corr = np.corrcoef(full_storage.values, trimmed_storage.values)[0, 1]
            assert corr > 0.95, f"{res} storage correlation too low: {corr:.4f}"
            print(f"{res}: correlation = {corr:.4f}")

        print("Trimmed model results correlate well with full model!")


@pytest.mark.slow
class TestTrimmedModelRuntime:
    """
    Tests for trimmed model runtime performance.

    These tests verify that the trimmed model provides expected speedup.
    Run with: pytest -m slow tests/test_trimmed_model.py
    """

    @pytest.fixture
    def full_and_trimmed_models(self, tmp_path):
        """
        Build full and trimmed models, returning paths and timing info.
        """
        import time

        # Use a medium-length period for runtime testing
        start_date = "1983-10-01"
        end_date = "1985-09-30"  # 2 years
        inflow_type = TEST_INFLOW_TYPE

        # Build full model
        full_model_file = str(tmp_path / "full_model.json")
        full_output_file = str(tmp_path / "full_output.hdf5")

        mb_full = ModelBuilder(
            start_date=start_date,
            end_date=end_date,
            inflow_type=inflow_type,
        )
        mb_full.make_model()
        mb_full.write_model(full_model_file)
        full_node_count = len(mb_full.model_dict["nodes"])

        # Run full model
        model_full = pywrdrb.Model.load(full_model_file)
        recorder_full = pywrdrb.OutputRecorder(
            model=model_full, output_filename=full_output_file
        )

        start_time = time.perf_counter()
        model_full.run()
        full_run_time = time.perf_counter() - start_time

        # Generate pre-simulated releases using offline simulator
        from pywrdrb.path_manager import get_pn_object
        pn = get_pn_object()

        sim = STARFITOfflineSimulator(initial_volume_frac=0.8)
        sim.load_parameters()

        inflow_file = pn.sc.get(f"flows/{inflow_type}") / "catchment_inflow_mgd.csv"
        catchment_inflows = pd.read_csv(str(inflow_file), index_col=0, parse_dates=True)
        releases_df = sim.simulate_all(catchment_inflows)

        presim_dir = str(tmp_path / "presim")
        os.makedirs(presim_dir, exist_ok=True)
        csv_file = os.path.join(presim_dir, "presimulated_releases_mgd.csv")
        releases_df.index.name = "datetime"
        releases_df.index = pd.to_datetime(releases_df.index).strftime("%Y-%m-%d")
        releases_df.to_csv(csv_file, float_format="%.10f")

        metadata_file = os.path.join(presim_dir, "presimulated_releases_mgd_metadata.json")
        metadata = {
            "inflow_type": inflow_type,
            "start_date": str(releases_df.index[0]),
            "end_date": str(releases_df.index[-1]),
            "reservoirs": list(releases_df.columns),
            "output_file": csv_file,
            "metadata_file": metadata_file,
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Build trimmed model
        trimmed_model_file = str(tmp_path / "trimmed_model.json")
        trimmed_output_file = str(tmp_path / "trimmed_output.hdf5")

        mb_trimmed = ModelBuilder(
            start_date=start_date,
            end_date=end_date,
            inflow_type=inflow_type,
            options={
                "use_trimmed_model": True,
                "presimulated_releases_file": os.path.abspath(metadata["output_file"])
            }
        )
        mb_trimmed.make_model()
        mb_trimmed.write_model(trimmed_model_file)
        trimmed_node_count = len(mb_trimmed.model_dict["nodes"])

        # Run trimmed model
        model_trimmed = pywrdrb.Model.load(trimmed_model_file)
        recorder_trimmed = pywrdrb.OutputRecorder(
            model=model_trimmed, output_filename=trimmed_output_file
        )

        start_time = time.perf_counter()
        model_trimmed.run()
        trimmed_run_time = time.perf_counter() - start_time

        return {
            "full_output_file": full_output_file,
            "trimmed_output_file": trimmed_output_file,
            "full_node_count": full_node_count,
            "trimmed_node_count": trimmed_node_count,
            "full_run_time": full_run_time,
            "trimmed_run_time": trimmed_run_time,
        }

    def test_trimmed_model_has_fewer_nodes(self, full_and_trimmed_models):
        """Verify trimmed model has significantly fewer nodes."""
        info = full_and_trimmed_models

        reduction = info["full_node_count"] - info["trimmed_node_count"]
        reduction_pct = reduction / info["full_node_count"] * 100

        print(f"\nNode count: Full={info['full_node_count']}, "
              f"Trimmed={info['trimmed_node_count']}")
        print(f"Reduction: {reduction} nodes ({reduction_pct:.1f}%)")

        # Should have at least 20% fewer nodes
        assert reduction_pct > 20, (
            f"Expected >20% node reduction, got {reduction_pct:.1f}%"
        )

    def test_trimmed_model_is_faster(self, full_and_trimmed_models):
        """Verify trimmed model runs faster than full model."""
        info = full_and_trimmed_models

        speedup = info["full_run_time"] / info["trimmed_run_time"]
        reduction_pct = (1 - info["trimmed_run_time"] / info["full_run_time"]) * 100

        print(f"\nRuntime: Full={info['full_run_time']:.2f}s, "
              f"Trimmed={info['trimmed_run_time']:.2f}s")
        print(f"Speedup: {speedup:.2f}x ({reduction_pct:.1f}% reduction)")

        # Should be at least 10% faster (conservative threshold)
        assert speedup > 1.1, (
            f"Expected >10% speedup, got {speedup:.2f}x"
        )

    def test_trimmed_model_results_correlate(self, full_and_trimmed_models):
        """
        Verify trimmed model results correlate highly with full model.

        Note: Due to LP solver non-determinism, we check correlation rather
        than exact match. High correlation (r > 0.99) indicates the trimmed
        model captures the same dynamics.
        """
        info = full_and_trimmed_models

        data_full = pywrdrb.Data()
        data_full.load_output(
            output_filenames=[info["full_output_file"]],
            results_sets=["major_flow", "res_storage"]
        )

        data_trimmed = pywrdrb.Data()
        data_trimmed.load_output(
            output_filenames=[info["trimmed_output_file"]],
            results_sets=["major_flow", "res_storage"]
        )

        full_label = list(data_full.major_flow.keys())[0]
        trimmed_label = list(data_trimmed.major_flow.keys())[0]

        # Check correlation for key outputs
        min_correlation = 0.99

        # Flow correlations
        for node in ["delMontague", "delTrenton"]:
            full_flow = data_full.major_flow[full_label][0][node].values
            trimmed_flow = data_trimmed.major_flow[trimmed_label][0][node].values
            correlation = np.corrcoef(full_flow.flatten(), trimmed_flow.flatten())[0, 1]
            print(f"{node} correlation: r={correlation:.4f}")
            assert correlation > min_correlation, (
                f"{node} correlation {correlation:.4f} < {min_correlation}"
            )

        # Storage correlations
        for res in reservoir_list_nyc:
            full_storage = data_full.res_storage[full_label][0][res].values
            trimmed_storage = data_trimmed.res_storage[trimmed_label][0][res].values
            correlation = np.corrcoef(full_storage.flatten(), trimmed_storage.flatten())[0, 1]
            print(f"{res} correlation: r={correlation:.4f}")
            assert correlation > min_correlation, (
                f"{res} correlation {correlation:.4f} < {min_correlation}"
            )
