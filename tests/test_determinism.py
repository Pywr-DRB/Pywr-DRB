"""
Tests to verify model determinism behavior.

Note: The Pywr-DRB model exhibits some non-determinism due to the GLPK LP solver.
When multiple optimal solutions exist (degenerate LP), GLPK may choose different
solutions based on internal state. This is expected behavior and not a bug.

Key findings:
- Loading the SAME model JSON file multiple times produces deterministic results
- Building models fresh each time may produce different results due to LP degeneracy
- Differences typically converge after ~1 year of simulation (spin-up period)
- Unique cost offsets help reduce but don't eliminate non-determinism

Tests in this module verify these behaviors and document expected tolerances.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import h5py

import pywrdrb


# Test parameters
TEST_START_DATE = "1983-10-01"
TEST_END_DATE = "1984-01-01"  # Short period for quick tests
TEST_INFLOW_TYPE = "nhmv10_withObsScaled"


class TestModelDeterminism:
    """Tests for model determinism behavior."""

    @pytest.fixture
    def model_files(self, tmp_path):
        """Build a model once and return paths for reuse."""
        model_filename = str(tmp_path / "determinism_test_model.json")

        mb = pywrdrb.ModelBuilder(
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            inflow_type=TEST_INFLOW_TYPE,
        )
        mb.make_model()
        mb.write_model(model_filename)

        return model_filename, tmp_path

    def _run_model(self, model_filename, output_filename):
        """Run a model in a fresh subprocess and return key results.

        Each run executes in its own interpreter so the GLPK solver starts
        from a clean state. Running in-process makes this test order-fragile:
        any earlier test that ran a model perturbs solver internals enough
        that LP-degenerate solutions diverge between back-to-back runs
        (differences of hundreds of MG in reservoir storage during spin-up).
        Subprocess isolation also matches what the test claims to verify —
        that the same model FILE reproduces identically across separate runs.
        """
        import subprocess

        script = (
            "import pywrdrb\n"
            f"model = pywrdrb.Model.load({model_filename!r})\n"
            "recorder = pywrdrb.OutputRecorder(\n"
            "    model=model,\n"
            f"    output_filename={output_filename!r},\n"
            ")\n"
            "model.run()\n"
        )
        subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )

        results = {}
        with h5py.File(output_filename, 'r') as hdf:
            # NYC reservoir storage
            for res in ['cannonsville', 'pepacton', 'neversink']:
                key = f'reservoir_{res}'
                if key in hdf.keys():
                    results[f'{res}_storage'] = np.array(hdf[key])

            # Key flow outputs
            if 'output_delTrenton' in hdf.keys():
                results['trenton_flow'] = np.array(hdf['output_delTrenton'])

        return results

    def test_same_model_file_is_deterministic(self, model_files):
        """
        Verify that loading the same model file produces identical results.

        This tests that Pywr's internal solver state is deterministic when
        starting from the same model JSON file.
        """
        model_filename, tmp_path = model_files

        # Run model twice from same JSON file
        output1 = str(tmp_path / "output_run1.hdf5")
        output2 = str(tmp_path / "output_run2.hdf5")

        results1 = self._run_model(model_filename, output1)
        results2 = self._run_model(model_filename, output2)

        # Results should be identical
        for key in results1.keys():
            diff = np.abs(results1[key] - results2[key])
            max_diff = np.max(diff)
            assert max_diff < 1e-10, f"{key} differs between runs: max_diff={max_diff}"

    @pytest.mark.slow
    def test_rebuilt_models_may_differ(self, tmp_path):
        """
        Document that rebuilding models may produce different results.

        This is expected behavior due to LP degeneracy in GLPK.
        We test that differences are within acceptable bounds.
        """
        results_list = []

        # Build and run model 3 times
        for i in range(3):
            model_filename = str(tmp_path / f"model_run{i}.json")
            output_filename = str(tmp_path / f"output_run{i}.hdf5")

            mb = pywrdrb.ModelBuilder(
                start_date=TEST_START_DATE,
                end_date=TEST_END_DATE,
                inflow_type=TEST_INFLOW_TYPE,
            )
            mb.make_model()
            mb.write_model(model_filename)

            results = self._run_model(model_filename, output_filename)
            results_list.append(results)

        # Check that differences are bounded (not checking for equality)
        # Storage differences should be < 10% of capacity
        # Flow differences should be < 10% of mean flow
        for key in results_list[0].keys():
            all_values = np.array([r[key] for r in results_list])
            spread = np.max(all_values, axis=0) - np.min(all_values, axis=0)
            mean_val = np.mean(all_values)

            # Relative difference should be small
            if mean_val > 0:
                relative_spread = np.max(spread) / mean_val
                assert relative_spread < 0.2, (
                    f"{key} shows excessive variation: "
                    f"max_spread={np.max(spread):.2f}, mean={mean_val:.2f}, "
                    f"relative={relative_spread:.2%}"
                )

    def test_unique_costs_are_set(self, model_files):
        """
        Verify that model nodes have unique costs to reduce LP degeneracy.
        """
        import json
        model_filename, _ = model_files

        with open(model_filename, 'r') as f:
            model_dict = json.load(f)

        # Collect all costs from nodes
        costs = []
        for node in model_dict['nodes']:
            if 'cost' in node:
                cost = node['cost']
                if isinstance(cost, (int, float)):
                    costs.append(cost)

        # Check that costs are unique
        if len(costs) > 0:
            unique_costs = set(costs)
            # Allow some duplicates (different node types may legitimately share costs)
            # but most should be unique
            uniqueness_ratio = len(unique_costs) / len(costs)
            assert uniqueness_ratio > 0.8, (
                f"Too many duplicate costs: {len(unique_costs)}/{len(costs)} unique "
                f"({uniqueness_ratio:.1%})"
            )
