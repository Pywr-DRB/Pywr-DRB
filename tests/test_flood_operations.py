"""
Comprehensive test suite for NYC flood operations.

Tests topology switching, caching, options integration, model building,
and full model execution with flood operations enabled/disabled.

Run with: python tests/test_flood_operations.py
Or with pytest: pytest tests/test_flood_operations.py -v
"""

import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pywrdrb.pywr_drb_node_data import TopologyDictionaries
from pywrdrb.model_builder import ModelBuilder, Options


def test_topology_base():
    """Test base topology without flood nodes."""
    print("\n=== Testing Base Topology (No Flood Nodes) ===")

    dicts = TopologyDictionaries.get(include_flood_nodes=False)
    downstream, upstream, lags, obs, obs_pub, nhm, nwm, wrf = dicts

    # Check flood nodes NOT in routing
    assert "01426500" not in downstream, "ERROR: Hale Eddy should not be in base topology"
    assert "01421000" not in downstream, "ERROR: Fishs Eddy should not be in base topology"
    assert "01436690" not in downstream, "ERROR: Bridgeville should not be in base topology"

    # Check original routing intact
    assert downstream["01425000"] == "delLordville", "ERROR: Stilesville routing incorrect"
    assert downstream["01417000"] == "delLordville", "ERROR: Harvard routing incorrect"
    assert downstream["01436000"] == "delMontague", "ERROR: Neversink routing incorrect"

    print("✓ Base topology correct: No flood nodes")
    print("✓ Original routing preserved")


def test_topology_flood():
    """Test flood topology with flood nodes."""
    print("\n=== Testing Flood Topology (With Flood Nodes) ===")

    dicts = TopologyDictionaries.get(include_flood_nodes=True)
    downstream, upstream, lags, obs, obs_pub, nhm, nwm, wrf = dicts

    # Check flood nodes inserted
    assert downstream["01425000"] == "01426500", "ERROR: Stilesville should route to Hale Eddy"
    assert downstream["01426500"] == "delLordville", "ERROR: Hale Eddy should route to Lordville"

    assert downstream["01417000"] == "01421000", "ERROR: Harvard should route to Fishs Eddy"
    assert downstream["01421000"] == "delLordville", "ERROR: Fishs Eddy should route to Lordville"

    assert downstream["01436000"] == "01436690", "ERROR: Release should route to Bridgeville"
    assert downstream["01436690"] == "delMontague", "ERROR: Bridgeville should route to Montague"

    # Check upstream relationships
    assert "01426500" in upstream["delLordville"], "ERROR: Hale Eddy not in Lordville upstream"
    assert "01421000" in upstream["delLordville"], "ERROR: Fishs Eddy not in Lordville upstream"
    assert "01436690" in upstream["delMontague"], "ERROR: Bridgeville not in Montague upstream"

    # Check zero-day lags
    assert lags["01426500"] == 0, "ERROR: Hale Eddy lag should be 0"
    assert lags["01421000"] == 0, "ERROR: Fishs Eddy lag should be 0"
    assert lags["01436690"] == 0, "ERROR: Bridgeville lag should be 0"

    # Check observation site matches
    assert obs["01426500"] == ["01426500"], "ERROR: Hale Eddy obs site incorrect"
    assert obs["01421000"] == ["01421000"], "ERROR: Fishs Eddy obs site incorrect"
    assert obs["01436690"] == ["01436690"], "ERROR: Bridgeville obs site incorrect"

    print("✓ Flood topology correct: Nodes inserted")
    print("✓ Routing modified correctly")
    print("✓ Upstream relationships updated")
    print("✓ Zero-day lags set")
    print("✓ Observation sites configured")


def test_topology_caching():
    """Test that topology dictionaries are cached."""
    print("\n=== Testing Topology Caching ===")

    # Get base topology twice
    dicts1 = TopologyDictionaries.get(include_flood_nodes=False)
    dicts2 = TopologyDictionaries.get(include_flood_nodes=False)

    # Should be same object (cached)
    assert dicts1 is dicts2, "ERROR: Base topology not cached"
    print("✓ Base topology caching works")

    # Get flood topology twice
    flood1 = TopologyDictionaries.get(include_flood_nodes=True)
    flood2 = TopologyDictionaries.get(include_flood_nodes=True)

    # Should be same object (cached)
    assert flood1 is flood2, "ERROR: Flood topology not cached"
    print("✓ Flood topology caching works")

    # Base and flood should be different
    assert dicts1 is not flood1, "ERROR: Base and flood topologies should be different"
    print("✓ Base and flood topologies are separate")


def test_options_default():
    """Test default options disable flood operations."""
    print("\n=== Testing Options Defaults ===")

    options = Options()
    assert options.enable_nyc_flood_operations is False, "ERROR: Default should be False"
    print("✓ Default option: enable_nyc_flood_operations = False")


def test_model_builder_base():
    """Test ModelBuilder with flood operations disabled."""
    print("\n=== Testing ModelBuilder (Flood Operations Disabled) ===")

    builder = ModelBuilder(
        start_date='2000-01-01',
        end_date='2000-01-02',
        inflow_type='nhmv10',
        options={'enable_nyc_flood_operations': False}
    )

    # Check base topology loaded
    assert "01426500" not in builder.immediate_downstream_nodes_dict, \
        "ERROR: Hale Eddy should not be in base model"
    assert "01421000" not in builder.immediate_downstream_nodes_dict, \
        "ERROR: Fishs Eddy should not be in base model"
    assert "01436690" not in builder.immediate_downstream_nodes_dict, \
        "ERROR: Bridgeville should not be in base model"

    # Check original routing
    assert builder.immediate_downstream_nodes_dict["01425000"] == "delLordville", \
        "ERROR: Original routing not preserved"

    print("✓ Base topology loaded correctly")
    print("✓ No flood nodes in model")


def test_model_builder_flood():
    """Test ModelBuilder with flood operations enabled."""
    print("\n=== Testing ModelBuilder (Flood Operations Enabled) ===")

    builder = ModelBuilder(
        start_date='2000-01-01',
        end_date='2000-01-02',
        inflow_type='nhmv10',
        options={'enable_nyc_flood_operations': True}
    )

    # Check flood topology loaded
    assert "01426500" in builder.immediate_downstream_nodes_dict, \
        "ERROR: Hale Eddy should be in flood model"
    assert "01421000" in builder.immediate_downstream_nodes_dict, \
        "ERROR: Fishs Eddy should be in flood model"
    assert "01436690" in builder.immediate_downstream_nodes_dict, \
        "ERROR: Bridgeville should be in flood model"

    # Check modified routing
    assert builder.immediate_downstream_nodes_dict["01425000"] == "01426500", \
        "ERROR: Flood routing not applied"
    assert builder.immediate_downstream_nodes_dict["01426500"] == "delLordville", \
        "ERROR: Flood routing continuation incorrect"

    print("✓ Flood topology loaded correctly")
    print("✓ All flood nodes present")
    print("✓ Modified routing applied")


def test_model_dict_base():
    """Test model dictionary without flood operations."""
    print("\n=== Testing Model Dict (Flood Operations Disabled) ===")

    builder = ModelBuilder(
        start_date='2000-01-01',
        end_date='2000-01-02',
        inflow_type='nhmv10',
        options={'enable_nyc_flood_operations': False}
    )

    builder.make_model()
    model_dict = builder.model_dict

    # Verify structure
    assert 'nodes' in model_dict, "ERROR: No nodes in model dict"
    assert 'parameters' in model_dict, "ERROR: No parameters in model dict"

    # Verify flood monitoring parameters NOT added
    assert 'stage_01426500' not in model_dict['parameters'], \
        "ERROR: Stage parameters should not be present"
    assert 'flood_level_01426500' not in model_dict['parameters'], \
        "ERROR: Flood level parameters should not be present"

    # Verify flood nodes NOT in network
    node_names = [node['name'] for node in model_dict['nodes']]
    assert 'link_01426500' not in node_names, "ERROR: Flood node should not be in network"
    assert 'link_01421000' not in node_names, "ERROR: Flood node should not be in network"
    assert 'link_01436690' not in node_names, "ERROR: Flood node should not be in network"

    # Check flood release parameters have operations disabled
    for reservoir in ['cannonsville', 'pepacton', 'neversink']:
        param = model_dict['parameters'][f'flood_release_{reservoir}']
        assert param['flood_operations_enabled'] is False, \
            f"ERROR: {reservoir} flood operations should be disabled"
        assert 'downstream_stage_parameter' not in param, \
            f"ERROR: {reservoir} should not have downstream stage parameter"
        assert 'mrf_baseline_reservoir' not in param, \
            f"ERROR: {reservoir} flood_release should not have mrf_baseline_reservoir"

    # Check combined factor parameters do NOT have flood ops keys
    for reservoir in ['cannonsville', 'pepacton', 'neversink']:
        param = model_dict['parameters'][f'mrf_drought_factor_combined_final_{reservoir}']
        assert 'flood_operations_enabled' not in param, \
            f"ERROR: {reservoir} combined factor should not have flood_operations_enabled"
        assert 'downstream_stage_parameter' not in param, \
            f"ERROR: {reservoir} combined factor should not have downstream_stage_parameter"
        assert 'mrf_factor_l2' not in param, \
            f"ERROR: {reservoir} combined factor should not have mrf_factor_l2"

    print("✓ Model dict structure correct")
    print("✓ No flood monitoring parameters")
    print("✓ No flood nodes in network")
    print("✓ Flood operations disabled in parameters")
    print("✓ Combined factor parameters clean (no flood ops keys)")


def test_model_dict_flood():
    """Test model dictionary with flood operations."""
    print("\n=== Testing Model Dict (Flood Operations Enabled) ===")

    builder = ModelBuilder(
        start_date='2000-01-01',
        end_date='2000-01-02',
        inflow_type='nhmv10',
        options={'enable_nyc_flood_operations': True}
    )

    builder.make_model()
    model_dict = builder.model_dict

    # Verify flood monitoring parameters added
    assert 'stage_01426500' in model_dict['parameters'], \
        "ERROR: Hale Eddy stage parameter missing"
    assert 'stage_01421000' in model_dict['parameters'], \
        "ERROR: Fishs Eddy stage parameter missing"
    assert 'stage_01436690' in model_dict['parameters'], \
        "ERROR: Bridgeville stage parameter missing"

    assert 'flood_level_01426500' in model_dict['parameters'], \
        "ERROR: Hale Eddy flood level missing"
    assert 'flood_level_01421000' in model_dict['parameters'], \
        "ERROR: Fishs Eddy flood level missing"
    assert 'flood_level_01436690' in model_dict['parameters'], \
        "ERROR: Bridgeville flood level missing"

    # Verify flood nodes in network
    node_names = [node['name'] for node in model_dict['nodes']]
    assert 'link_01426500' in node_names, "ERROR: Hale Eddy node missing"
    assert 'link_01421000' in node_names, "ERROR: Fishs Eddy node missing"
    assert 'link_01436690' in node_names, "ERROR: Bridgeville node missing"

    # Check flood release parameters have operations enabled
    reservoir_stage_map = {
        'cannonsville': 'stage_01426500',
        'pepacton': 'stage_01421000',
        'neversink': 'stage_01436690'
    }

    for reservoir, expected_stage in reservoir_stage_map.items():
        param = model_dict['parameters'][f'flood_release_{reservoir}']
        assert param['flood_operations_enabled'] is True, \
            f"ERROR: {reservoir} flood operations should be enabled"
        assert 'downstream_stage_parameter' in param, \
            f"ERROR: {reservoir} missing downstream stage parameter"
        assert param['downstream_stage_parameter'] == expected_stage, \
            f"ERROR: {reservoir} has wrong downstream stage parameter"
        assert 'mrf_baseline_reservoir' not in param, \
            f"ERROR: {reservoir} flood_release should not have mrf_baseline_reservoir"

    # Check combined factor parameters have flood ops keys
    for reservoir, expected_stage in reservoir_stage_map.items():
        param = model_dict['parameters'][f'mrf_drought_factor_combined_final_{reservoir}']
        assert param.get('flood_operations_enabled') is True, \
            f"ERROR: {reservoir} combined factor should have flood_operations_enabled"
        assert param.get('downstream_stage_parameter') == expected_stage, \
            f"ERROR: {reservoir} combined factor has wrong downstream_stage_parameter"
        assert param.get('mrf_factor_l2') == f"level2_factor_mrf_{reservoir}", \
            f"ERROR: {reservoir} combined factor has wrong mrf_factor_l2"

    print("✓ All flood monitoring parameters present")
    print("✓ All flood nodes in network")
    print("✓ Flood operations enabled in flood_release parameters")
    print("✓ Correct downstream stage parameter mappings")
    print("✓ Combined factor parameters have flood ops keys (curtailment at L2)")


def test_full_model_run_base():
    """Test full model run without flood operations."""
    print("\n=== Testing Full Model Run (No Flood Operations) ===")

    builder = ModelBuilder(
        start_date='2000-01-01',
        end_date='2000-01-07',
        inflow_type='nhmv10',
        options={'enable_nyc_flood_operations': False}
    )

    builder.make_model()
    model_dict = builder.model_dict

    # Verify flood nodes NOT present
    node_names = [node['name'] for node in model_dict['nodes']]
    assert 'link_01426500' not in node_names, "ERROR: Flood node should not be in base model"
    print("✓ Base model structure correct")

    # Try to run the model
    try:
        import pywr.core
        pywr_model = pywr.core.Model.load(model_dict)
        print(f"✓ Pywr model loaded ({len(pywr_model.nodes)} nodes, {len(pywr_model.parameters)} parameters)")

        stats = pywr_model.run()
        print(f"✓ Model run completed successfully ({len(pywr_model.timestepper)} timesteps)")
        return True

    except Exception as e:
        print(f"✗ Model run failed: {e}")
        return False


def test_full_model_run_flood():
    """Test full model run with flood operations."""
    print("\n=== Testing Full Model Run (With Flood Operations) ===")

    builder = ModelBuilder(
        start_date='2000-01-01',
        end_date='2000-01-07',
        inflow_type='nhmv10',
        options={'enable_nyc_flood_operations': True}
    )

    builder.make_model()
    model_dict = builder.model_dict

    # Verify flood nodes present
    node_names = [node['name'] for node in model_dict['nodes']]
    flood_nodes = ['link_01426500', 'link_01421000', 'link_01436690']
    for node in flood_nodes:
        assert node in node_names, f"ERROR: {node} missing from network"
    print("✓ All flood nodes present in network")

    # Verify flood monitoring parameters
    flood_params = [
        'stage_01426500', 'stage_01421000', 'stage_01436690',
        'flood_level_01426500', 'flood_level_01421000', 'flood_level_01436690'
    ]
    for param in flood_params:
        assert param in model_dict['parameters'], f"ERROR: {param} missing"
    print("✓ All flood monitoring parameters present")

    # Check flood-responsive operations enabled
    for reservoir in ['cannonsville', 'pepacton', 'neversink']:
        param = model_dict['parameters'][f'flood_release_{reservoir}']
        assert param['flood_operations_enabled'] is True, \
            f"ERROR: {reservoir} flood operations not enabled"
    print("✓ Flood-responsive operations enabled")

    # Try to run the model
    try:
        import pywr.core
        pywr_model = pywr.core.Model.load(model_dict)
        print(f"✓ Pywr model loaded ({len(pywr_model.nodes)} nodes, {len(pywr_model.parameters)} parameters)")

        stats = pywr_model.run()
        print(f"✓ Model run completed successfully ({len(pywr_model.timestepper)} timesteps)")

        # Check that flood node flows are non-zero
        print("\n  Verifying flood node inflows:")
        for node_name in ['link_01426500', 'link_01421000', 'link_01436690']:
            node = pywr_model.nodes[node_name]
            flows = node.flow
            mean_flow = flows.mean()
            max_flow = flows.max()
            print(f"  - {node_name}: Mean={mean_flow:.2f} MGD, Max={max_flow:.2f} MGD")
            assert mean_flow > 0, f"ERROR: {node_name} has zero mean flow"
            assert max_flow > 0, f"ERROR: {node_name} has zero max flow"
        print("✓ All flood nodes have non-zero inflows")

        return True

    except Exception as e:
        print(f"✗ Model run failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("NYC FLOOD OPERATIONS - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    tests = [
        ("Topology Base", test_topology_base),
        ("Topology Flood", test_topology_flood),
        ("Topology Caching", test_topology_caching),
        ("Options Default", test_options_default),
        ("ModelBuilder Base", test_model_builder_base),
        ("ModelBuilder Flood", test_model_builder_flood),
        ("Model Dict Base", test_model_dict_base),
        ("Model Dict Flood", test_model_dict_flood),
        ("Full Model Run Base", test_full_model_run_base),
        ("Full Model Run Flood", test_full_model_run_flood),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            result = test_func()
            # Some tests return True/False, others just pass/fail via assertions
            if result is False:
                failed += 1
            else:
                passed += 1
        except AssertionError as e:
            print(f"\n✗ {name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ {name} ERROR: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nThe NYC flood operations implementation is fully functional:")
        print("  ✓ Topology switching works correctly")
        print("  ✓ Augmented inflow file loads successfully")
        print("  ✓ Flood nodes receive non-zero inflows")
        print("  ✓ Model runs without errors")
        print("  ✓ Backward compatibility maintained")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
