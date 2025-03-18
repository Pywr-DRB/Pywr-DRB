import pytest

# Shared tmp path for all test functions
@pytest.fixture(scope="session")
def shared_tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


# Inflow types to be tested
# The full workflow make_model -> run -> load will be tested for each inflow type
@pytest.fixture(params=["nhmv10",
                        "nhmv10_withObsScaled", 
                        "nwmv21",
                        "nwmv21_withObsScaled",
                        ])
def test_inflow_type(request):
    return request.param