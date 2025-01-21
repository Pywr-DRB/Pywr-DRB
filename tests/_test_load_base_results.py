import pytest
import pandas as pd

from pywrdrb import Output
from pywrdrb.utils.results_sets import base_results_set_opts


@pytest.mark.parametrize(
    "model, results_set",
    [
        ("nhmv10", "reservoir_downstream_gage"),
        ("nhmv10", "major_flow"),
        ("nwmv21", "reservoir_downstream_gage"),
        ("nwmv21", "major_flow"),
        ("obs", "reservoir_downstream_gage"),
        ("obs", "major_flow"),
    ],
)
def test_output_loader_with_valid_base_results(model, results_set):
    """
    Check if the pywrdrb.Output class properly loads the base results
    for the given model and results_set.

    Checks that the Output.load() function:
    1. stores results_set as an attribute in the Output object, and that it is a dictionary
    2. stores the model as a key in the results_set attribute dictionary
    3. Returns data as a pd.DataFrame for the 0th scenario
    """
    # setup output loader
    output = Output(
        models=[model],
        results_sets=[results_set],
        base_results=True,
        print_status=False,
    )
    output.load()

    ## Access the data using format:
    # output.results_set[model][0]

    # Check if `results_set` attribute exists in `output`
    assert hasattr(
        output, results_set
    ), f"Expected attribute {results_set} in output but not found."

    # Check if `model` exists within the attribute and is a dictionary key
    result_data = getattr(output, results_set)
    assert (
        model in result_data
    ), f"Expected '{model}' key within OutputLoader.{results_set} but not found."

    ## Check formatting of OutputLoader
    # if the first element is a list and contains a pd.DataFrame
    assert isinstance(
        result_data[model], dict
    ), f"Expected dict in output.{results_set}[{model}] but got {type(result_data[model])}."
    assert isinstance(
        result_data[model][0], pd.DataFrame
    ), f"Expected pd.DataFrame at output.{results_set}[{model}][0] but got {type(result_data[model][0])}."
    return
