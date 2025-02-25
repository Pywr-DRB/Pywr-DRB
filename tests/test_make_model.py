import pytest

from pywrdrb import ModelBuilder
from pywrdrb.utils.dates import model_date_ranges


def test_make_model_with_default_options(test_inflow_type, shared_tmp_path):
    """
    Test if the ModelBuilder class can create a model with the default options.
    """
    start_date, end_date = model_date_ranges[test_inflow_type]

    try:
        mb = ModelBuilder(test_inflow_type, start_date, end_date)
        mb.make_model()
    except Exception as e:
        pytest.fail(f"ModelBuilder.make_model() raised an exception: {e}")
        
    model_dict = mb.model_dict
    assert isinstance(
        model_dict, dict
    ), f"Expected ModelBuilder.model_dict to be type dict but got {type(model_dict)}."

    ## test write_model
    # NOTE: Doo not change the filename! It is used in test_pywrdrb_simulation()
    model_filename = shared_tmp_path / f"{test_inflow_type}_model.json"
    mb.write_model(str(model_filename))

    assert (
        model_filename.exists()
    ), f"Expected file {model_filename} to exist but it was not found."

    return
