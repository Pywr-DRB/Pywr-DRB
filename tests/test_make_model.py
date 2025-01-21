import pytest

from pywr.model import Model
from pywrdrb.parameters import *
from pywrdrb import ModelBuilder
from pywrdrb.utils.dates import model_date_ranges


@pytest.mark.parametrize(
    "inflow_type",
    [
        ("nhmv10"),
        ("nwmv21"),
        ("nhmv10_withObsScaled"),
        ("nwmv21_withObsScaled"),
    ],
)

def test_make_model_with_default_options(inflow_type, tmp_path):
    """
    Test if the ModelBuilder class can create a model with the default options.
    """
    start_date, end_date = model_date_ranges[inflow_type]

    mb = ModelBuilder(inflow_type, start_date, end_date)
    mb.make_model()

    model_dict = mb.model_dict
    assert isinstance(
        model_dict, dict
    ), f"Expected ModelBuilder.model_dict to be type dict but got {type(model_dict)}."

    ## test write_model
    # tmp_path is a pytest fixture that creates a temporary directory
    model_filename = tmp_path / f"drb_model_full_{inflow_type}.json"
    mb.write_model(str(model_filename))

    assert (
        model_filename.exists()
    ), f"Expected file {model_filename} to exist but it was not found."

    # test Model.load()
    model = Model.load(str(model_filename))
    assert isinstance(
        model, Model
    ), f"Expected Model.load() to return a pywr.Model object but got {type(model)}."
    return
