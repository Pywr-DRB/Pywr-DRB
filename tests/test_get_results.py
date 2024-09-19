import numpy as np

from pywrdrb.post.get_results import get_pywrdrb_results, get_pywr_results
from pywrdrb.utils.directories import output_dir


def test_get_pywrdrb_results_equals_get_pywr_results():
    """
    Tests that get_pywrdrb_results() and get_pywr_results() yield the
    same values when retrieving data from a model output file.

    get_pywr_results() is depreciated, but get_pywrdrb_results()
    is designed to be backwards compatible with get_pywr_results().
    """

    model = "nhmv10"
    test_results_sets = ["reservoir_downstream_gage", "res_release"]
    datetime_a = None
    datetime_b = None
    for results_set in test_results_sets:
        results_a, datetime_a = get_pywrdrb_results(
            output_dir,
            model,
            results_set=results_set,
            scenarios=[0],
            datetime_index=datetime_a,
            units=None,
        )

        results_b, datetime_b = get_pywr_results(
            output_dir,
            model,
            results_set=results_set,
            scenario=0,
            datetime_index=datetime_b,
            units=None,
        )

        # Check that the value of results are the same
        err_msg = f"Results from get_pywrdrb_results and get_pywr_results"
        err_msg += f" are not the same for {results_set}"
        assert np.allclose(results_a[0].values, results_b.values), err_msg

        # Check that pandas datetime indices are the same
        err_msg = f"dateime results from get_pywrdrb_results and get_pywr_results"
        err_msg += f" are not the same for {results_set}"
        assert datetime_a.equals(datetime_b), err_msg
