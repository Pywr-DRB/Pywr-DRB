"""
Stress the v2.2beta options: error handling, malformed inputs, and short runs.

Checks that invalid configurations fail loudly with the right exception, and
that the flow_prediction_mode options complete a short simulation.
Prints a PASS/FAIL table; independent of the other scripts.

Usage:
    python 03_stress_tests.py
"""
import pandas as pd

import pywrdrb
from pywrdrb.path_manager import get_pn_object
from pywrdrb.pre import STARFITOfflineSimulator

from utils import INFLOW_TYPE, MODEL_DIR, OUTPUT_DIR, make_dirs, run_pywrdrb

MINI_START = "2005-06-01"
MINI_END = "2005-06-30"

pn = get_pn_object()
results = []


def check(name, fn, expect=None):
    """Run fn; PASS if it raises expect (or completes cleanly when expect is None)."""
    try:
        fn()
        if expect is None:
            results.append((name, "PASS", "completed"))
        else:
            results.append((name, "FAIL", f"expected {expect.__name__}, no error raised"))
    except Exception as e:
        if expect is not None and isinstance(e, expect):
            results.append((name, "PASS", f"raised {type(e).__name__}"))
        else:
            results.append((name, "FAIL", f"raised {type(e).__name__}: {e}"))


def build(options):
    mb = pywrdrb.ModelBuilder(
        inflow_type=INFLOW_TYPE, start_date=MINI_START, end_date=MINI_END, options=options
    )
    mb.make_model()
    return mb


def missing_row_csv():
    params = pd.read_csv(pn.operational_constants.get_str("istarf_conus.csv"), index_col=0)
    params = params.drop("prompton")
    path = OUTPUT_DIR / "stress_params_missing_row.csv"
    params.index.name = "reservoir"
    params.to_csv(path)
    return path


def data_round_trip():
    h5_path = OUTPUT_DIR / "stress_mini_regression_disagg.hdf5"
    export_path = OUTPUT_DIR / "stress_roundtrip_export.hdf5"

    data = pywrdrb.Data(print_status=False)
    data.load_output(
        output_filenames=[str(h5_path)], results_sets=["res_storage", "major_flow"]
    )
    label = h5_path.stem
    before = data.res_storage[label][0]

    data.export(str(export_path))
    data2 = pywrdrb.Data(print_status=False)
    data2.load_from_export(str(export_path))
    after = data2.res_storage[label][0]

    pd.testing.assert_frame_equal(before, after, check_names=False, check_freq=False)


if __name__ == "__main__":
    make_dirs()

    check(
        "invalid flow_prediction_mode raises at make_model",
        lambda: build({"flow_prediction_mode": "not_a_mode"}),
        expect=ValueError,
    )
    check(
        "missing starfit params file raises at ModelBuilder init",
        lambda: pywrdrb.ModelBuilder(
            inflow_type=INFLOW_TYPE, start_date=MINI_START, end_date=MINI_END,
            options={"starfit_params_filename": "no_such_file.csv"},
        ),
        expect=FileNotFoundError,
    )
    check(
        "starfit params + sensitivity analysis is rejected",
        lambda: pywrdrb.ModelBuilder(
            inflow_type=INFLOW_TYPE, start_date=MINI_START, end_date=MINI_END,
            options={
                "starfit_params_filename": pn.operational_constants.get_str("istarf_conus.csv"),
                "run_starfit_sensitivity_analysis": True,
            },
        ),
        expect=ValueError,
    )
    check(
        "starfit CSV missing a reservoir row fails at build",
        lambda: build({"starfit_params_filename": str(missing_row_csv())}),
        expect=Exception,
    )
    check(
        "offline simulator rejects missing inflow column",
        lambda: STARFITOfflineSimulator().simulate_all(
            pd.DataFrame(
                {"blueMarsh": [100.0] * 10},
                index=pd.date_range("2005-06-01", periods=10),
            )
        ),
        expect=ValueError,
    )

    for mode in ["regression_disagg", "perfect_foresight"]:
        check(
            f"one-month run completes ({mode})",
            lambda m=mode: run_pywrdrb(
                f"stress_mini_{m}", {"flow_prediction_mode": m}, MINI_START, MINI_END
            ),
        )

    check("Data export/load_from_export round trip", data_round_trip)

    print()
    width = max(len(r[0]) for r in results)
    for name, status, detail in results:
        print(f"{name:<{width}}  {status}  ({detail})")
    n_fail = sum(1 for r in results if r[1] == "FAIL")
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
