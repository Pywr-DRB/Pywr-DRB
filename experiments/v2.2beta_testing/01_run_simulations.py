"""
Run the three v2.2beta test simulations.

Runs pub_nhmv10_BC_withObsScaled over the full 1945-2023 record with both
flow_prediction_mode options, plus one run using a custom STARFIT parameter
CSV generated from the packaged defaults (see utils.make_demo_starfit_csv).

The legacy "gage_flow" prediction mode has been removed from the package; see TODO.md.

Usage:
    python 01_run_simulations.py
"""
import time

from utils import (
    START_DATE, END_DATE, OUTPUT_DIR,
    make_dirs, make_demo_starfit_csv, run_pywrdrb,
)

RERUN = True  # if False, skip runs whose output hdf5 already exists


if __name__ == "__main__":
    make_dirs()
    demo_csv = make_demo_starfit_csv()
    runs = {
        "regression_disagg": {"flow_prediction_mode": "regression_disagg"},
        "perfect_foresight": {"flow_prediction_mode": "perfect_foresight"},
        "custom_starfit_demo": {
            "flow_prediction_mode": "perfect_foresight",
            "starfit_params_filename": str(demo_csv),
        },
    }
    for label, options in runs.items():
        h5_path = OUTPUT_DIR / f"{label}.hdf5"
        if h5_path.exists() and not RERUN:
            print(f"[{label}] output exists, skipping")
            continue
        print(f"[{label}] running {START_DATE} to {END_DATE}")
        t0 = time.time()
        run_pywrdrb(label, options)
        print(f"[{label}] done in {time.time() - t0:.1f}s -> {h5_path.name}")
