#!/usr/bin/env python
"""
Re-run predicted inflow (and optionally diversion) generation.

- Inflows: writes `predicted_inflows_mgd.csv` per flow type with columns for each
  requested mode (default: regression_disagg + perfect_foresight).
- Use `remove_zeros=True` if you need the storage-bug mitigation for regression
  predictions (see project notes).
- Diversions: optional NYC predicted diversions with both modes (for FFMP).

Output (per inflow type):
  src/pywrdrb/data/flows/<inflow_type>/predicted_inflows_mgd.csv
"""

import os
from pathlib import Path
from typing import Optional, Tuple

from pywrdrb.pre import (
    PredictedInflowPreprocessor,
    PredictedDiversionPreprocessor,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

REDO_INFLOW_PREDICTION = True
REDO_DIVERSION_PREDICTION = False

# Modes written into predicted_inflows_mgd.csv (ModelBuilder.flow_prediction_mode selects which to use).
PREDICTION_MODES: Tuple[str, ...] = ("regression_disagg", "perfect_foresight")

# Use remove_zeros=True to mitigate odd storage at some reservoirs (regression path).
REMOVE_ZEROS = True

INFLOW_TYPE_OPTIONS = [
    "nhmv10",
    "nhmv10_withObsScaled",
    "nwmv21",
    "nwmv21_withObsScaled",
    "wrf1960s_calib_nlcd2016",
    "wrf2050s_calib_nlcd2016",
    "wrfaorc_calib_nlcd2016",
    "wrfaorc_withObsScaled",
    "pub_nhmv10_BC_withObsScaled",
]


def _abs_path(path) -> str:
    return str(Path(path).resolve())


def run_one(inflow_type: str, redo: bool, remove_zeros: bool, modes: Tuple[str, ...]) -> Optional[str]:
    """Run inflow prediction for one flow type. Returns output path if saved."""
    print(f"\n  Inflow type: {inflow_type}")

    try:
        inflow_predictor = PredictedInflowPreprocessor(
            flow_type=inflow_type,
            remove_zeros=remove_zeros,
            modes=modes,
        )
    except AttributeError as e:
        if "No shortcut found" in str(e):
            print("    Skipping: no data folder for this flow type (not in repo).")
            print(f"    Reason: {e}")
            return None
        raise

    out_key = "predicted_inflows_mgd.csv"
    output_path = inflow_predictor.output_dirs[out_key]
    output_abs = _abs_path(output_path)
    input_path = inflow_predictor.input_dirs["catchment_inflow_mgd.csv"]
    input_abs = _abs_path(input_path)

    if not redo:
        print("    Skipping (REDO_INFLOW_PREDICTION=False).")
        return None

    print(f"    modes={modes}, remove_zeros={remove_zeros}")
    print(f"    Reading input: {input_abs}")

    print("    Step: load()")
    inflow_predictor.load()

    print("    Step: process()")
    inflow_predictor.process()

    replacing = os.path.exists(output_path)
    if replacing:
        print(f"    Replacing existing file: {output_abs}")
    else:
        print(f"    Writing new file: {output_abs}")
    print("    Step: save()")
    inflow_predictor.save()

    print(f"    Done. Output: {output_abs}")
    return output_abs


def main():
    print("=" * 70)
    print("RUN INFLOW PREDICTION")
    print("=" * 70)
    print(f"REDO_INFLOW_PREDICTION = {REDO_INFLOW_PREDICTION}")
    print(f"PREDICTION_MODES = {PREDICTION_MODES}")
    print(f"REMOVE_ZEROS = {REMOVE_ZEROS}")
    print(f"Inflow types ({len(INFLOW_TYPE_OPTIONS)}):")
    for t in INFLOW_TYPE_OPTIONS:
        print(f"  - {t}")
    print()

    if REDO_DIVERSION_PREDICTION:
        print("Predicted diversions (NYC, both modes)...")
        nyc_pred = PredictedDiversionPreprocessor(
            modes=PREDICTION_MODES,
            remove_zeros=REMOVE_ZEROS,
        )
        nyc_pred.load()
        nyc_pred.process()
        nyc_pred.save()
        print("  Diversions done.\n")

    if not REDO_INFLOW_PREDICTION:
        print("Nothing to do for inflows (REDO_INFLOW_PREDICTION is False). Exiting.")
        return

    saved_paths = []
    for inflow_type in INFLOW_TYPE_OPTIONS:
        path = run_one(
            inflow_type,
            redo=REDO_INFLOW_PREDICTION,
            remove_zeros=REMOVE_ZEROS,
            modes=PREDICTION_MODES,
        )
        if path:
            saved_paths.append((inflow_type, path))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if saved_paths:
        for inflow_type, path in saved_paths:
            print(f"  {inflow_type}\n    -> {path}")
        print("\nFor simulations with perfect information, set ModelBuilder options:")
        print('  flow_prediction_mode="perfect_foresight"')
    else:
        print("No inflow files written.")
    print()


if __name__ == "__main__":
    main()
