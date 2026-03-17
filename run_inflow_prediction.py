#!/usr/bin/env python
"""
Re-run predicted inflow generation with remove_zeros=True.

Fixes the predicted-inflow bug that can cause odd storage at Blue Marsh,
Beltzville, and other reservoirs. After running, re-run model simulations
and diagnostic figures to verify.

Output (per inflow type):
  src/pywrdrb/data/flows/<inflow_type>/predicted_inflows_mgd.csv
"""

import os
from pathlib import Path
from typing import Optional

from pywrdrb.pre import PredictedInflowPreprocessor


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Set to True to regenerate predicted inflows; False to skip (no-op).
REDO_INFLOW_PREDICTION = True

# Use remove_zeros=True to fix the storage bug (recommended).
REMOVE_ZEROS = True

# Inflow types to process (must exist under src/pywrdrb/data/flows/).
# pub_nhmv10_withObsScaled is not in this repo; only types with a flows/ folder are valid.
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

# Example: process only the type used for Beltzville/Blue Marsh experiments:
# INFLOW_TYPE_OPTIONS = ["pub_nhmv10_BC_withObsScaled"]


def _abs_path(path) -> str:
    """Resolve to absolute path for clearer logging."""
    return str(Path(path).resolve())


def run_one(inflow_type: str, redo: bool, remove_zeros: bool) -> Optional[str]:
    """
    Run inflow prediction for one inflow type. Returns output path if saved, else None.
    Skips with a warning if this flow type has no data folder (no shortcut in path manager).
    """
    print(f"\n  Inflow type: {inflow_type}")

    try:
        inflow_predictor = PredictedInflowPreprocessor(
            flow_type=inflow_type,
            remove_zeros=remove_zeros,
        )
    except AttributeError as e:
        if "No shortcut found" in str(e):
            print(f"    Skipping: no data folder for this flow type (not in repo).")
            print(f"    Reason: {e}")
            return None
        raise

    # Create preprocessor (output path is set above)
    out_key = "predicted_inflows_mgd.csv"
    output_path = inflow_predictor.output_dirs[out_key]
    output_abs = _abs_path(output_path)
    input_path = inflow_predictor.input_dirs["catchment_inflow_mgd.csv"]
    input_abs = _abs_path(input_path)

    if not redo:
        print(f"    Skipping (REDO_INFLOW_PREDICTION=False).")
        return None

    # What we're changing
    print(f"    Setting: remove_zeros={remove_zeros} (fix for storage bug).")
    print(f"    Reading input: {input_abs}")

    # Load
    print(f"    Step: load()")
    inflow_predictor.load()

    # Process
    print(f"    Step: process() (training + prediction with remove_zeros={remove_zeros})")
    inflow_predictor.process()

    # Save (and say whether we're replacing or creating)
    replacing = os.path.exists(output_path)
    if replacing:
        print(f"    Replacing existing file: {output_abs}")
    else:
        print(f"    Writing new file: {output_abs}")
    print(f"    Step: save()")
    inflow_predictor.save()

    print(f"    Done. Output now lives at: {output_abs}")
    return output_abs


def main():
    print("=" * 70)
    print("RUN INFLOW PREDICTION (remove_zeros fix)")
    print("=" * 70)
    print(f"REDO_INFLOW_PREDICTION = {REDO_INFLOW_PREDICTION}")
    print(f"REMOVE_ZEROS = {REMOVE_ZEROS}")
    print(f"Inflow types to process ({len(INFLOW_TYPE_OPTIONS)}):")
    for t in INFLOW_TYPE_OPTIONS:
        print(f"  - {t}")
    print()

    if not REDO_INFLOW_PREDICTION:
        print("Nothing to do (REDO_INFLOW_PREDICTION is False). Exiting.")
        return

    saved_paths = []
    for inflow_type in INFLOW_TYPE_OPTIONS:
        path = run_one(inflow_type, redo=REDO_INFLOW_PREDICTION, remove_zeros=REMOVE_ZEROS)
        if path:
            saved_paths.append((inflow_type, path))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if saved_paths:
        print("Predicted inflow files written (used by the model):")
        for inflow_type, path in saved_paths:
            print(f"  {inflow_type}")
            print(f"    -> {path}")
        print("\nNext: re-run your model runs and diagnostic figures to verify the fix.")
    else:
        print("No files written (REDO_INFLOW_PREDICTION was False or no inflow types).")
    print()


if __name__ == "__main__":
    main()
