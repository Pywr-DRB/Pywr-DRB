"""
Shared paths, constants, and helpers for the v2.2beta testing scripts.
"""
import sys
from pathlib import Path

import pandas as pd

import pywrdrb
from pywrdrb.path_manager import get_pn_object
from pywrdrb.utils.lists import modified_starfit_reservoir_list

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model_json"
OUTPUT_DIR = HERE / "outputs"
FIG_DIR = HERE / "figures"

INFLOW_TYPE = "pub_nhmv10_BC_withObsScaled"
START_DATE = "1945-01-01"
END_DATE = "2023-12-31"

# Perturbation used to demonstrate the custom STARFIT parameter option
DEMO_RESERVOIRS = ["blueMarsh", "beltzvilleCombined", "fewalter", "prompton"]
DEMO_RELEASE_C_BUMP = 0.25


def make_dirs():
    for d in (MODEL_DIR, OUTPUT_DIR, FIG_DIR):
        d.mkdir(exist_ok=True)


def require_files(paths, hint):
    """Exit with a clear message if any required files are missing."""
    missing = [Path(p) for p in paths if not Path(p).exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        sys.exit(f"Missing required files: {names}\n{hint}")


def make_demo_starfit_csv():
    """
    Build a demonstration STARFIT parameter CSV from the packaged defaults.

    Copies istarf_conus.csv and raises Release_c on a few reservoirs,
    increasing their target release rate so storage rides visibly lower.
    Release_c avoids the NOR bound clipping, so the change always takes
    effect. Writes outputs/custom_starfit_demo.csv and returns the path.
    """
    pn = get_pn_object()
    params = pd.read_csv(
        pn.operational_constants.get_str("istarf_conus.csv"), index_col=0
    )
    for res in DEMO_RESERVOIRS:
        row = f"modified_{res}" if res in modified_starfit_reservoir_list else res
        params.loc[row, "Release_c"] += DEMO_RELEASE_C_BUMP
    path = OUTPUT_DIR / "custom_starfit_demo.csv"
    params.index.name = "reservoir"
    params.to_csv(path)
    return path


def load_catchment_inflows(start=START_DATE, end=END_DATE):
    """Catchment inflows (MGD) for the test dataset over the test window."""
    pn = get_pn_object()
    fname = pn.sc.get(f"flows/{INFLOW_TYPE}") / "catchment_inflow_mgd.csv"
    inflows = pd.read_csv(str(fname), index_col=0, parse_dates=True)
    return inflows.loc[start:end]


def run_pywrdrb(label, options, start=START_DATE, end=END_DATE):
    """
    Build and run one pywrdrb simulation.

    Writes model_json/{label}.json and outputs/{label}.hdf5; returns the
    hdf5 path.
    """
    json_path = MODEL_DIR / f"{label}.json"
    h5_path = OUTPUT_DIR / f"{label}.hdf5"

    mb = pywrdrb.ModelBuilder(
        inflow_type=INFLOW_TYPE, start_date=start, end_date=end, options=options
    )
    mb.make_model()
    mb.write_model(str(json_path))

    model = pywrdrb.Model.load(str(json_path))
    pywrdrb.OutputRecorder(
        model=model,
        output_filename=str(h5_path),
        parameters=[p for p in model.parameters if p.name],
    )
    model.run()
    return h5_path
