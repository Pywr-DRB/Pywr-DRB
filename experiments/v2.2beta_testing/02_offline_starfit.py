"""
Exercise the new STARFITOfflineSimulator outside of a Pywr-DRB run.

Simulates STARFIT releases directly from catchment inflows with:
- default parameters (all STARFIT reservoirs)
- a sweep over initial_volume_frac (storage trajectories for one reservoir)
- a reservoir subset
- a demo custom parameter CSV generated from the packaged defaults

Results are written to outputs/ as CSVs for use by 04_make_figures.py.

Usage:
    python 02_offline_starfit.py
"""
import pandas as pd

from pywrdrb.pre import STARFITOfflineSimulator

from utils import OUTPUT_DIR, make_dirs, make_demo_starfit_csv, load_catchment_inflows

INITIAL_VOLUME_FRACS = [0.2, 0.5, 0.8, 1.0]
SWEEP_RESERVOIR = "prompton"
SUBSET = ["blueMarsh", "beltzvilleCombined"]


def main():
    make_dirs()
    inflows = load_catchment_inflows()
    print(f"inflows: {inflows.index[0].date()} to {inflows.index[-1].date()}, {len(inflows)} days")

    # Default parameters, all STARFIT reservoirs
    sim = STARFITOfflineSimulator(initial_volume_frac=0.8)
    releases = sim.simulate_all(inflows)
    releases.to_csv(OUTPUT_DIR / "offline_releases_default.csv")
    print(f"default params: {releases.shape[1]} reservoirs simulated")

    # Sensitivity to initial storage: trajectories for one reservoir
    doy = inflows.index.dayofyear.values
    inflow_vals = inflows[SWEEP_RESERVOIR].values.astype(float)
    storage_sweep = {}
    for frac in INITIAL_VOLUME_FRACS:
        sim_i = STARFITOfflineSimulator(initial_volume_frac=frac)
        sim_i.load_parameters()
        _, storage = sim_i.simulate_reservoir(SWEEP_RESERVOIR, inflow_vals, doy)
        storage_sweep[f"ivf_{frac}"] = storage[1:]
    pd.DataFrame(storage_sweep, index=inflows.index).to_csv(
        OUTPUT_DIR / f"offline_storage_sweep_{SWEEP_RESERVOIR}.csv"
    )
    print(f"initial_volume_frac sweep on {SWEEP_RESERVOIR}: {INITIAL_VOLUME_FRACS}")

    # Reservoir subset
    subset_releases = sim.simulate_all(inflows, reservoir_list=SUBSET)
    assert list(subset_releases.columns) == SUBSET
    print(f"subset run ok: {SUBSET}")

    # Custom parameters: demo CSV derived from the packaged defaults
    demo_csv = make_demo_starfit_csv()
    sim_custom = STARFITOfflineSimulator(
        initial_volume_frac=0.8, starfit_params_filename=str(demo_csv)
    )
    releases_custom = sim_custom.simulate_all(inflows)
    releases_custom.to_csv(OUTPUT_DIR / "offline_releases_custom.csv")

    diff = (releases_custom - releases).abs().mean()
    print("mean |release difference| custom demo vs default (MGD):")
    print(diff[diff > 0.01].round(2).to_string())


if __name__ == "__main__":
    main()
