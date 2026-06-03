#!/usr/bin/env python3
"""
Run apples-to-apples STARFIT implementation comparisons.

Compares:
1) STARFITReservoirRelease (legacy)
2) ParametricReservoirRelease with policy_type=STARFIT

for selected reservoirs and policy_ids over the same simulation period.
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
import pywrdrb
from pywrdrb.model_builder import Options
from pywrdrb.utils.dates import model_date_ranges


DEFAULT_RESERVOIRS = ["beltzvilleCombined", "fewalter", "prompton", "blueMarsh"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inflow-type", default="nhmv10_withObsScaled")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--reservoirs",
        default=",".join(DEFAULT_RESERVOIRS),
        help="Comma-separated reservoir names.",
    )
    parser.add_argument(
        "--policy-id",
        default="default",
        help="STARFIT policy_id to compare in both implementations.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("model_runs/starfit_comparison"),
        help="Directory for temporary model/output files and summary CSV.",
    )
    return parser.parse_args()


def _split_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _build_release_policy_dict(
    reservoirs: Iterable[str],
    class_type: str,
    policy_type: str,
    policy_id: str,
) -> Dict[str, dict]:
    return {
        r: {
            "class_type": class_type,
            "policy_type": policy_type,
            "policy_id": policy_id,
            "params": None,
        }
        for r in reservoirs
    }


def _run_model(
    inflow_type: str,
    start_date: str,
    end_date: str,
    release_policy_dict: dict,
    model_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    opts = Options(release_policy_dict=release_policy_dict)
    mb = pywrdrb.ModelBuilder(
        inflow_type=inflow_type,
        start_date=start_date,
        end_date=end_date,
        options=opts,
    )
    mb.make_model()
    mb.write_model(str(model_path))

    model = pywrdrb.Model.load(str(model_path))
    _ = pywrdrb.OutputRecorder(model=model, output_filename=str(output_path))
    _ = model.run()

    data = pywrdrb.Data()
    data.load_output(output_filenames=[str(output_path)], results_sets=["res_release"])
    run_name = output_path.stem
    return data.res_release[run_name][0]


def _summarize_release_differences(
    df_legacy: pd.DataFrame,
    df_parametric: pd.DataFrame,
    reservoirs: Iterable[str],
) -> pd.DataFrame:
    rows = []
    for r in reservoirs:
        if r not in df_legacy.columns or r not in df_parametric.columns:
            raise KeyError(f"Missing reservoir column '{r}' in one of the model outputs.")
        diff = (df_parametric[r] - df_legacy[r]).astype(float)
        rows.append(
            {
                "reservoir": r,
                "n_steps": int(diff.shape[0]),
                "mean_legacy_release_mgd": float(df_legacy[r].astype(float).mean()),
                "mean_parametric_release_mgd": float(df_parametric[r].astype(float).mean()),
                "bias_mgd_param_minus_legacy": float(diff.mean()),
                "mean_abs_diff_mgd": float(diff.abs().mean()),
                "max_abs_diff_mgd": float(diff.abs().max()),
                "rmse_mgd": float((diff.pow(2).mean()) ** 0.5),
            }
        )
    return pd.DataFrame(rows).sort_values("reservoir").reset_index(drop=True)


def _build_diagnostics_timeseries(
    df_legacy: pd.DataFrame,
    df_parametric: pd.DataFrame,
    reservoirs: Iterable[str],
) -> pd.DataFrame:
    out = pd.DataFrame(index=df_legacy.index)
    for r in reservoirs:
        out[f"{r}_legacy_mgd"] = df_legacy[r].astype(float)
        out[f"{r}_parametric_mgd"] = df_parametric[r].astype(float)
        out[f"{r}_diff_mgd"] = out[f"{r}_parametric_mgd"] - out[f"{r}_legacy_mgd"]
    return out


def _plot_reservoir_diagnostics(
    diagnostics_ts: pd.DataFrame,
    reservoir: str,
    figure_path: Path,
) -> None:
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 6), sharex=True)
    legacy_col = f"{reservoir}_legacy_mgd"
    param_col = f"{reservoir}_parametric_mgd"
    diff_col = f"{reservoir}_diff_mgd"

    diagnostics_ts[[legacy_col, param_col]].plot(ax=axes[0], linewidth=1.0)
    axes[0].set_ylabel("Release (MGD)")
    axes[0].set_title(f"{reservoir}: legacy STARFIT vs parametric STARFIT")
    axes[0].legend(["legacy", "parametric"], loc="upper right")

    diagnostics_ts[diff_col].plot(ax=axes[1], color="tab:red", linewidth=0.9)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Delta (MGD)")
    axes[1].set_title("Parametric - legacy")
    axes[1].set_xlabel("Date")

    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    reservoirs = _split_csv_list(args.reservoirs)

    if args.start_date is None or args.end_date is None:
        default_start, default_end = model_date_ranges[args.inflow_type]
        start_date = args.start_date or default_start
        end_date = args.end_date or default_end
    else:
        start_date, end_date = args.start_date, args.end_date

    args.work_dir.mkdir(parents=True, exist_ok=True)

    legacy_dict = _build_release_policy_dict(
        reservoirs,
        class_type="STARFITReservoirRelease",
        policy_type="STARFIT",
        policy_id=args.policy_id,
    )
    parametric_dict = _build_release_policy_dict(
        reservoirs,
        class_type="ParametricReservoirRelease",
        policy_type="STARFIT",
        policy_id=args.policy_id,
    )

    legacy_model = args.work_dir / "legacy_starfit_model.json"
    legacy_out = args.work_dir / "legacy_starfit.hdf5"
    param_model = args.work_dir / "parametric_starfit_model.json"
    param_out = args.work_dir / "parametric_starfit.hdf5"

    print(f"Running legacy STARFIT model: {legacy_model}")
    df_legacy = _run_model(
        args.inflow_type, start_date, end_date, legacy_dict, legacy_model, legacy_out
    )

    print(f"Running parametric STARFIT model: {param_model}")
    df_parametric = _run_model(
        args.inflow_type, start_date, end_date, parametric_dict, param_model, param_out
    )

    summary = _summarize_release_differences(df_legacy, df_parametric, reservoirs)
    summary_path = args.work_dir / "starfit_parametric_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    diagnostics_ts = _build_diagnostics_timeseries(df_legacy, df_parametric, reservoirs)
    ts_path = args.work_dir / "starfit_parametric_release_timeseries.csv"
    diagnostics_ts.to_csv(ts_path, index=True)

    fig_dir = args.work_dir / "figures"
    for reservoir in reservoirs:
        fig_path = fig_dir / f"{reservoir}_starfit_parametric_comparison.png"
        _plot_reservoir_diagnostics(diagnostics_ts, reservoir, fig_path)

    print("\nComparison summary (MGD):")
    print(summary.to_string(index=False))
    print(f"\nSaved summary CSV: {summary_path}")
    print(f"Saved diagnostics time-series CSV: {ts_path}")
    print(f"Saved reservoir figures in: {fig_dir}")


if __name__ == "__main__":
    main()
