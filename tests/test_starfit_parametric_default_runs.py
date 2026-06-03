import numpy as np
import pandas as pd

import pywrdrb
from pywrdrb.path_manager import get_pn_object
from pywrdrb.utils.dates import model_date_ranges


PARAMETRIC_RESERVOIRS = [
    "fewalter",
    "prompton",
    "beltzvilleCombined",
    "blueMarsh",
]


def _build_parametric_release_policy_dict(policy_id_value: str):
    return {
        reservoir: {
            "class_type": "ParametricReservoirRelease",
            "policy_type": "STARFIT",
            "policy_id": policy_id_value,
            "params": None,
        }
        for reservoir in PARAMETRIC_RESERVOIRS
    }


def _run_model_and_load_releases(tmp_path, tag: str, release_policy_dict=None):
    inflow_type = "nhmv10"
    start, _ = model_date_ranges[inflow_type]
    end = start + pd.Timedelta(days=30)

    options = {}
    if release_policy_dict is not None:
        options["release_policy_dict"] = release_policy_dict

    mb = pywrdrb.ModelBuilder(
        inflow_type=inflow_type,
        start_date=start,
        end_date=end,
        options=options,
    )
    mb.make_model()
    model_json = tmp_path / f"{tag}.json"
    output_h5 = tmp_path / f"{tag}.hdf5"
    mb.write_model(str(model_json))

    model = pywrdrb.Model.load(str(model_json))
    _ = pywrdrb.OutputRecorder(model=model, output_filename=str(output_h5))
    _ = model.run()

    data = pywrdrb.Data(results_sets=["reservoir_downstream_gage"])
    data.load_output(output_filenames=[str(output_h5)])
    key = f"output_{tag}"
    release_df = data.reservoir_downstream_gage[key][0].copy()
    return mb, release_df


def test_starfit_csv_has_single_default_row_per_parametric_reservoir():
    pn = get_pn_object()
    starfit_path = pn.operational_constants.get_str("starfit.csv")
    df = pd.read_csv(starfit_path)

    for reservoir in PARAMETRIC_RESERVOIRS:
        default_rows = df[
            (df["reservoir"].astype(str) == reservoir)
            & (df["policy_id"].astype(str) == "default")
        ]
        assert len(default_rows) == 1, (
            f"Expected exactly one default row for reservoir '{reservoir}' in {starfit_path}, "
            f"found {len(default_rows)}."
        )


def test_parametric_starfit_default_policy_id_runs_and_compares_to_legacy(tmp_path):
    # Baseline (legacy STARFITReservoirRelease): no release_policy_dict override.
    _, legacy_release = _run_model_and_load_releases(tmp_path, tag="legacy_starfit")

    # Parametric STARFIT with explicit default policy_id.
    mb_default, param_default_release = _run_model_and_load_releases(
        tmp_path,
        tag="parametric_starfit_default",
        release_policy_dict=_build_parametric_release_policy_dict("default"),
    )

    # Parametric STARFIT with blank policy_id should fall back to default row.
    mb_blank, param_blank_release = _run_model_and_load_releases(
        tmp_path,
        tag="parametric_starfit_blank",
        release_policy_dict=_build_parametric_release_policy_dict(""),
    )

    for reservoir in PARAMETRIC_RESERVOIRS:
        param_name_default = f"ParametricReservoirRelease_STARFIT_{reservoir}"
        param_name_blank = f"ParametricReservoirRelease_STARFIT_{reservoir}"

        p_default = mb_default.model_dict["parameters"][param_name_default]
        p_blank = mb_blank.model_dict["parameters"][param_name_blank]
        assert p_default["type"] == "ParametricReservoirRelease"
        assert p_default["policy_id"] == "default"
        assert p_blank["type"] == "ParametricReservoirRelease"
        assert p_blank["policy_id"] == ""

        # Both default and blank-id fallback should produce finite release traces.
        s_default = param_default_release[reservoir].astype(float)
        s_blank = param_blank_release[reservoir].astype(float)
        assert np.isfinite(s_default.values).all()
        assert np.isfinite(s_blank.values).all()

        # Blank policy_id run should resolve to default row and match explicit default.
        assert np.allclose(s_default.values, s_blank.values, rtol=1e-10, atol=1e-10), (
            f"Blank policy_id did not match explicit default for reservoir '{reservoir}'."
        )

    # Nice-to-have: comparison output vs legacy STARFIT behavior.
    comparison_rows = []
    for reservoir in PARAMETRIC_RESERVOIRS:
        legacy = legacy_release[reservoir].astype(float)
        param = param_default_release[reservoir].astype(float)
        aligned = pd.concat([legacy.rename("legacy"), param.rename("param")], axis=1).dropna()
        diff = aligned["param"] - aligned["legacy"]
        comparison_rows.append(
            {
                "reservoir": reservoir,
                "n_days": int(len(aligned)),
                "mean_legacy_mgd": float(aligned["legacy"].mean()),
                "mean_param_mgd": float(aligned["param"].mean()),
                "rmse_param_minus_legacy_mgd": float(np.sqrt(np.mean(np.square(diff)))),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows).sort_values("reservoir")
    print("\nSTARFIT legacy vs parametric(default) release comparison (MGD):")
    print(comparison_df.to_string(index=False))
