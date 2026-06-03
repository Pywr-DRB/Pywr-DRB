# STARFIT Parametric Merge Checklist

## Near-Term (This Push)

- Keep STARFIT as the policy family for selected defaults.
- Preserve legacy STARFIT behavior (`STARFITReservoirRelease`) as reference.
- Route new recommended defaults through `ParametricReservoirRelease` using:
  - `policy_type = STARFIT`
  - `policy_id = <selected_id>`
  - STARFIT constants source: `starfit.csv` (CEE optimization handoff file)
- Do not remove/overwrite existing STARFIT defaults in `istarf_conus.csv`.

## CSV Handoff Contract (CEE -> PywrDRB)

`starfit.csv` should include:

- `reservoir`
- `policy_id`
- STARFIT params:
  - `NORhi_mu`, `NORhi_min`, `NORhi_max`, `NORhi_alpha`, `NORhi_beta`
  - `NORlo_mu`, `NORlo_min`, `NORlo_max`, `NORlo_alpha`, `NORlo_beta`
  - `Release_alpha1`, `Release_alpha2`, `Release_beta1`, `Release_beta2`
  - `Release_c`, `Release_p1`, `Release_p2`

Notes:

- Store multiple candidates per reservoir as unique `policy_id`s (e.g., 3 defaults per reservoir).
- Keep a `default` row for fallback behavior.
- Inline injection is test-only convenience; production path should use CSV `policy_id`.

## Required Tests Before Merge

- `tests/test_parametric_release_loading.py`
  - STARFIT source file mapping (`starfit.csv`)
  - row selection and fallback logic for (`reservoir`, `policy_id`)
- `tests/test_starfit_parametric_consistency.py`
  - `fewalter` and `blueMarsh`
  - legacy STARFIT reference math from `istarf_conus.csv`
  - parametric inline injection from test CSV
  - explicit param equality check between inline CSV and ISTARF defaults

## Branch-to-Branch Comparison (Current vs 2.2 Dev)

Use `tests/compare_branch_versions.py`:

```bash
python3 tests/compare_branch_versions.py \
  --repo-a "/path/to/current-branch-checkout" \
  --repo-b "/path/to/pywrdrb-2.2-dev-checkout" \
  --inflow-types "nhmv10_withObsScaled,nwmv21_withObsScaled" \
  --output-dir "model_runs/branch_comparison"
```

This generates:

- pytest status in each repo
- STARFIT legacy-vs-parametric comparison runs for each inflow scenario
- combined report: `model_runs/branch_comparison/branch_version_comparison_report.md`

