# Variable-N Storage Zone Support in pywrdrb

**Scope:** Technical notes on how pywrdrb supports arbitrary-N NYC reservoir
storage-zone configurations beyond the stock FFMP 6-curve / 7-drought-level
formulation. Covers architecture, data flow, parameter naming, the bug fix
to `salt_front_location.py`, invariants, and known limits.

**Companion docs:**
[NYC_Operations_Configuration.md](NYC_Operations_Configuration.md) (user-facing),
and `Appendix_A_FFMP-20180716-Final.pdf` (primary FFMP reference).

---

## 1. Motivation

NYCOptimization's multi-objective DV sweep (`ffmp_N` formulation) explores the
complexity–performance frontier by varying the number of storage-zone
threshold curves used in NYC reservoir operations. In the stock FFMP scheme
there are 6 curves (`level1b`…`level5`) separating 7 drought levels
(`level1a`, `level1b`, `level1c`, `level2`, `level3`, `level4`, `level5`).
An optimizer may ask: *what if we used 10 curves? 20?* To answer that, the
simulator must accept an arbitrary-N configuration end-to-end without the
FFMP-6 count being hardcoded.

Most of the FFMP logic is already zone-count-agnostic — parameters like
`drought_level_agg_nyc` are driven by `ControlCurveIndex` which accepts any
list of curves. The changes documented here close the remaining gaps:

1. A first-class `NYCOperationsConfig.from_n_zones(N)` constructor that
   interpolates the FFMP defaults onto N zones.
2. Parametric drought-emergency-level gates in `LowerBasinMaxMRFContribution`
   and `FlowTargetSaltFrontAdjustmentRatio` (previously hardcoded to 6 and 5
   respectively — the 5 was also a latent bug, see §5).
3. Output-loader autodiscovery for the `ffmp_level_boundaries` result set.

Backward compatibility: all default FFMP runs produce byte-identical outputs
for every non-salinity parameter. `from_n_zones(6)` is element-wise equivalent
to `from_defaults()` (exercised by
`tests/test_nzone_support.py::test_endpoints_match_defaults_at_N6`).

---

## 2. Level-name taxonomy

pywrdrb supports two naming conventions, selected at config construction
time and propagated throughout the parameter graph:

| Scheme | Storage curves (`STORAGE_LEVELS`)         | Drought levels (`DROUGHT_LEVELS`) |
|--------|-------------------------------------------|-----------------------------------|
| FFMP default | `['level1b', 'level1c', 'level2', 'level3', 'level4', 'level5']` | `['level1a', 'level1b', 'level1c', 'level2', 'level3', 'level4', 'level5']` |
| N-zone | `['zone_1', 'zone_2', …, 'zone_N']`       | `['zone_0', 'zone_1', …, 'zone_N']` |

Relationships (invariants):

- `DROUGHT_LEVELS[0]` is always the implicit "above top curve" band
  (`level1a` / `zone_0`) — no threshold row exists for this level in
  `storage_zones_df`.
- `len(DROUGHT_LEVELS) == len(STORAGE_LEVELS) + 1 == n_drought_levels`.
- Storage curves are stored **in decreasing order** (highest curve first).
  This ordering is a hard requirement of pywr's `ControlCurveIndex`.
- `STORAGE_LEVELS` and `DROUGHT_LEVELS` are exposed as *properties* on
  `NYCOperationsConfig`, derived at call time from the DataFrame index
  (`src/pywrdrb/parameters/nyc_operations_config.py:59-87`). The class
  attributes `_DEFAULT_STORAGE_LEVELS` / `_DEFAULT_DROUGHT_LEVELS` only
  provide fallbacks when no DataFrame is loaded.

---

## 3. `NYCOperationsConfig.from_n_zones(N)`

The alternate constructor lives next to `from_defaults` at
`src/pywrdrb/parameters/nyc_operations_config.py`. Key points:

**Interpolation scheme.** Linear in a normalized index on `[0, 1]`.
Storage thresholds interpolate from 6 default points to N target points;
MRF factors (per reservoir, per drought level) interpolate from 7 default
points to N+1 target points. Per-day, per-month profiles are interpolated
column-by-column; no smoothing across time.

**Artifacts produced:**

- `storage_zones_df` indexed by `zone_1..zone_N`, with the same 366 date
  columns as the default.
- `mrf_factors_daily_df` indexed by `zone_1..zone_N` plus three
  `{level}_factor_mrf_{reservoir}` rows per drought level — so for N=10,
  shape `(10 zones + 3 × 11 levels, 366) = (43, 366)`.
- `mrf_factors_monthly_df` indexed by `delMontague` / `delTrenton` factors
  per drought level — shape `(2 × 11, 12) = (22, 12)` for N=10.
- `constants` with `{zone_i}_factor_delivery_{nyc|nj}` for `i` in `0..N`,
  plus all non-level-specific scalars (`mrf_baseline_*`,
  `max_flow_baseline_*`, `flood_max_release_*`, reset dates).

**Invariant checks (built into the constructor):**

- `n_zones >= 2`; otherwise `ValueError`.
- Pairwise monotonicity of interpolated thresholds (pywr
  `ControlCurveIndex` requirement); violations raise `ValueError`.
- `"zone_0"` is never a row in `storage_zones_df` (assertion).

**Warnings:** `n_zones < 6` emits a `UserWarning` — the
`flood_conservation_boundary` semantic (see §6) becomes permissive at small N.

**Parity with NYCOptimization.** A parallel `build_nzone_config` exists in
`NYCOptimization/src/simulation.py:261-386`. The pywrdrb port is algorithmically
identical; element-wise parity is asserted in
`tests/test_nzone_support.py::test_parity_with_nyc_optimization`. Once
NYCOptimization migrates its caller to use the pywrdrb classmethod, its local
copy should be deleted.

---

## 4. Parametric drought-emergency level

Two custom parameters previously encoded NYC drought-emergency status via
hardcoded `ControlCurveIndex` values:

- `LowerBasinMaxMRFContribution.get_current_usable_reservoirs`
  (`src/pywrdrb/parameters/lower_basin_ffmp.py`) — gates which lower-basin
  reservoirs are usable for the Trenton flow target.
- `FlowTargetSaltFrontAdjustmentRatio.value`
  (`src/pywrdrb/parameters/salt_front_location.py`) — gates whether salt-front
  location adjusts Montague/Trenton flow targets.

Both now accept a keyword argument `nyc_drought_emergency_level: int = 6`
with default 6 matching the FFMP 6-curve scheme. The wiring in
`model_builder.py` passes `self.nyc_operations_config.n_drought_levels - 1`
to every emission, so:

- Default FFMP (7 drought levels): emergency = 6 (byte-compatible with
  prior runs).
- N-zone with N=10 (11 drought levels): emergency = 10.

Gates use `int(round(x))` for defensive integer comparison since pywr's
`ControlCurveIndex.get_value` returns a float.

---

## 5. Salt-front bug fix (2026-04-20)

**Symptom.** The pre-fix code compared `drought_level_agg_nyc_idx` to the
literal integer 5:

```python
if drought_level_agg_nyc_idx != 5:  # "5 is drought emergency"
    return 1.0
```

**Problem.** Under pywr's `ControlCurveIndex` with 6 storage curves, the
returned value `5` corresponds to the band between `level4` and `level5` —
i.e., **Drought Warning (L4)**, *not* Drought Emergency (L5 = index 6). The
adjustment was therefore firing at L4 only and *not* at L5, which is the
opposite of the FFMP prescription.

**FFMP evidence.** From `docs/Appendix_A_FFMP-20180716-Final.pdf`:

- Drought-stage taxonomy (p. 5, §III.a-c): **L3 = Watch, L4 = Warning,
  L5 = Emergency**.
- Table 2 (p. 6): *"…during Drought Emergency (L5) Operations"*.
- Table 5 heading (p. 38 — salt-front adjustment formula): *"(Interstate
  Operation Formula for Adjusting Montague And Trenton Flow Objectives
  During Drought Emergency (L5) Operations), the City shall make releases
  to meet the Montague flow objectives according to the location of the
  salt front."*

**Fix.** Replace the integer literal with the parametric
`self.nyc_drought_emergency_level`, default 6 (index of L5 in the 6-curve
scheme). See `src/pywrdrb/parameters/salt_front_location.py:387` for the
inline BUGFIX comment.

**Behavioral impact.** In the 1980-1985 diagnostic window the NYC system
never reached L5 or L4, so neither the pre-fix nor post-fix gate fires (see
`experiments/variable_n_zone_sensitivity/figures/salt_front_fix_diagnostic.png`).
The fix will change outputs in severe-drought windows (e.g., 1960s-style
runs with historical inflows calibrated to the Record Drought) where NYC
reaches L5. Reviewers concerned about downstream impact should re-run
salinity-enabled simulations over such windows and compare.

---

## 6. `flood_conservation_boundary` and `mrf_factor_l2`

The custom parameters `NYCFloodRelease` and `NYCCombinedReleaseFactor` take
a `flood_conservation_boundary: int` (default 2) that governs the Zone-L1
entry/exit condition and the downstream-flood MRF cap (FFMP §6.iv-vi). This
integer is *not* derived from the zone count — it stays fixed at 2 by
convention regardless of N.

**Rationale (user-specified, plan Q2).** The semantic "top 2 drought levels
form Zone L1, the next level is the MRF cap (L2)" is FFMP-prescribed and
does not generalize cleanly to arbitrary N. For N=6 this resolves to:

- Top 2 drought levels: `level1a` (idx 0), `level1b` (idx 1). These form
  Zone L1.
- MRF cap level: `level1c` (idx 2) boundary — no wait, the cap is the *L2
  factor* (`level2_factor_mrf_*`, idx 3). So the correct index is
  `flood_conservation_boundary + 1`, which resolves to 3 →
  `drought_levels[3] == "level2"`.

For N=10 (11 drought levels), the same formula yields `drought_levels[3] ==
"zone_3"`. The flood-zone semantic is then "top 3 drought levels form L1;
`zone_3` is the factor used to cap MRF during downstream flooding" — which
is more permissive than FFMP's "top 2 out of 7" semantic. For N=3 or 4
this becomes very permissive; the constructor emits a warning for N < 6.

**A pre-existing regression was fixed along the way.** Earlier uncommitted
code at `model_builder.py:1649` used `drought_levels[2]` (resolving to
`level1c`) instead of `drought_levels[3]`. The test
`tests/test_flood_operations.py::test_model_dict_flood` caught this when
rerun. The fix substitutes the semantic form:

```python
flood_B = 2
combined_factor_param["mrf_factor_l2"] = (
    f"{self.drought_levels[flood_B + 1]}_factor_mrf_{reservoir}"
)
```

Default FFMP: resolves to `level2_factor_mrf_*`. N-zone: resolves to
`zone_3_factor_mrf_*`.

---

## 7. Output-loader autodiscovery

`pywrdrb.Data().load_output(..., results_sets=["ffmp_level_boundaries"])`
extracts the zone threshold time series for plotting. The branch in
`src/pywrdrb/load/output_loader.py:287` now:

1. First checks for any of the default six names (`level1b..level5`). If
   *any* are present, the result is those names in legacy order —
   backward-compatible with prior HDF5 outputs and downstream tooling.
2. Otherwise, autodiscovers `zone_*` keys (excluding `zone_0` and any
   `_factor_` rows) and sorts **numerically** so `zone_2` precedes
   `zone_10`.
3. Emits a `UserWarning` if no recognized keys are found.

The numeric sort is important: lexical sort would give
`zone_1, zone_10, zone_11, zone_2, ...`, producing scrambled boundary
overlays.

---

## 8. Data flow end-to-end

Config → ModelBuilder → JSON → pywr Model → Parameters:

```
user code                                           pywr runtime
─────────────────                                   ─────────────
NYCOperationsConfig.from_defaults()      ──┐
  or from_n_zones(N)                       │
                                           ▼
 ModelBuilder(nyc_operations_config=cfg)   self.storage_levels = cfg.STORAGE_LEVELS
                                           self.drought_levels = cfg.DROUGHT_LEVELS
                                           │
                                           ▼
    add_parameter_nyc_reservoirs_*()       emits f"{level}_*" params verbatim
                                           │    (level1b / zone_2 — same code path)
                                           ▼
    mrf_factor_l2 = f"{drought_levels[3]}…"  (semantic index, not hardcoded name)
    LowerBasinMaxMRFContribution payload      "nyc_drought_emergency_level": n-1
    FlowTargetSaltFrontAdjustmentRatio        "nyc_drought_emergency_level": n-1
                                           │
                                           ▼
        make_model()  →  model_dict        (pywr-json, all levels resolved)
                                           │
                                           ▼
              model.run()                  ControlCurveIndex(drought_level_agg_nyc)
                                           returns int in [0, N]; custom parameters
                                           gate on self.nyc_drought_emergency_level
```

---

## 9. Invariants / contract

Any `NYCOperationsConfig` accepted by `ModelBuilder` must satisfy:

| # | Invariant | Enforced where |
|---|-----------|----------------|
| 1 | `DROUGHT_LEVELS[0]` is the implicit "normal" band; absent from `storage_zones_df` | Property definition + assertion in `from_n_zones` |
| 2 | `len(DROUGHT_LEVELS) == len(STORAGE_LEVELS) + 1` | Property definition |
| 3 | Storage curves are in strictly decreasing order of threshold per day | `from_n_zones` raises `ValueError`; runtime: LP gives undefined behavior if violated |
| 4 | All `{level}_factor_delivery_{nyc|nj}` constants exist for every drought level | `NYCOperationsConfig._validate()` warns; `ModelBuilder` raises on None value |
| 5 | All `{level}_factor_mrf_{reservoir}` daily rows exist | `ModelBuilder.add_parameter_nyc_reservoirs_min_require_flow` looks them up; absent → KeyError |
| 6 | All `{level}_factor_mrf_{delMontague|delTrenton}` monthly rows exist | Same, but in `add_parameter_montague_trenton_flow_targets` |
| 7 | `RESERVOIRS == ['cannonsville', 'pepacton', 'neversink']` | Class constant; not parameterized |
| 8 | `flood_conservation_boundary == 2` | Hardcoded in `ModelBuilder` + `NYCFloodRelease` / `NYCCombinedReleaseFactor` defaults |

---

## 10. Testing

`tests/test_nzone_support.py` is organized into four suites:

1. **`TestFromNZones`** — shape, naming, invariants, monotonicity,
   endpoints-at-N6-match-defaults, no orphan level* keys, parity with
   NYCOptimization.
2. **`TestOutputLoaderFFMPBoundaries`** — default name preservation and
   N-zone numeric sort.
3. **`TestParametricEmergencyLevel`** — `LowerBasinMaxMRFContribution`
   fires at configured index and only that index.
4. **`TestEndToEndNZoneModelBuild`** — `make_model()` succeeds for
   N ∈ {6, 10}; wiring confirms `n_drought_levels - 1` is threaded through
   the payload.

Regression suite: `test_flood_operations.py`,
`test_nyc_operations_config.py`, and `test_import_pywrdrb.py` all pass
unchanged.

---

## 11. Known limits and follow-ups

1. **`flood_conservation_boundary` stays at 2 for all N.** This is user-
   prescribed (plan Q2) but should be revisited for small or very large N.
   A config-level override would be a minimal future change.
2. **`LowerBasinMaxMRFContribution` emergency gate is legacy.** Per the user,
   the lower-basin Trenton trigger in practice is bank-exhaustion, not
   drought emergency. The current `== emergency` check remains for
   parity-of-behavior but may be dead code in contemporary runs. Future
   work: audit and either remove or unify with the bank-exhaustion logic.
3. **NYCOptimization duplication.** `build_nzone_config` still exists as a
   parallel implementation in `NYCOptimization/src/simulation.py:261-386`.
   Parity is tested but the two will drift if not maintained jointly.
   Recommended follow-up: replace that function with a one-liner calling
   `NYCOperationsConfig.from_n_zones(N)`.
4. **Salt-front bug fix changes behavior in severe-drought periods.** The
   1980-85 diagnostic window showed no live impact, but dry-period runs
   (1960s Record Drought inflows) will see the salt-front adjustment
   activate exclusively at L5, which may shift Montague/Trenton flow
   trajectories. Recommended: add a regression run across the 1960s drought
   window to quantify.
5. **Non-linear interpolation schemes.** The `interpolation` kwarg on
   `from_n_zones` is currently accepted but only `linear` is implemented.
   Monotone cubic could be added if smoother zone progressions become
   necessary.

---

## 12. File index

| File | Role |
|------|------|
| [src/pywrdrb/parameters/nyc_operations_config.py](../src/pywrdrb/parameters/nyc_operations_config.py) | `NYCOperationsConfig` including `from_n_zones` |
| [src/pywrdrb/parameters/lower_basin_ffmp.py](../src/pywrdrb/parameters/lower_basin_ffmp.py) | Parametric emergency level |
| [src/pywrdrb/parameters/salt_front_location.py](../src/pywrdrb/parameters/salt_front_location.py) | Parametric emergency level + bug fix |
| [src/pywrdrb/load/output_loader.py](../src/pywrdrb/load/output_loader.py) | Autodiscovery of `ffmp_level_boundaries` |
| [src/pywrdrb/model_builder.py](../src/pywrdrb/model_builder.py) | Wires emergency kwarg; uses semantic `flood_B + 1` for `mrf_factor_l2` |
| [tests/test_nzone_support.py](../tests/test_nzone_support.py) | Unit + integration tests for the above |
| [experiments/variable_n_zone_sensitivity/](../experiments/variable_n_zone_sensitivity/) | Diagnostic runner + plots + README |
| `docs/Appendix_A_FFMP-20180716-Final.pdf` | Primary FFMP reference (for future readers) |
