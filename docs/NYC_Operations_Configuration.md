
# NYC Reservoir Operations Configuration

## Overview

NYC reservoir operations in Pywr-DRB follow the 2017 Flexible Flow Management Program (FFMP) framework. The model enforces complex, interactive operational rules across the three NYC reservoirs (Cannonsville, Pepacton, Neversink) including storage-based drought levels, minimum release requirements, flood control, and delivery constraints.

As of the latest update, all operational parameters can be modified programmatically via the `NYCOperationsConfig` API, enabling sensitivity analysis while maintaining backward compatibility with default FFMP operations.

## Operational Components

### 1. Storage Zones (Drought Levels)

**Purpose**: Define operational regimes based on combined or individual reservoir storage.

**Levels**: 1a (flood), 1b, 1c, 2 (normal), 3, 4, 5 (severe drought)

**Parameters**:
- `level1b` through `level5`: Daily storage thresholds (fraction of capacity, 366 values)
- Varies seasonally (e.g., level2 ranges 0.75-1.0 across the year)
- Loaded from `ffmp_reservoir_operation_daily_profiles.csv`

**Operational Logic**:
```python
# Aggregate drought level based on combined NYC storage
drought_level_agg_nyc = ControlCurveIndex(reservoir_agg_nyc, [level1b, level1c, level2, level3, level4, level5])

# Individual drought levels for each reservoir
drought_level_{reservoir} = ControlCurveIndex(reservoir_{reservoir}, [level1b, ..., level5])
```

**Implementation**: `add_parameter_nyc_reservoirs_operational_regimes()` in [model_builder.py:1157](src/pywrdrb/model_builder.py#L1157)

---

### 2. Delivery Constraints

**Purpose**: Limit NYC and NJ diversions based on drought conditions.

#### NYC Delivery
- **Baseline**: 800 MGD (constant `max_flow_baseline_delivery_nyc`)
- **Drought factors** (by level): [1e6, 1e6, 1e6, 1e6, 0.85, 0.7, 0.65] for levels 1a-5
- **Running average constraint**: `FfmpNycRunningAvgParameter` enforces moving average limit
  - Resets annually on May 31st
  - Tracks daily deviations from average

#### NJ Delivery
- **Daily baseline**: 120 MGD (`max_flow_baseline_daily_delivery_nj`)
- **Monthly avg baseline**: 100 MGD (`max_flow_baseline_monthlyAvg_delivery_nj`)
- **Drought factors** (by level): [1, 1, 1, 1, 1, 0.9, 0.8]
- **Running average constraint**: `FfmpNjRunningAvgParameter` with monthly resets

**Implementation**: `add_parameter_nyc_and_nj_delivery_constraints()` in [model_builder.py:1211](src/pywrdrb/model_builder.py#L1211)

---

### 3. Minimum Release Requirements (MRF)

**Purpose**: Maintain minimum streamflows from reservoirs and at downstream gages.

#### Reservoir-Specific MRFs
- **Baselines** (MGD):
  - Cannonsville: 122.8
  - Pepacton: 64.63
  - Neversink: 48.47

- **Daily multiplier factors**: Profile for each drought level × reservoir × day of year (366 values)
  - Example: `level2_factor_mrf_cannonsville` ranges 1.5-7.5 seasonally
  - Loaded from `ffmp_reservoir_operation_daily_profiles.csv`

- **Combined release factor**: Weighted combination of aggregate and individual drought levels
  - `NYCCombinedReleaseFactor` switches between aggregate (normal/drought) and individual (flood) operations
  - Formula: `factor = min(max(D_agg - 2, 0), 1) × factor_agg + min(max(3 - D_agg, 0), 1) × factor_indiv`

- **Final MRF**: `mrf_target_individual_{reservoir} = mrf_baseline × combined_factor`

#### Downstream Flow Targets
- **Baselines** (MGD):
  - Delaware at Montague: 1131.05
  - Delaware at Trenton: 1938.95

- **Monthly multiplier factors**: Profile for each drought level × location × month (12 values)
  - Example: Montague level5 ranges 0.77-0.91 seasonally
  - Trenton level3-5 all use 0.9 year-round
  - Loaded from `ffmp_reservoir_operation_monthly_profiles.csv`

**Implementation**: `add_parameter_nyc_reservoirs_min_require_flow()` in [model_builder.py:1302](src/pywrdrb/model_builder.py#L1302)

---

### 4. Flood Control Releases

**Purpose**: Return reservoirs to level 1b/1c boundary over 7 days during high storage conditions.

**Trigger**: Drought level < 2 (levels 1a or 1b)

**Calculation**:
```python
excess_volume = current_volume - level1c_volume
flood_release = max(min((excess_volume/7 - mrf_target), max_release - mrf_target), 0)
```

**Maximum flood releases** (CFS):
- Cannonsville: 4200
- Pepacton: 2400
- Neversink: 3400
- Based on FFMP Table 5, loaded from `constants.csv`

**Parameter**: `NYCFloodRelease` uses 7-day rolling mean inflow for prediction

**Implementation**: `add_parameter_nyc_reservoirs_flood_control()` in [model_builder.py:1375](src/pywrdrb/model_builder.py#L1375)

---

### 5. Release Balancing

**Purpose**: Distribute MRF and delivery requirements across reservoirs to maintain storage equity.

#### Downstream MRF Contributions
Uses 4-step staggered release process accounting for travel lags:
1. **Step 1**: Cannonsville/Pepacton releases (2-day lag to Montague, 4-day to Trenton)
2. **Step 2**: Neversink releases (1-day lag to Montague, 3-day to Trenton)
3. **Steps 3-4**: Lower basin reservoir contributions

**Balancing algorithm** (`VolBalanceNYCDownstreamMRF`):
1. Calculate proportional targets based on relative storage fractions
2. Enforce non-negativity and max release constraints
3. Iteratively adjust if total doesn't match requirement

#### NYC Delivery Distribution
- **Algorithm**: `VolBalanceNYCDemand` distributes total NYC demand across 3 reservoirs
- **Objective**: Maintain proportional storage levels while meeting delivery requirements
- **Constraints**: Individual max diversions, MRF obligations

**Implementation**: `add_parameter_nyc_reservoirs_balancing_methods()` in [model_builder.py:1613](src/pywrdrb/model_builder.py#L1613)

---

## NYCOperationsConfig API

### Basic Usage

```python
from pywrdrb.parameters.nyc_operations_config import NYCOperationsConfig
from pywrdrb.model_builder import ModelBuilder

# Default operations (backward compatible)
model = ModelBuilder(
    start_date="2000-01-01",
    end_date="2000-12-31",
    inflow_type="nhmv10_withObsScaled"
)

# Custom operations
config = NYCOperationsConfig.from_defaults()
config.update_delivery_constraints(max_nyc_delivery=850)
model = ModelBuilder(
    start_date="2000-01-01",
    end_date="2000-12-31",
    inflow_type="nhmv10_withObsScaled",
    nyc_operations_config=config
)
```

### Configuration Methods

#### Load Configuration
```python
# From default CSV files
config = NYCOperationsConfig.from_defaults()

# From custom directory
config = NYCOperationsConfig.from_defaults(data_dir='path/to/custom/csvs')

# From scratch
config = NYCOperationsConfig(
    storage_zones_df=zones_dataframe,
    mrf_factors_daily_df=daily_factors_dataframe,
    mrf_factors_monthly_df=monthly_factors_dataframe,
    constants=constants_dict
)
```

#### Update Storage Zones
```python
import numpy as np

# Update single level
custom_level2 = np.linspace(0.80, 1.0, 366)
config.update_storage_zones(level='level2', daily_values=custom_level2)

# Replace entire zones dataframe
config.update_storage_zones(zones_df=new_zones_dataframe)
```

#### Update Delivery Constraints
```python
config.update_delivery_constraints(
    max_nyc_delivery=850,           # Baseline NYC limit (MGD)
    max_nj_daily=130,                # Daily NJ limit (MGD)
    max_nj_monthly_avg=110,          # Monthly avg NJ limit (MGD)
    drought_factors_nyc=np.array([1e6, 1e6, 1e6, 1e6, 0.9, 0.75, 0.65]),  # 7 levels
    drought_factors_nj=np.array([1, 1, 1, 1, 1, 0.85, 0.75]),              # 7 levels
    delivery_reset_month=5,          # NYC reset month
    delivery_reset_day=31            # NYC reset day
)
```

#### Update MRF Baselines
```python
config.update_mrf_baselines(
    cannonsville=135.0,    # MGD
    pepacton=70.0,         # MGD
    neversink=50.0,        # MGD
    montague=1200.0,       # MGD
    trenton=2000.0         # MGD
)
```

#### Update MRF Factors
```python
# Update daily factors for specific reservoir and level
daily_factors = np.linspace(2.0, 5.0, 366)
config.update_mrf_factors(
    reservoir='cannonsville',
    level='level2',
    daily_factors=daily_factors
)

# Update monthly factors
monthly_factors = np.array([0.95, 0.95, 0.95, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 0.95, 0.95, 0.95])
config.update_mrf_factors(
    reservoir='cannonsville',
    level='level2',
    monthly_factors=monthly_factors
)

# Replace entire factor dataframes
config.update_mrf_factors(
    factors_daily_df=new_daily_dataframe,
    factors_monthly_df=new_monthly_dataframe
)
```

#### Update Flood Limits
```python
config.update_flood_limits(
    max_release_cannonsville=5000,  # CFS
    max_release_pepacton=3000,      # CFS
    max_release_neversink=4000      # CFS
)
```

#### Query Configuration
```python
# Get constant value
nyc_limit = config.get_constant('max_flow_baseline_delivery_nyc')

# Get storage zone profile
level2_profile = config.get_storage_zone_profile('level2')  # Returns 366-element array

# Get MRF factor profile
daily_mrf = config.get_mrf_factor_profile('level2_factor_mrf_cannonsville', daily=True)   # 366 values
monthly_mrf = config.get_mrf_factor_profile('level2_factor_mrf_delMontague', daily=False) # 12 values
```

#### Export and Copy
```python
# Export to CSV files
config.to_csv('output_directory/')

# Create independent copy
config2 = config.copy()
```

---

## Sensitivity Analysis Example

```python
import numpy as np
import pandas as pd

# Define parameter ranges
nyc_delivery_limits = [750, 800, 850, 900]
storage_zone_shifts = [-0.05, 0.0, 0.05]
flood_release_multipliers = [0.8, 1.0, 1.2]

results = []

for delivery in nyc_delivery_limits:
    for zone_shift in storage_zone_shifts:
        for flood_mult in flood_release_multipliers:

            # Create custom configuration
            config = NYCOperationsConfig.from_defaults()

            # Modify delivery limit
            config.update_delivery_constraints(max_nyc_delivery=delivery)

            # Shift storage zones
            for level in ['level2', 'level3', 'level4', 'level5']:
                original = config.get_storage_zone_profile(level)
                shifted = np.clip(original + zone_shift, 0, 1)
                config.update_storage_zones(level=level, daily_values=shifted)

            # Scale flood releases
            config.update_flood_limits(
                max_release_cannonsville=4200 * flood_mult,
                max_release_pepacton=2400 * flood_mult,
                max_release_neversink=3400 * flood_mult
            )

            # Build and run model
            model = ModelBuilder(
                start_date="2000-01-01",
                end_date="2010-12-31",
                inflow_type="nhmv10_withObsScaled",
                nyc_operations_config=config
            )
            model.make_model()
            model.write_model('model.json')

            # Run simulation (pseudo-code)
            # sim_results = run_simulation(model)
            # results.append({
            #     'delivery_limit': delivery,
            #     'zone_shift': zone_shift,
            #     'flood_mult': flood_mult,
            #     'nyc_reliability': sim_results['nyc_reliability'],
            #     'mrf_compliance': sim_results['mrf_compliance']
            # })

# results_df = pd.DataFrame(results)
```

---

## Data Sources

All operational parameters are loaded from CSV files in `src/pywrdrb/data/operational_constants/`:

1. **constants.csv**: Baseline values and conversion factors (31 parameters)
   - MRF baselines for reservoirs and gages
   - Delivery limits for NYC and NJ
   - Drought factors for delivery constraints
   - Flood release maximum limits
   - Delivery reset dates

2. **ffmp_reservoir_operation_daily_profiles.csv**: Daily varying parameters (366 columns)
   - Storage zone thresholds (level1b-5)
   - MRF multiplier factors for each reservoir × drought level
   - ~32 profiles total

3. **ffmp_reservoir_operation_monthly_profiles.csv**: Monthly varying parameters (12 columns)
   - Downstream flow target multipliers for Montague and Trenton × drought level
   - ~16 profiles total

4. **istarf_conus.csv**: Reservoir physical capacities
   - Used for converting fractional storage to volumes

---

## Implementation Details

### Parameter Class Hierarchy

```
ffmp.py (8 custom Pywr parameters):
├── FfmpNycRunningAvgParameter      → NYC delivery running average constraint
├── FfmpNjRunningAvgParameter       → NJ delivery running average constraint
├── NYCCombinedReleaseFactor        → Weighted aggregate/individual drought factors
├── NYCFloodRelease                 → Flood control excess release calculation
├── TotalReleaseNeededForDownstreamMRF → 4-step staggered MRF requirements
├── VolBalanceNYCDownstreamMRF_step1   → Cannonsville/Pepacton MRF distribution
├── VolBalanceNYCDownstreamMRF_step2   → Neversink MRF contribution
└── VolBalanceNYCDemand                → NYC delivery distribution across reservoirs
```

### Dependency Chain Example (NYC Releases)

```
Storage → drought_level_agg_nyc → drought_factor_combined → mrf_target_individual
                                ↓
Storage → drought_level_{reservoir} → flood_release_{reservoir}
                                ↓
Predicted flows → TotalReleaseNeededForDownstreamMRF → release_needed_mrf
                                ↓
All above → VolBalanceNYCDownstreamMRF → mrf_montagueTrenton_{reservoir}
                                ↓
                    downstream_release_target_{reservoir}
```

### Data Flow

```
CSV Files → NYCOperationsConfig → ModelBuilder.add_parameter_nyc_*()
                                         ↓
                                  Pywr Parameters
                                         ↓
                            Runtime Custom Parameter Classes
                                         ↓
                                   Simulation Results
```

---

## Current Limitations

### Hardcoded Logic Not Yet Parameterized

1. **May 31st reset date**: Added to `constants.csv` but not yet used by `FfmpNycRunningAvgParameter` (line 151 in ffmp.py)
2. **4-step lag structure**: Multi-day travel times hardcoded in balancing methods
3. **Lower basin constants**: Max discharges, conservation releases in `lower_basin_ffmp.py` (lines 64-115)
4. **7-day flood release window**: Hardcoded in `NYCFloodRelease` calculation

These could be parameterized in future updates if required for specific analyses.

### Validation

The `NYCOperationsConfig` class performs basic validation:
- Ensures 366 daily values for all profiles
- Ensures 12 monthly values for monthly profiles
- Checks for required constants
- Warns about missing parameters

However, semantic validation (e.g., ensuring level2 > level3) is not enforced. Users should ensure modifications maintain physical consistency.

---

## References

- **FFMP Documentation**: Delaware River Basin Commission FFMP 2017
- **Model Paper**: Hamilton, A. L., Amestoy, T. J., & Reed, P. M. (2024). Pywr-DRB: An open-source Python model for water availability and drought risk assessment in the Delaware River Basin. *Environmental Modelling & Software*, 106185.
- **Code**: [src/pywrdrb/parameters/ffmp.py](src/pywrdrb/parameters/ffmp.py), [src/pywrdrb/model_builder.py](src/pywrdrb/model_builder.py)
