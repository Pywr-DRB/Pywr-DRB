"""
Configuration class for NYC reservoir operations.

This module provides a flexible configuration system for NYC reservoir operational
parameters, allowing users to customize operational rules while maintaining
backward compatibility with default FFMP operations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
import warnings


class NYCOperationsConfig:
    """
    Configuration class for NYC reservoir operational parameters.

    Supports initialization from default CSV files or custom data structures,
    enabling sensitivity analysis and operational rule modifications.

    Attributes:
        storage_zones_df: DataFrame with storage zone thresholds (level1b-5) by day of year
        mrf_factors_daily_df: DataFrame with MRF release factors by day of year
        mrf_factors_monthly_df: DataFrame with MRF downstream flow factors by month
        constants: Dictionary of operational constants (baselines, limits, etc.)

    """

    RESERVOIRS = ['cannonsville', 'pepacton', 'neversink']
    _DEFAULT_STORAGE_LEVELS = ['level1b', 'level1c', 'level2', 'level3', 'level4', 'level5']
    _DEFAULT_DROUGHT_LEVELS = ['level1a', 'level1b', 'level1c', 'level2', 'level3', 'level4', 'level5']

    def __init__(self,
                 storage_zones_df: Optional[pd.DataFrame] = None,
                 mrf_factors_daily_df: Optional[pd.DataFrame] = None,
                 mrf_factors_monthly_df: Optional[pd.DataFrame] = None,
                 constants: Optional[Dict] = None):
        """
        Initialize NYC operations configuration.

        Parameters:
            storage_zones_df: DataFrame with columns for each day of year (366 cols)
                            and rows for storage zone thresholds
            mrf_factors_daily_df: DataFrame with daily MRF release factors
            mrf_factors_monthly_df: DataFrame with monthly MRF downstream flow factors
            constants: Dictionary of operational constants
        """
        self.storage_zones_df = storage_zones_df
        self.mrf_factors_daily_df = mrf_factors_daily_df
        self.mrf_factors_monthly_df = mrf_factors_monthly_df
        self.constants = constants if constants is not None else {}

        if any(x is not None for x in [storage_zones_df, mrf_factors_daily_df,
                                        mrf_factors_monthly_df, constants]):
            self._validate()

    @property
    def STORAGE_LEVELS(self):
        """Zone threshold curve names derived from storage_zones_df.

        Filters the DataFrame index to only the actual storage zone rows,
        excluding MRF factor rows, DOY/harmonic rows, etc. that share the
        same DataFrame for the default config (from_defaults sets
        mrf_factors_daily_df = storage_zones_df.copy()).

        For N-zone configs the DataFrame contains only zone rows (zone_1..zone_N).
        For the default 7-level config the DataFrame contains all profile rows;
        we filter to those matching _DEFAULT_STORAGE_LEVELS.
        """
        if self.storage_zones_df is None:
            return self._DEFAULT_STORAGE_LEVELS
        idx = list(self.storage_zones_df.index)
        # N-zone configs: all rows are zone rows with 'zone_N' naming
        if idx and any(n.startswith('zone_') for n in idx):
            return [n for n in idx if n.startswith('zone_')]
        # Default config: filter to the 6 known zone threshold level names
        return [l for l in idx if l in self._DEFAULT_STORAGE_LEVELS]

    @property
    def DROUGHT_LEVELS(self):
        storage = self.STORAGE_LEVELS
        if not storage:
            return self._DEFAULT_DROUGHT_LEVELS
        normal = 'zone_0' if storage[0].startswith('zone_') else 'level1a'
        return [normal] + storage

    @property
    def n_zones(self):
        return len(self.STORAGE_LEVELS)

    @property
    def n_drought_levels(self):
        return len(self.DROUGHT_LEVELS)

    @classmethod
    def from_defaults(cls, data_dir: Optional[Union[str, Path]] = None):
        """
        Load default NYC operations configuration from CSV files.

        Parameters:
            data_dir: Path to operational_constants directory. If None, uses package default.

        Returns:
            NYCOperationsConfig instance with default parameters
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / 'data' / 'operational_constants'
        else:
            data_dir = Path(data_dir)

        storage_zones_df = pd.read_csv(
            data_dir / 'ffmp_reservoir_operation_daily_profiles.csv',
            index_col='profile'
        )

        mrf_factors_daily_df = storage_zones_df.copy()

        mrf_factors_monthly_df = pd.read_csv(
            data_dir / 'ffmp_reservoir_operation_monthly_profiles.csv',
            index_col='profile'
        )

        constants_df = pd.read_csv(
            data_dir / 'constants.csv',
            index_col='parameter'
        )
        constants = constants_df['value'].to_dict()

        return cls(
            storage_zones_df=storage_zones_df,
            mrf_factors_daily_df=mrf_factors_daily_df,
            mrf_factors_monthly_df=mrf_factors_monthly_df,
            constants=constants
        )

    @classmethod
    def from_n_zones(cls,
                     n_zones: int,
                     data_dir: Optional[Union[str, Path]] = None) -> "NYCOperationsConfig":
        """
        Build an N-zone config by linearly interpolating the FFMP 6-zone defaults.

        Produces storage curves named ``zone_1..zone_N`` (N boundary curves) and
        drought levels ``zone_0..zone_N`` (N+1 levels, where ``zone_0`` is the
        implicit "above top curve" normal band and is NOT stored in
        ``storage_zones_df``).

        Notes
        -----
        - The ``flood_conservation_boundary`` used by NYCFloodRelease and
          NYCCombinedReleaseFactor stays fixed at 2 regardless of N (see FFMP
          Section 6). For N != 6 the semantic meaning of "top 2 drought levels
          are flood zone" may warrant revisiting.
        - N < 6 will emit a warning because the flood-zone convention is very
          permissive at small N.
        - Interpolation is linear over a normalized index in [0, 1] in both
          the zone and drought-level directions.

        Parameters
        ----------
        n_zones : int
            Number of storage zone boundary curves. Must be >= 2.
        data_dir : str or Path, optional
            Path to operational_constants directory for the default FFMP CSVs.
            If None, uses the package default.

        Returns
        -------
        NYCOperationsConfig
            A configuration with ``n_zones`` storage curves and ``n_zones + 1``
            drought levels.
        """
        if n_zones < 2:
            raise ValueError(f"n_zones must be >= 2, got {n_zones}")
        if n_zones < 6:
            warnings.warn(
                f"from_n_zones(n_zones={n_zones}): flood_conservation_boundary "
                "is kept at 2 by convention, which is very permissive at small "
                "N. Consider N >= 6 for FFMP-like flood semantics."
            )

        base = cls.from_defaults(data_dir)

        default_storage_levels = list(cls._DEFAULT_STORAGE_LEVELS)  # 6
        default_drought_levels = list(cls._DEFAULT_DROUGHT_LEVELS)  # 7
        n_default_storage = len(default_storage_levels)
        n_default_drought = len(default_drought_levels)

        date_cols = [c for c in base.storage_zones_df.columns if c != 'doy']
        n_cols = len(date_cols)

        # --- 1. Interpolate storage zone threshold profiles (6 → N) ---
        default_profiles = base.storage_zones_df.loc[
            default_storage_levels, date_cols
        ].values.astype(float)  # (6, 366)

        x_default = np.linspace(0.0, 1.0, n_default_storage)
        x_target = np.linspace(0.0, 1.0, n_zones)
        new_profiles = np.zeros((n_zones, n_cols))
        for col_idx in range(n_cols):
            new_profiles[:, col_idx] = np.interp(
                x_target, x_default, default_profiles[:, col_idx]
            )

        zone_names = [f"zone_{i+1}" for i in range(n_zones)]
        new_storage_df = pd.DataFrame(
            new_profiles, index=zone_names, columns=date_cols
        )
        new_storage_df.index.name = 'profile'

        # Monotonicity check (pywr ControlCurveIndex requires descending thresholds)
        for i in range(n_zones - 1):
            if not np.all(new_profiles[i, :] >= new_profiles[i + 1, :]):
                raise ValueError(
                    f"from_n_zones: interpolated zone thresholds are not "
                    f"monotonically decreasing between zone_{i+1} and zone_{i+2}. "
                    f"pywr ControlCurveIndex requires descending thresholds."
                )

        # --- 2. Build new constants dict ---
        drought_level_names = ["zone_0"] + zone_names  # N+1 names
        new_constants = {}
        for k, v in base.constants.items():
            is_level_delivery = any(
                k.startswith(f"{lev}_factor_delivery")
                for lev in default_drought_levels
            )
            if not is_level_delivery:
                new_constants[k] = v

        default_nyc = [float(base.constants[f"{lev}_factor_delivery_nyc"])
                       for lev in default_drought_levels]
        default_nj = [float(base.constants[f"{lev}_factor_delivery_nj"])
                      for lev in default_drought_levels]

        x_def_drought = np.linspace(0.0, 1.0, n_default_drought)
        x_tgt_drought = np.linspace(0.0, 1.0, n_zones + 1)
        interp_nyc = np.interp(x_tgt_drought, x_def_drought, default_nyc)
        interp_nj = np.interp(x_tgt_drought, x_def_drought, default_nj)

        for i, level in enumerate(drought_level_names):
            new_constants[f"{level}_factor_delivery_nyc"] = float(interp_nyc[i])
            new_constants[f"{level}_factor_delivery_nj"] = float(interp_nj[i])

        # --- 3. Build mrf_factors_daily_df ---
        # Mirrors the default: zone threshold rows live in the same DataFrame
        # as the MRF factor rows (see from_defaults which sets
        # mrf_factors_daily_df = storage_zones_df.copy()).
        mrf_daily_rows = {}
        for i, zn in enumerate(zone_names):
            mrf_daily_rows[zn] = new_profiles[i, :]

        for res in cls.RESERVOIRS:
            default_mrf = np.array([
                base.mrf_factors_daily_df.loc[
                    f"{lev}_factor_mrf_{res}", date_cols
                ].values.astype(float)
                for lev in default_drought_levels
            ])  # (7, 366)
            new_mrf = np.zeros((n_zones + 1, n_cols))
            for col_idx in range(n_cols):
                new_mrf[:, col_idx] = np.interp(
                    x_tgt_drought, x_def_drought, default_mrf[:, col_idx]
                )
            for j, level in enumerate(drought_level_names):
                mrf_daily_rows[f"{level}_factor_mrf_{res}"] = new_mrf[j, :]

        new_mrf_daily_df = pd.DataFrame.from_dict(
            mrf_daily_rows, orient='index', columns=date_cols
        )
        new_mrf_daily_df.index.name = 'profile'

        # --- 4. Build mrf_factors_monthly_df (delMontague / delTrenton) ---
        month_cols = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                      'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        mrf_monthly_rows = {}
        for loc in ['delMontague', 'delTrenton']:
            default_mrf = np.array([
                base.mrf_factors_monthly_df.loc[
                    f"{lev}_factor_mrf_{loc}", month_cols
                ].values.astype(float)
                for lev in default_drought_levels
            ])  # (7, 12)
            new_mrf = np.zeros((n_zones + 1, 12))
            for col_idx in range(12):
                new_mrf[:, col_idx] = np.interp(
                    x_tgt_drought, x_def_drought, default_mrf[:, col_idx]
                )
            for j, level in enumerate(drought_level_names):
                mrf_monthly_rows[f"{level}_factor_mrf_{loc}"] = new_mrf[j, :]

        new_mrf_monthly_df = pd.DataFrame.from_dict(
            mrf_monthly_rows, orient='index', columns=month_cols
        )
        new_mrf_monthly_df.index.name = 'profile'

        # Invariant: zone_0 must NOT appear as a storage threshold row
        assert "zone_0" not in new_storage_df.index, \
            "zone_0 is the implicit 'above top curve' band; it must not be a threshold row"

        return cls(
            storage_zones_df=new_storage_df,
            mrf_factors_daily_df=new_mrf_daily_df,
            mrf_factors_monthly_df=new_mrf_monthly_df,
            constants=new_constants,
        )

    def update_storage_zones(self,
                            zones_df: Optional[pd.DataFrame] = None,
                            level: Optional[str] = None,
                            daily_values: Optional[np.ndarray] = None):
        """
        Update storage zone thresholds.

        Parameters:
            zones_df: Complete DataFrame with all storage zones (replaces existing)
            level: Specific level to update (e.g., 'level2')
            daily_values: Array of 366 values for the specified level
        """
        if zones_df is not None:
            self.storage_zones_df = zones_df
        elif level is not None and daily_values is not None:
            if self.storage_zones_df is None:
                raise ValueError("Cannot update individual level without existing storage_zones_df")
            if level not in self.STORAGE_LEVELS:
                raise ValueError(f"Invalid level: {level}. Must be one of {self.STORAGE_LEVELS}")
            if len(daily_values) != 366:
                raise ValueError(f"daily_values must have 366 elements, got {len(daily_values)}")

            date_cols = [col for col in self.storage_zones_df.columns if col not in ['doy']]
            if len(date_cols) != 366:
                raise ValueError(f"Expected 366 date columns, found {len(date_cols)}")

            self.storage_zones_df.loc[level, date_cols] = daily_values

        self._validate()

    def update_mrf_factors(self,
                          factors_daily_df: Optional[pd.DataFrame] = None,
                          factors_monthly_df: Optional[pd.DataFrame] = None,
                          reservoir: Optional[str] = None,
                          level: Optional[str] = None,
                          daily_factors: Optional[np.ndarray] = None,
                          monthly_factors: Optional[np.ndarray] = None):
        """
        Update MRF release factors.

        Parameters:
            factors_daily_df: Complete DataFrame with all daily MRF factors
            factors_monthly_df: Complete DataFrame with all monthly MRF factors
            reservoir: Specific reservoir to update (e.g., 'cannonsville')
            level: Specific drought level to update (e.g., 'level2')
            daily_factors: Array of 366 values for daily factors
            monthly_factors: Array of 12 values for monthly factors
        """
        if factors_daily_df is not None:
            self.mrf_factors_daily_df = factors_daily_df
        elif reservoir is not None and level is not None and daily_factors is not None:
            if self.mrf_factors_daily_df is None:
                raise ValueError("Cannot update individual factor without existing mrf_factors_daily_df")
            if reservoir not in self.RESERVOIRS:
                raise ValueError(f"Invalid reservoir: {reservoir}. Must be one of {self.RESERVOIRS}")
            if level not in self.DROUGHT_LEVELS:
                raise ValueError(f"Invalid level: {level}. Must be one of {self.DROUGHT_LEVELS}")
            if len(daily_factors) != 366:
                raise ValueError(f"daily_factors must have 366 elements, got {len(daily_factors)}")

            profile_name = f"{level}_factor_mrf_{reservoir}"
            date_cols = [col for col in self.mrf_factors_daily_df.columns if col not in ['doy']]
            self.mrf_factors_daily_df.loc[profile_name, date_cols] = daily_factors

        if factors_monthly_df is not None:
            self.mrf_factors_monthly_df = factors_monthly_df
        elif monthly_factors is not None:
            if self.mrf_factors_monthly_df is None:
                raise ValueError("Cannot update monthly factors without existing mrf_factors_monthly_df")
            if len(monthly_factors) != 12:
                raise ValueError(f"monthly_factors must have 12 elements, got {len(monthly_factors)}")

        self._validate()

    def update_delivery_constraints(self,
                                   max_nyc_delivery: Optional[float] = None,
                                   max_nj_daily: Optional[float] = None,
                                   max_nj_monthly_avg: Optional[float] = None,
                                   drought_factors_nyc: Optional[np.ndarray] = None,
                                   drought_factors_nj: Optional[np.ndarray] = None,
                                   delivery_reset_month: Optional[int] = None,
                                   delivery_reset_day: Optional[int] = None):
        """
        Update NYC and NJ delivery constraints.

        Parameters:
            max_nyc_delivery: Maximum baseline NYC delivery (MGD)
            max_nj_daily: Maximum daily NJ delivery (MGD)
            max_nj_monthly_avg: Maximum monthly average NJ delivery (MGD)
            drought_factors_nyc: Array of 7 factors for NYC (level1a-5)
            drought_factors_nj: Array of 7 factors for NJ (level1a-5)
            delivery_reset_month: Month for NYC delivery reset (default 5 = May)
            delivery_reset_day: Day for NYC delivery reset (default 31)
        """
        if max_nyc_delivery is not None:
            self.constants['max_flow_baseline_delivery_nyc'] = max_nyc_delivery

        if max_nj_daily is not None:
            self.constants['max_flow_baseline_daily_delivery_nj'] = max_nj_daily

        if max_nj_monthly_avg is not None:
            self.constants['max_flow_baseline_monthlyAvg_delivery_nj'] = max_nj_monthly_avg

        if drought_factors_nyc is not None:
            if len(drought_factors_nyc) != self.n_drought_levels:
                raise ValueError(f"drought_factors_nyc must have {self.n_drought_levels} elements")
            for i, level in enumerate(self.DROUGHT_LEVELS):
                self.constants[f'{level}_factor_delivery_nyc'] = drought_factors_nyc[i]

        if drought_factors_nj is not None:
            if len(drought_factors_nj) != self.n_drought_levels:
                raise ValueError(f"drought_factors_nj must have {self.n_drought_levels} elements")
            for i, level in enumerate(self.DROUGHT_LEVELS):
                self.constants[f'{level}_factor_delivery_nj'] = drought_factors_nj[i]

        if delivery_reset_month is not None:
            self.constants['delivery_reset_month'] = delivery_reset_month

        if delivery_reset_day is not None:
            self.constants['delivery_reset_day'] = delivery_reset_day

        self._validate()

    def update_flood_limits(self,
                           max_release_cannonsville: Optional[float] = None,
                           max_release_pepacton: Optional[float] = None,
                           max_release_neversink: Optional[float] = None):
        """
        Update maximum flood release limits.

        Parameters:
            max_release_cannonsville: Maximum flood release for Cannonsville (CFS)
            max_release_pepacton: Maximum flood release for Pepacton (CFS)
            max_release_neversink: Maximum flood release for Neversink (CFS)
        """
        if max_release_cannonsville is not None:
            self.constants['flood_max_release_cannonsville_cfs'] = max_release_cannonsville

        if max_release_pepacton is not None:
            self.constants['flood_max_release_pepacton_cfs'] = max_release_pepacton

        if max_release_neversink is not None:
            self.constants['flood_max_release_neversink_cfs'] = max_release_neversink

        self._validate()

    def update_mrf_baselines(self,
                            cannonsville: Optional[float] = None,
                            pepacton: Optional[float] = None,
                            neversink: Optional[float] = None,
                            montague: Optional[float] = None,
                            trenton: Optional[float] = None):
        """
        Update minimum required flow baselines.

        Parameters:
            cannonsville: Baseline MRF for Cannonsville (MGD)
            pepacton: Baseline MRF for Pepacton (MGD)
            neversink: Baseline MRF for Neversink (MGD)
            montague: Baseline flow target for Delaware at Montague (MGD)
            trenton: Baseline flow target for Delaware at Trenton (MGD)
        """
        if cannonsville is not None:
            self.constants['mrf_baseline_cannonsville'] = cannonsville

        if pepacton is not None:
            self.constants['mrf_baseline_pepacton'] = pepacton

        if neversink is not None:
            self.constants['mrf_baseline_neversink'] = neversink

        if montague is not None:
            self.constants['mrf_baseline_delMontague'] = montague

        if trenton is not None:
            self.constants['mrf_baseline_delTrenton'] = trenton

        self._validate()

    def get_constant(self, name: str, default=None):
        """Get a constant value by name, with optional default."""
        return self.constants.get(name, default)

    def get_storage_zone_profile(self, level: str) -> np.ndarray:
        """
        Get storage zone threshold profile for a specific level.

        Parameters:
            level: Storage level name (e.g., 'level2')

        Returns:
            Array of 366 daily threshold values
        """
        if self.storage_zones_df is None:
            raise ValueError("No storage zones data loaded")

        if level not in self.storage_zones_df.index:
            raise ValueError(f"Level {level} not found in storage zones")

        date_cols = [col for col in self.storage_zones_df.columns if col not in ['doy']]
        return self.storage_zones_df.loc[level, date_cols].values.astype(float)

    def get_mrf_factor_profile(self, profile_name: str, daily: bool = True) -> np.ndarray:
        """
        Get MRF factor profile by name.

        Parameters:
            profile_name: Profile name (e.g., 'level2_factor_mrf_cannonsville')
            daily: If True, return daily profile (366 values), else monthly (12 values)

        Returns:
            Array of factor values
        """
        df = self.mrf_factors_daily_df if daily else self.mrf_factors_monthly_df

        if df is None:
            raise ValueError(f"No {'daily' if daily else 'monthly'} MRF factors data loaded")

        if profile_name not in df.index:
            raise ValueError(f"Profile {profile_name} not found in MRF factors")

        if daily:
            date_cols = [col for col in df.columns if col not in ['doy']]
        else:
            date_cols = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                        'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

        return df.loc[profile_name, date_cols].values.astype(float)

    def _validate(self):
        """
        Validate configuration parameters for consistency and completeness.

        Raises warnings for potential issues, errors for critical problems.
        """
        if self.storage_zones_df is not None:
            for level in self.STORAGE_LEVELS:
                if level not in self.storage_zones_df.index:
                    warnings.warn(f"Storage level {level} not found in storage_zones_df")

            date_cols = [col for col in self.storage_zones_df.columns if col not in ['doy']]
            if len(date_cols) != 366:
                raise ValueError(f"storage_zones_df must have 366 date columns, got {len(date_cols)}")

        if self.mrf_factors_daily_df is not None:
            date_cols = [col for col in self.mrf_factors_daily_df.columns if col not in ['doy']]
            if len(date_cols) != 366:
                raise ValueError(f"mrf_factors_daily_df must have 366 date columns, got {len(date_cols)}")

        if self.mrf_factors_monthly_df is not None:
            month_cols = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                         'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
            for col in month_cols:
                if col not in self.mrf_factors_monthly_df.columns:
                    raise ValueError(f"mrf_factors_monthly_df missing column: {col}")

        required_constants = [
            'mrf_baseline_cannonsville',
            'mrf_baseline_pepacton',
            'mrf_baseline_neversink',
            'mrf_baseline_delMontague',
            'mrf_baseline_delTrenton',
            'max_flow_baseline_delivery_nyc',
            'max_flow_baseline_daily_delivery_nj',
            'max_flow_baseline_monthlyAvg_delivery_nj'
        ]

        for const in required_constants:
            if const not in self.constants:
                warnings.warn(f"Required constant {const} not found in configuration")

        for level in self.DROUGHT_LEVELS:
            for delivery in ['nyc', 'nj']:
                key = f'{level}_factor_delivery_{delivery}'
                if key not in self.constants:
                    warnings.warn(f"Delivery factor {key} not found in configuration")

    def to_csv(self, output_dir: Union[str, Path]):
        """
        Export configuration to CSV files.

        Parameters:
            output_dir: Directory to write CSV files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.storage_zones_df is not None:
            self.storage_zones_df.to_csv(
                output_dir / 'ffmp_reservoir_operation_daily_profiles.csv'
            )

        if self.mrf_factors_monthly_df is not None:
            self.mrf_factors_monthly_df.to_csv(
                output_dir / 'ffmp_reservoir_operation_monthly_profiles.csv'
            )

        if self.constants:
            constants_df = pd.DataFrame([
                {'parameter': k, 'value': v, 'units': ''}
                for k, v in self.constants.items()
            ])
            constants_df.to_csv(
                output_dir / 'constants.csv',
                index=False
            )

    def copy(self):
        """Create a deep copy of this configuration."""
        return NYCOperationsConfig(
            storage_zones_df=self.storage_zones_df.copy() if self.storage_zones_df is not None else None,
            mrf_factors_daily_df=self.mrf_factors_daily_df.copy() if self.mrf_factors_daily_df is not None else None,
            mrf_factors_monthly_df=self.mrf_factors_monthly_df.copy() if self.mrf_factors_monthly_df is not None else None,
            constants=self.constants.copy()
        )
