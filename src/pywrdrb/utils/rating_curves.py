"""
Rating curve utilities for loading and interpolating USGS stage-discharge data.

USGS rating curve files are in NWIS format:
- Header lines start with '#'
- Data columns: INDEP (stage, ft), SHIFT (adjustment, ft), DEP (discharge, cfs), STOR
- Logarithmic interpolation between points

Change Log:
TJA, 2026-01-08, Initial implementation for flood monitoring.
TJA, 2026-01-08, Removed extrapolation warnings (now handled in parameters/flood_stage.py).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import interpolate
import warnings


class RatingCurve:
    """
    Rating curve for converting discharge to stage or vice versa.

    Attributes
    ----------
    site_no : str
        USGS site number
    stage : np.ndarray
        Stage values (ft above gage datum)
    discharge : np.ndarray
        Discharge values (cfs)
    interpolator_q_to_h : scipy interpolator
        Discharge → Stage converter
    interpolator_h_to_q : scipy interpolator
        Stage → Discharge converter
    stage_min : float
        Minimum valid stage (ft)
    stage_max : float
        Maximum valid stage (ft)
    discharge_min : float
        Minimum valid discharge (cfs)
    discharge_max : float
        Maximum valid discharge (cfs)
    """

    def __init__(self, site_no, stage, discharge):
        """
        Parameters
        ----------
        site_no : str
            USGS site number
        stage : array-like
            Stage values in feet
        discharge : array-like
            Discharge values in cfs
        """
        self.site_no = site_no
        self.stage = np.array(stage)
        self.discharge = np.array(discharge)

        # Create interpolators (log-log for power law relationship)
        self._create_interpolators()

    def _create_interpolators(self):
        """Create interpolation functions for both directions."""
        # Remove any duplicate or non-monotonic points
        df = pd.DataFrame({'stage': self.stage, 'discharge': self.discharge})
        df = df.drop_duplicates(subset=['stage']).sort_values('stage')
        df = df[df['discharge'] > 0]  # Positive discharge only

        # Discharge to Stage (Q → h)
        # Use log-log interpolation for better behavior
        log_q = np.log10(df['discharge'].values)
        stage = df['stage'].values

        self.interpolator_q_to_h = interpolate.interp1d(
            log_q, stage,
            kind='linear',
            bounds_error=False,
            fill_value=(stage[0], stage[-1])  # Extrapolate with endpoints
        )

        # Stage to Discharge (h → Q)
        log_q_from_h = interpolate.interp1d(
            stage, log_q,
            kind='linear',
            bounds_error=False,
            fill_value=(log_q[0], log_q[-1])
        )
        self.interpolator_h_to_q = lambda h: 10 ** log_q_from_h(h)

        # Store valid ranges
        self.stage_min = stage[0]
        self.stage_max = stage[-1]
        self.discharge_min = df['discharge'].values[0]
        self.discharge_max = df['discharge'].values[-1]

    def discharge_to_stage(self, discharge_cfs):
        """
        Convert discharge to stage.

        Parameters
        ----------
        discharge_cfs : float or array-like
            Discharge in cubic feet per second

        Returns
        -------
        float or np.ndarray
            Stage in feet above gage datum

        Notes
        -----
        Values outside the rating curve range are extrapolated using endpoint values.
        Warnings about extrapolation are handled by the calling parameter, not here.
        """
        discharge_cfs = np.atleast_1d(discharge_cfs)

        # Clip to valid range (extrapolate with endpoints)
        discharge_clipped = np.clip(discharge_cfs, self.discharge_min, self.discharge_max)

        log_q = np.log10(discharge_clipped)
        stage = self.interpolator_q_to_h(log_q)

        return stage.item() if stage.size == 1 else stage

    def stage_to_discharge(self, stage_ft):
        """
        Convert stage to discharge.

        Parameters
        ----------
        stage_ft : float or array-like
            Stage in feet above gage datum

        Returns
        -------
        float or np.ndarray
            Discharge in cubic feet per second

        Notes
        -----
        Values outside the rating curve range are extrapolated using endpoint values.
        """
        stage_ft = np.atleast_1d(stage_ft)

        # Clip to valid range (extrapolate with endpoints)
        stage_clipped = np.clip(stage_ft, self.stage_min, self.stage_max)

        discharge = self.interpolator_h_to_q(stage_clipped)

        return discharge.item() if discharge.size == 1 else discharge


def load_usgs_rating_curve(filepath):
    """
    Load USGS NWIS rating curve file.

    Parameters
    ----------
    filepath : str or Path
        Path to USGS rating curve text file

    Returns
    -------
    RatingCurve
        Rating curve object with interpolation functions
    """
    filepath = Path(filepath)

    # Read file and extract metadata
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find site number from header
    site_no = None
    for line in lines:
        if 'STATION' in line and 'NUMBER=' in line:
            site_no = line.split('NUMBER=')[1].split()[0].strip('"')
            break

    if site_no is None:
        site_no = filepath.stem  # Use filename as fallback

    # Read data table (skip header lines starting with #)
    data_lines = [line for line in lines if not line.startswith('#')]

    # Parse data
    data = []
    header_found = False
    for line in data_lines:
        if 'INDEP' in line:
            header_found = True
            continue
        if not header_found:
            continue

        parts = line.strip().split('\t')
        if len(parts) >= 3:
            try:
                stage = float(parts[0])
                shift = float(parts[1])
                discharge = float(parts[2])
                # Apply shift to stage (already included in USGS files typically)
                adjusted_stage = stage + shift
                data.append((adjusted_stage, discharge))
            except ValueError:
                continue  # Skip non-numeric lines

    if not data:
        raise ValueError(f"No valid rating data found in {filepath}")

    # Create DataFrame and rating curve
    df = pd.DataFrame(data, columns=['stage', 'discharge'])

    rating_curve = RatingCurve(
        site_no=site_no,
        stage=df['stage'].values,
        discharge=df['discharge'].values
    )

    print(f"Loaded rating curve for site {site_no}:")
    print(f"  Stage range: {rating_curve.stage_min:.2f} - {rating_curve.stage_max:.2f} ft")
    print(f"  Discharge range: {rating_curve.discharge_min:.1f} - {rating_curve.discharge_max:.1f} cfs")
    print(f"  Points: {len(df)}")

    return rating_curve


def load_all_flood_monitoring_curves(data_dir=None):
    """
    Load rating curves for all flood monitoring locations.

    Parameters
    ----------
    data_dir : str or Path, optional
        Path to rating_curves directory. If None, uses default location.

    Returns
    -------
    dict
        Dictionary mapping site numbers to RatingCurve objects
    """
    if data_dir is None:
        # Get the pywrdrb package directory and construct path to rating_curves
        import pywrdrb
        pkg_dir = Path(pywrdrb.__file__).parent
        data_dir = pkg_dir / 'data' / 'rating_curves'
    else:
        data_dir = Path(data_dir)

    rating_curves = {}

    # Load available rating curves
    for site_no in ['01426500', '01421000', '01436690']:
        filepath = data_dir / f'{site_no}.txt'
        if filepath.exists():
            rating_curves[site_no] = load_usgs_rating_curve(filepath)
        else:
            warnings.warn(f"Rating curve file not found: {filepath}", UserWarning)

    return rating_curves
