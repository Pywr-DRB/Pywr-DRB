"""
Fitting helpers: derive STARFIT parameters from observed storage climatology
and dataset inflows.

Approach
--------
1. NOR bands: fit the clipped-sinusoid form used by STARFIT
   (clip(mu + alpha*sin(2c) + beta*cos(2c), min, max), c = pi*doy/365)
   to smoothed day-of-year quantile curves of observed storage.
2. Mean flow: set Adjusted_MEANFLOW_MGD to the dataset catchment inflow mean
   so the standardized-inflow response term is centered correctly.
3. Release harmonic: choose (alpha1, alpha2, beta1, beta2, c) by least squares
   so the in-NOR release reproduces the mass-balance release implied by the
   dataset inflow climatology and the observed storage trajectory:
       R*(d) = I_clim(d) - dS*/dt(d)
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

DOY = np.arange(1, 367)
_C = np.pi / 365 * DOY
SIN2, COS2 = np.sin(2 * _C), np.cos(2 * _C)
SIN4, COS4 = np.sin(4 * _C), np.cos(4 * _C)


def doy_climatology(series, stat="median", q=None, smooth_days=15):
    """
    Smoothed day-of-year climatology (366 values) of a daily series.

    Uses a circular rolling mean of the per-DOY statistic.
    """
    grouped = series.groupby(series.index.dayofyear)
    if q is not None:
        curve = grouped.quantile(q)
    elif stat == "median":
        curve = grouped.median()
    else:
        curve = grouped.mean()
    curve = curve.reindex(DOY).interpolate(limit_direction="both")
    # circular smoothing
    ext = pd.concat([curve.iloc[-smooth_days:], curve, curve.iloc[:smooth_days]])
    sm = ext.rolling(smooth_days, center=True, min_periods=1).mean()
    return sm.iloc[smooth_days:smooth_days + 366].to_numpy()


def nor_curve(mu, alpha, beta, mn, mx):
    """Evaluate the STARFIT NOR bound (percent units) for all DOY."""
    return np.clip(mu + alpha * SIN2 + beta * COS2, mn, mx)


def fit_nor_bound(target_pct, fix_min=None, fix_max=None):
    """
    Fit (mu, alpha, beta, min, max) of a NOR bound to a target curve (percent).

    The clip levels default to the target's low/high plateaus but can be fixed.
    Returns dict of the 5 parameters.
    """
    t = np.asarray(target_pct, dtype=float)
    mn0 = fix_min if fix_min is not None else float(np.percentile(t, 2))
    mx0 = fix_max if fix_max is not None else float(np.percentile(t, 98))

    # initial phase/amplitude from first-harmonic regression, scaled up so
    # the sinusoid saturates against the clips (trapezoid shape)
    X = np.column_stack([SIN2, COS2, np.ones_like(SIN2)])
    coef, *_ = np.linalg.lstsq(X, t, rcond=None)
    a0, b0, m0 = coef
    scale = 2.0

    def residual(p):
        mu, alpha, beta = p[0], p[1], p[2]
        mn = fix_min if fix_min is not None else p[3]
        mx = fix_max if fix_max is not None else p[4]
        return nor_curve(mu, alpha, beta, mn, mx) - t

    p0 = [m0, a0 * scale, b0 * scale]
    if fix_min is None:
        p0.append(mn0)
    if fix_max is None:
        p0.append(mx0)
    # pad p0 so indices [3],[4] exist when only one clip is fixed
    while len(p0) < 5:
        p0.append(mn0 if len(p0) == 3 else mx0)

    res = least_squares(residual, p0, method="lm", max_nfev=5000)
    p = res.x
    mu, alpha, beta = p[0], p[1], p[2]
    mn = fix_min if fix_min is not None else p[3]
    mx = fix_max if fix_max is not None else p[4]
    if mn > mx:
        mn, mx = mx, mn
    return {"mu": float(mu), "alpha": float(alpha), "beta": float(beta),
            "min": float(mn), "max": float(mx)}


def fit_release_harmonic(I_clim, S_target_mg, nor_lo_frac, nor_hi_frac,
                         I_bar, cap_mg, p1, p2, r_min=None, r_max=None):
    """
    Least-squares fit of (Release_alpha1, alpha2, beta1, beta2, c).

    Parameters
    ----------
    I_clim : array (366,)
        Smoothed dataset inflow climatology (MGD).
    S_target_mg : array (366,)
        Smoothed target (observed) storage climatology in MG.
    nor_lo_frac, nor_hi_frac : arrays (366,)
        Fitted NOR bounds as fractions (0-1).
    I_bar : float
        Adjusted_MEANFLOW_MGD to be used with these parameters.
    cap_mg : float
        Adjusted_CAP_MG to be used with these parameters.
    p1, p2 : float
        Release_p1 (storage feedback) and Release_p2 (inflow response).
    r_min, r_max : float, optional
        Clip the mass-balance target release before fitting.

    Returns
    -------
    dict with Release_alpha1/alpha2/beta1/beta2/c and diagnostics.
    """
    # implied release from mass balance on the target trajectory
    dSdt = np.gradient(S_target_mg, edge_order=2)
    # circular gradient at the year boundary
    dSdt[0] = (S_target_mg[1] - S_target_mg[-1]) / 2
    dSdt[-1] = (S_target_mg[0] - S_target_mg[-2]) / 2
    R_star = I_clim - dSdt
    if r_min is not None:
        R_star = np.maximum(R_star, r_min)
    if r_max is not None:
        R_star = np.minimum(R_star, r_max)

    S_hat = S_target_mg / cap_mg
    A_t = (S_hat - nor_lo_frac) / nor_hi_frac
    I_hat = (I_clim - I_bar) / I_bar

    y = R_star / I_bar - 1.0 - p2 * I_hat - p1 * A_t
    X = np.column_stack([SIN2, SIN4, COS2, COS4, np.ones_like(SIN2)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a1, a2, b1, b2, c = [float(v) for v in coef]

    fitted = I_bar * (X @ coef + 1.0 + p2 * I_hat + p1 * A_t)
    return {
        "Release_alpha1": a1, "Release_alpha2": a2,
        "Release_beta1": b1, "Release_beta2": b2, "Release_c": c,
        "_R_star": R_star, "_fitted": fitted,
    }


def build_override(nor_hi, nor_lo, release, I_bar, cap_mg=None,
                   p1=None, p2=None, release_min=None, release_max=None):
    """Assemble an OVERRIDES-style dict from fitted pieces."""
    out = {
        "NORhi_mu": nor_hi["mu"], "NORhi_alpha": nor_hi["alpha"],
        "NORhi_beta": nor_hi["beta"], "NORhi_min": nor_hi["min"],
        "NORhi_max": nor_hi["max"],
        "NORlo_mu": nor_lo["mu"], "NORlo_alpha": nor_lo["alpha"],
        "NORlo_beta": nor_lo["beta"], "NORlo_min": nor_lo["min"],
        "NORlo_max": nor_lo["max"],
        "Release_alpha1": release["Release_alpha1"],
        "Release_alpha2": release["Release_alpha2"],
        "Release_beta1": release["Release_beta1"],
        "Release_beta2": release["Release_beta2"],
        "Release_c": release["Release_c"],
        "Adjusted_MEANFLOW_MGD": float(I_bar),
    }
    if cap_mg is not None:
        out["Adjusted_CAP_MG"] = float(cap_mg)
    if p1 is not None:
        out["Release_p1"] = float(p1)
    if p2 is not None:
        out["Release_p2"] = float(p2)
    if release_min is not None:
        out["Release_min"] = float(release_min)
    if release_max is not None:
        out["Release_max"] = float(release_max)
    return out
