"""Volatility surface models: SABR and SVI parameterizations.

Implements industry-standard volatility surface parameterizations:
- SABR (Hagan et al., 2002): stochastic alpha-beta-rho model
- SVI (Gatheral, 2004): stochastic volatility inspired parameterization
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# SABR model — Hagan (2002) approximation
# ---------------------------------------------------------------------------


def sabr_implied_vol(
    F: float,
    K: float | np.ndarray,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float | np.ndarray:
    """Compute SABR implied volatility using the Hagan (2002) approximation.

    Parameters
    ----------
    F : float
        Forward price.
    K : float or array
        Strike price(s).
    T : float
        Time to expiry in years.
    alpha : float
        Initial volatility level (α > 0).
    beta : float
        CEV exponent (0 ≤ β ≤ 1).
    rho : float
        Correlation between forward and vol (-1 < ρ < 1).
    nu : float
        Vol-of-vol (ν ≥ 0).

    Returns
    -------
    float or array
        SABR implied Black volatility.
    """
    K = np.asarray(K, dtype=float)
    scalar_input = K.ndim == 0
    K = np.atleast_1d(K)

    # Avoid division by zero for ATM
    eps = 1e-12

    log_FK = np.log(F / K)
    atm_mask = np.abs(log_FK) < eps

    sigma = np.empty_like(K, dtype=float)

    # --- ATM formula ---
    if np.any(atm_mask):
        Fm = F ** (1.0 - beta)
        term1 = alpha / Fm
        corr = (
            1.0
            + (
                ((1.0 - beta) ** 2 / 24.0) * alpha**2 / (F ** (2.0 - 2.0 * beta))
                + 0.25 * rho * beta * nu * alpha / (F ** (1.0 - beta))
                + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
            )
            * T
        )
        sigma[atm_mask] = term1 * corr

    # --- Non-ATM formula ---
    non_atm = ~atm_mask
    if np.any(non_atm):
        K_na = K[non_atm]
        FK_na = F * K_na
        FK_mid_na = np.sqrt(FK_na)
        FK_beta_na = FK_mid_na ** (1.0 - beta)
        log_FK_na = np.log(F / K_na)

        z = (nu / alpha) * FK_beta_na * log_FK_na
        x = np.log((np.sqrt(1.0 - 2.0 * rho * z + z**2) + z - rho) / (1.0 - rho))

        # Protect against x = 0 (use safe division to avoid RuntimeWarning)
        safe_x = np.where(np.abs(x) < eps, 1.0, x)
        ratio = np.where(np.abs(x) < eps, 1.0, z / safe_x)

        # Denominator corrections
        term1 = (1.0 - beta) ** 2 / 24.0 * log_FK_na**2
        term2 = (1.0 - beta) ** 4 / 1920.0 * log_FK_na**4
        denom = FK_beta_na * (1.0 + term1 + term2)

        # Numerator correction
        corr = (
            1.0
            + (
                ((1.0 - beta) ** 2 / 24.0) * alpha**2 / (FK_mid_na ** (2.0 - 2.0 * beta))
                + 0.25 * rho * beta * nu * alpha / FK_beta_na
                + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
            )
            * T
        )

        sigma[non_atm] = (alpha / denom) * ratio * corr

    return float(sigma[0]) if scalar_input else sigma


def sabr_calibrate(
    market_vols: np.ndarray,
    F: float,
    strikes: np.ndarray,
    T: float,
    beta: float = 0.5,
    initial_guess: tuple[float, float, float] | None = None,
) -> dict[str, float]:
    """Calibrate SABR parameters (alpha, rho, nu) to market implied vols.

    Parameters
    ----------
    market_vols : array
        Market implied volatilities.
    F : float
        Forward price.
    strikes : array
        Strike prices corresponding to market_vols.
    T : float
        Time to expiry.
    beta : float
        Fixed CEV exponent (default 0.5).
    initial_guess : tuple, optional
        Initial (alpha, rho, nu). Defaults to (0.2, -0.3, 0.4).

    Returns
    -------
    dict
        {'alpha': ..., 'beta': ..., 'rho': ..., 'nu': ..., 'rmse': ...}
    """
    market_vols = np.asarray(market_vols, dtype=float)
    strikes = np.asarray(strikes, dtype=float)

    if initial_guess is None:
        initial_guess = (0.2, -0.3, 0.4)

    def objective(params):
        alpha, rho, nu = params
        if alpha <= 0 or nu < 0 or rho <= -1 or rho >= 1:
            return 1e10
        try:
            model_vols = sabr_implied_vol(F, strikes, T, alpha, beta, rho, nu)
            return np.sum((model_vols - market_vols) ** 2)
        except (ValueError, RuntimeWarning):
            return 1e10

    result = minimize(
        objective,
        x0=initial_guess,
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-10},
    )

    alpha, rho, nu = result.x
    model_vols = sabr_implied_vol(F, strikes, T, alpha, beta, rho, nu)
    rmse = float(np.sqrt(np.mean((model_vols - market_vols) ** 2)))

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "rho": float(rho),
        "nu": float(nu),
        "rmse": rmse,
    }


# ---------------------------------------------------------------------------
# SVI model — Gatheral (2004) raw parameterization
# ---------------------------------------------------------------------------


def svi_raw(
    k: float | np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> float | np.ndarray:
    """SVI raw parameterization for total implied variance.

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

    Parameters
    ----------
    k : float or array
        Log-moneyness ln(K/F).
    a : float
        Overall level of variance.
    b : float
        Slope parameter (b ≥ 0).
    rho : float
        Rotation parameter (-1 ≤ ρ ≤ 1).
    m : float
        Translation parameter.
    sigma : float
        Smoothing parameter (σ > 0).

    Returns
    -------
    float or array
        Total implied variance w(k).
    """
    k = np.asarray(k, dtype=float)
    dm = k - m
    return a + b * (rho * dm + np.sqrt(dm**2 + sigma**2))


def svi_implied_vol(
    k: float | np.ndarray,
    T: float,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> float | np.ndarray:
    """Convert SVI total variance to Black implied volatility.

    Parameters
    ----------
    k : float or array
        Log-moneyness ln(K/F).
    T : float
        Time to expiry.
    a, b, rho, m, sigma : float
        SVI raw parameters.

    Returns
    -------
    float or array
        Implied volatility (annualized).
    """
    w = svi_raw(k, a, b, rho, m, sigma)
    # Total variance -> annualized vol
    return np.sqrt(np.maximum(w, 0.0) / T)


def svi_calibrate(
    market_vols: np.ndarray,
    strikes: np.ndarray,
    F: float,
    T: float,
    initial_guess: tuple[float, float, float, float, float] | None = None,
) -> dict[str, float]:
    """Calibrate SVI raw parameters to market implied vols.

    Parameters
    ----------
    market_vols : array
        Market implied volatilities.
    strikes : array
        Strike prices.
    F : float
        Forward price.
    T : float
        Time to expiry.
    initial_guess : tuple, optional
        Initial (a, b, rho, m, sigma).

    Returns
    -------
    dict
        {'a': ..., 'b': ..., 'rho': ..., 'm': ..., 'sigma': ..., 'rmse': ...}
    """
    market_vols = np.asarray(market_vols, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    k = np.log(strikes / F)
    market_total_var = market_vols**2 * T

    if initial_guess is None:
        initial_guess = (
            float(np.mean(market_total_var)),
            0.1,
            -0.1,
            0.0,
            0.1,
        )

    def objective(params):
        a, b, rho, m, sigma_p = params
        if b < 0 or sigma_p <= 0 or rho < -1 or rho > 1:
            return 1e10
        if a + b * sigma_p * np.sqrt(1.0 - rho**2) < 0:
            return 1e10
        try:
            w = svi_raw(k, a, b, rho, m, sigma_p)
            if np.any(w < 0):
                return 1e10
            return float(np.sum((w - market_total_var) ** 2))
        except (ValueError, RuntimeWarning):
            return 1e10

    result = minimize(
        objective,
        x0=initial_guess,
        method="Nelder-Mead",
        options={"maxiter": 10000, "xatol": 1e-12, "fatol": 1e-12},
    )

    a, b, rho, m, sigma_p = result.x
    model_vols = svi_implied_vol(k, T, a, b, rho, m, sigma_p)
    rmse = float(np.sqrt(np.mean((model_vols - market_vols) ** 2)))

    return {
        "a": float(a),
        "b": float(b),
        "rho": float(rho),
        "m": float(m),
        "sigma": float(sigma_p),
        "rmse": rmse,
    }


def svi_surface(
    strikes: np.ndarray,
    expiries: np.ndarray,
    F: float | np.ndarray,
    params_per_expiry: list[dict[str, float]],
) -> np.ndarray:
    """Build an implied volatility surface from SVI parameters per expiry.

    Parameters
    ----------
    strikes : array, shape (n_strikes,)
        Strike prices.
    expiries : array, shape (n_expiries,)
        Expiry times.
    F : float or array of shape (n_expiries,)
        Forward price(s) per expiry.
    params_per_expiry : list of dicts
        Each dict must have keys 'a', 'b', 'rho', 'm', 'sigma'.

    Returns
    -------
    array, shape (n_expiries, n_strikes)
        Implied volatility surface.
    """
    strikes = np.asarray(strikes, dtype=float)
    expiries = np.asarray(expiries, dtype=float)
    F_arr = np.broadcast_to(np.asarray(F, dtype=float), expiries.shape)

    surface = np.empty((len(expiries), len(strikes)))
    for i, (T, f, params) in enumerate(zip(expiries, F_arr, params_per_expiry)):
        k = np.log(strikes / f)
        surface[i] = svi_implied_vol(
            k, T, params["a"], params["b"], params["rho"], params["m"], params["sigma"]
        )
    return surface
