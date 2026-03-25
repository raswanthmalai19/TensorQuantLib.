"""Local volatility model (Dupire).

The Dupire (1994) local volatility formula extracts local vol from
an implied volatility surface, enabling exact calibration to European
option prices.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm


def dupire_local_vol(
    strikes: np.ndarray,
    expiries: np.ndarray,
    iv_surface: np.ndarray,
    S: float,
    r: float = 0.0,
    q: float = 0.0,
) -> np.ndarray:
    """Compute Dupire local volatility from an implied vol surface.

    Uses the Dupire (1994) formula:

        sigma_loc^2(K,T) = (dC/dT + (r-q)*K*dC/dK + q*C)
                           / (0.5 * K^2 * d2C/dK^2)

    where C(K,T) is the Black-Scholes price at implied vol sigma(K,T).

    Parameters
    ----------
    strikes : array, shape (nK,)
        Strike grid.
    expiries : array, shape (nT,)
        Expiry grid (must be > 0).
    iv_surface : array, shape (nK, nT)
        Implied volatility surface, ``iv_surface[i,j] = sigma(K_i, T_j)``.
    S : float
        Spot price.
    r : float
        Risk-free rate.
    q : float
        Dividend yield.

    Returns
    -------
    local_vol : array, shape (nK, nT)
        Local volatility surface.
    """
    nK, nT = iv_surface.shape

    # Build BS prices on the (K, T) grid
    C = np.zeros_like(iv_surface)
    for i in range(nK):
        for j in range(nT):
            K, T, sig = strikes[i], expiries[j], iv_surface[i, j]
            d1 = (np.log(S / K) + (r - q + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
            d2 = d1 - sig * np.sqrt(T)
            C[i, j] = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    # Use spline interpolation for smooth derivatives
    spline = RectBivariateSpline(strikes, expiries, C, kx=3, ky=3)

    local_vol2 = np.zeros_like(iv_surface)
    for i in range(nK):
        for j in range(nT):
            K, T = strikes[i], expiries[j]
            dC_dT = spline(K, T, dx=0, dy=1).item()
            dC_dK = spline(K, T, dx=1, dy=0).item()
            d2C_dK2 = spline(K, T, dx=2, dy=0).item()

            numerator = dC_dT + (r - q) * K * dC_dK + q * C[i, j]
            denominator = 0.5 * K**2 * d2C_dK2

            if denominator > 1e-15:
                local_vol2[i, j] = numerator / denominator
            else:
                local_vol2[i, j] = iv_surface[i, j] ** 2

    # Clamp to avoid negative values (numerical noise)
    local_vol2 = np.maximum(local_vol2, 1e-8)
    return np.sqrt(local_vol2)


def local_vol_mc(
    S: float,
    K: float,
    T: float,
    r: float,
    strikes: np.ndarray,
    expiries: np.ndarray,
    local_vol_surface: np.ndarray,
    option_type: str = "call",
    q: float = 0.0,
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int | None = None,
) -> float:
    """Price an option using local vol MC simulation.

    Simulates paths under dS = (r-q)*S*dt + sigma_loc(S,t)*S*dW.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike.
    T : float
        Expiry.
    r : float
        Risk-free rate.
    strikes : array
        Strike grid for local vol interpolation.
    expiries : array
        Time grid for local vol interpolation.
    local_vol_surface : array, shape (nK, nT)
        Local volatility surface.
    option_type : str
        ``'call'`` or ``'put'``.
    q : float
        Dividend yield.
    n_paths : int
        MC paths.
    n_steps : int
        Time steps.
    seed : int, optional
        Random seed.

    Returns
    -------
    float
        MC price.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    spline = RectBivariateSpline(strikes, expiries, local_vol_surface, kx=3, ky=3)

    S_paths = np.full(n_paths, float(S))
    for step in range(n_steps):
        t = step * dt
        # Evaluate local vol for each path's current S
        # Clamp S to grid bounds for interpolation
        S_clamped = np.clip(S_paths, strikes[0], strikes[-1])
        t_clamped = min(max(t, expiries[0]), expiries[-1])
        sigma_loc = np.array([spline(s, t_clamped).item() for s in S_clamped])
        sigma_loc = np.maximum(sigma_loc, 1e-6)

        z = rng.standard_normal(n_paths)
        S_paths = S_paths * np.exp((r - q - 0.5 * sigma_loc**2) * dt + sigma_loc * np.sqrt(dt) * z)

    if option_type == "call":
        payoff = np.maximum(S_paths - K, 0.0)
    else:
        payoff = np.maximum(K - S_paths, 0.0)

    return float(np.exp(-r * T) * np.mean(payoff))
