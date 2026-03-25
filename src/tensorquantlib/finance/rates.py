"""Interest rate models: Vasicek, CIR, Nelson-Siegel, and yield curve bootstrap.

Implements short-rate models and yield curve fitting:
- Vasicek (1977): mean-reverting Gaussian short rate
- CIR (1985): mean-reverting square-root short rate
- Nelson-Siegel (1987): parametric yield curve
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Vasicek model
# ---------------------------------------------------------------------------


def vasicek_bond_price(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
) -> float:
    """Zero-coupon bond price under the Vasicek model.

    dr = kappa * (theta - r) * dt + sigma * dW

    Parameters
    ----------
    r0 : float
        Current short rate.
    kappa : float
        Mean reversion speed.
    theta : float
        Long-term rate level.
    sigma : float
        Volatility of short rate.
    T : float
        Time to maturity.

    Returns
    -------
    float
        Bond price P(0, T).
    """
    if abs(kappa) < 1e-12:
        # No mean reversion: simple exponential
        return float(np.exp(-r0 * T))

    B = (1.0 - np.exp(-kappa * T)) / kappa
    A = np.exp((theta - sigma**2 / (2.0 * kappa**2)) * (B - T) - sigma**2 / (4.0 * kappa) * B**2)
    return float(A * np.exp(-B * r0))


def vasicek_yield(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
) -> float:
    """Continuously compounded yield under Vasicek: y = -ln(P)/T.

    Returns
    -------
    float
        Yield for maturity T.
    """
    P = vasicek_bond_price(r0, kappa, theta, sigma, T)
    return float(-np.log(P) / T)


def vasicek_option_price(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T_option: float,
    T_bond: float,
    K: float,
    option_type: str = "call",
) -> float:
    """European option on a zero-coupon bond under Vasicek.

    Parameters
    ----------
    r0 : float
        Current short rate.
    kappa, theta, sigma : float
        Vasicek parameters.
    T_option : float
        Option expiry.
    T_bond : float
        Bond maturity (T_bond > T_option).
    K : float
        Strike price of the option.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    float
        Option price.
    """
    from scipy.stats import norm

    B_s = (1.0 - np.exp(-kappa * (T_bond - T_option))) / kappa
    sigma_p = sigma * B_s * np.sqrt((1.0 - np.exp(-2.0 * kappa * T_option)) / (2.0 * kappa))

    P_T = vasicek_bond_price(r0, kappa, theta, sigma, T_bond)
    P_s = vasicek_bond_price(r0, kappa, theta, sigma, T_option)

    d1 = np.log(P_T / (K * P_s)) / sigma_p + 0.5 * sigma_p
    d2 = d1 - sigma_p

    if option_type == "call":
        price = P_T * norm.cdf(d1) - K * P_s * norm.cdf(d2)
    else:
        price = K * P_s * norm.cdf(-d2) - P_T * norm.cdf(-d1)

    return float(price)


def vasicek_simulate(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int = 252,
    n_paths: int = 10_000,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate Vasicek short rate paths.

    Returns
    -------
    array, shape (n_steps + 1, n_paths)
        Simulated short rate paths including initial value.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    rates = np.empty((n_steps + 1, n_paths))
    rates[0] = r0

    for i in range(n_steps):
        Z = rng.standard_normal(n_paths)
        rates[i + 1] = rates[i] + kappa * (theta - rates[i]) * dt + sigma * np.sqrt(dt) * Z

    return rates


# ---------------------------------------------------------------------------
# CIR model
# ---------------------------------------------------------------------------


def cir_bond_price(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
) -> float:
    """Zero-coupon bond price under the CIR model.

    dr = kappa * (theta - r) * dt + sigma * sqrt(r) * dW

    Parameters
    ----------
    r0, kappa, theta, sigma, T : float
        CIR model parameters and maturity.

    Returns
    -------
    float
        Bond price P(0, T).
    """
    gamma = np.sqrt(kappa**2 + 2.0 * sigma**2)
    eg = np.exp(gamma * T)

    denom = (gamma + kappa) * (eg - 1.0) + 2.0 * gamma

    B = 2.0 * (eg - 1.0) / denom
    A = (2.0 * gamma * np.exp((kappa + gamma) * T / 2.0) / denom) ** (
        2.0 * kappa * theta / sigma**2
    )

    return float(A * np.exp(-B * r0))


def cir_yield(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
) -> float:
    """Continuously compounded yield under CIR."""
    P = cir_bond_price(r0, kappa, theta, sigma, T)
    return float(-np.log(P) / T)


def cir_simulate(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int = 252,
    n_paths: int = 10_000,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate CIR short rate paths (full truncation scheme).

    Returns
    -------
    array, shape (n_steps + 1, n_paths)
        Simulated short rate paths.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    rates = np.empty((n_steps + 1, n_paths))
    rates[0] = r0

    for i in range(n_steps):
        Z = rng.standard_normal(n_paths)
        r_pos = np.maximum(rates[i], 0.0)
        rates[i + 1] = rates[i] + kappa * (theta - r_pos) * dt + sigma * np.sqrt(r_pos * dt) * Z
        rates[i + 1] = np.maximum(rates[i + 1], 0.0)

    return rates


def feller_condition(kappa: float, theta: float, sigma: float) -> bool:
    """Check the Feller condition: 2*kappa*theta >= sigma^2.

    When satisfied, CIR process stays strictly positive.
    """
    return 2.0 * kappa * theta >= sigma**2


# ---------------------------------------------------------------------------
# Nelson-Siegel yield curve
# ---------------------------------------------------------------------------


def nelson_siegel(
    T: float | np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
    tau: float,
) -> float | np.ndarray:
    """Nelson-Siegel (1987) yield curve model.

    y(T) = beta0 + beta1 * (1 - exp(-T/tau)) / (T/tau)
                  + beta2 * ((1 - exp(-T/tau)) / (T/tau) - exp(-T/tau))

    Parameters
    ----------
    T : float or array
        Maturity/maturities.
    beta0 : float
        Long-term rate (level).
    beta1 : float
        Short-term component (slope).
    beta2 : float
        Medium-term component (curvature).
    tau : float
        Decay parameter (> 0).

    Returns
    -------
    float or array
        Yield(s) at the given maturity/maturities.
    """
    T = np.asarray(T, dtype=float)
    scalar_input = T.ndim == 0
    T = np.atleast_1d(T)

    x = T / tau
    # Handle T=0 limit
    factor1 = np.where(x < 1e-12, 1.0, (1.0 - np.exp(-x)) / x)
    factor2 = factor1 - np.exp(-x)

    y = beta0 + beta1 * factor1 + beta2 * factor2

    return float(y[0]) if scalar_input else y


def nelson_siegel_calibrate(
    maturities: np.ndarray,
    yields: np.ndarray,
    initial_guess: tuple[float, float, float, float] | None = None,
) -> dict[str, float]:
    """Calibrate Nelson-Siegel parameters to market yields.

    Parameters
    ----------
    maturities : array
        Observed maturities.
    yields : array
        Observed yields.
    initial_guess : tuple, optional
        Initial (beta0, beta1, beta2, tau).

    Returns
    -------
    dict
        {'beta0': ..., 'beta1': ..., 'beta2': ..., 'tau': ..., 'rmse': ...}
    """
    from scipy.optimize import minimize

    maturities = np.asarray(maturities, dtype=float)
    yields = np.asarray(yields, dtype=float)

    if initial_guess is None:
        initial_guess = (yields[-1], yields[0] - yields[-1], 0.0, 1.0)

    def objective(params):
        b0, b1, b2, tau = params
        if tau <= 0:
            return 1e10
        model = nelson_siegel(maturities, b0, b1, b2, tau)
        return float(np.sum((model - yields) ** 2))

    result = minimize(
        objective,
        x0=initial_guess,
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-12, "fatol": 1e-12},
    )

    b0, b1, b2, tau = result.x
    model_yields = nelson_siegel(maturities, b0, b1, b2, tau)
    rmse = float(np.sqrt(np.mean((model_yields - yields) ** 2)))

    return {
        "beta0": float(b0),
        "beta1": float(b1),
        "beta2": float(b2),
        "tau": float(tau),
        "rmse": rmse,
    }


# ---------------------------------------------------------------------------
# Yield curve bootstrap
# ---------------------------------------------------------------------------


def bootstrap_yield_curve(
    maturities: np.ndarray,
    prices: np.ndarray,
) -> np.ndarray:
    """Bootstrap zero-coupon yields from bond prices.

    Assumes prices are for zero-coupon bonds with face value 1.

    Parameters
    ----------
    maturities : array
        Bond maturities.
    prices : array
        Bond prices (face value 1).

    Returns
    -------
    array
        Continuously compounded zero rates.
    """
    maturities = np.asarray(maturities, dtype=float)
    prices = np.asarray(prices, dtype=float)
    return -np.log(prices) / maturities
