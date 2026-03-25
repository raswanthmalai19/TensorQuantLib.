"""
Heston stochastic volatility model — analytical pricing and Monte Carlo simulation.

The Heston (1993) model augments Black-Scholes with a mean-reverting variance process:

    dS = (r - q) S dt + sqrt(V) S dW_S
    dV = kappa (theta - V) dt + xi sqrt(V) dW_V
    Corr(dW_S, dW_V) = rho * dt

This module provides:
    heston_price_mc   -- Monte Carlo price (reference implementation)
    heston_price      -- Semi-analytic price via COS/Carr-Madan characteristic function
    heston_greeks     -- Delta, Gamma, Vega, Theta via finite differences on heston_price
    HestonCalibrator  -- Calibrate Heston parameters to market IV surface

References:
    Heston, S.L. (1993). A Closed-Form Solution for Options with Stochastic Volatility
    with Applications to Bond and Currency Options. RFS 6(2), 327-343.

    Fang, F. & Oosterlee, C.W. (2008). A Novel Pricing Method for European Options Based
    on Fourier-Cosine Series Expansions. SIAM J. Sci. Comput., 31(2), 826-848.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import norm as _scipy_norm

# Local alias used in QE scheme (norm PPF for V sampling)
norm_ppf = _scipy_norm.ppf



# ------------------------------------------------------------------ #
# Parameter container
# ------------------------------------------------------------------ #


@dataclass
class HestonParams:
    """Heston model parameters.

    Attributes:
        kappa: Mean-reversion speed (>0).
        theta: Long-run variance (>0).
        xi: Vol-of-vol (>0).
        rho: Correlation between asset and variance Brownian motions (-1 < rho < 1).
        v0: Initial variance (>0).
    """

    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7
    v0: float = 0.04

    def feller_satisfied(self) -> bool:
        """Check the Feller condition: 2 * kappa * theta > xi^2."""
        return 2.0 * self.kappa * self.theta > self.xi**2

    def to_array(self) -> np.ndarray:
        return np.array([self.kappa, self.theta, self.xi, self.rho, self.v0])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> HestonParams:
        return cls(kappa=arr[0], theta=arr[1], xi=arr[2], rho=arr[3], v0=arr[4])


# ------------------------------------------------------------------ #
# Characteristic function
# ------------------------------------------------------------------ #


def _heston_cf(
    phi: complex,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    params: HestonParams,
) -> complex:
    """Log-price characteristic function under Heston model (Heston 1993 formulation).

    Returns E[exp(i * phi * log(S_T))] under the risk-neutral measure.
    """
    kappa, theta, xi, rho, v0 = params.kappa, params.theta, params.xi, params.rho, params.v0

    x = np.log(S) + (r - q) * T

    # Standard Heston characteristic function (Albrecher et al. 2007 stable form)
    d = np.sqrt((rho * xi * 1j * phi - kappa) ** 2 + xi**2 * (1j * phi + phi**2))
    g_num = kappa - rho * xi * 1j * phi - d
    g_den = kappa - rho * xi * 1j * phi + d
    g = g_num / g_den

    exp_dT = np.exp(-d * T)
    C = (r - q) * 1j * phi * T + (kappa * theta / xi**2) * (
        g_num * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))
    )
    D = (g_num / xi**2) * (1.0 - exp_dT) / (1.0 - g * exp_dT)

    return np.exp(C + D * v0 + 1j * phi * x)


# ------------------------------------------------------------------ #
# Semi-analytic (Carr-Madan) pricing
# ------------------------------------------------------------------ #


def heston_price(
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    q: float = 0.0,
    option_type: str = "call",
    *,
    integration_limit: float = 200.0,
    n_points: int = 100,
) -> float:
    """Semi-analytic Heston call/put price via Gil-Pelaez inversion.

    Uses numerical integration of the characteristic function.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry (years).
        r: Risk-free rate.
        params: HestonParams instance.
        q: Continuous dividend yield.
        option_type: 'call' or 'put'.
        integration_limit: Upper limit for numerical integration (default 200).
        n_points: Number of integration points (higher = more accurate).

    Returns:
        Option price.

    Example:
        >>> p = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
        >>> price = heston_price(100.0, 100.0, 1.0, 0.05, p)
        >>> 5.0 < price < 20.0
        True
    """
    log_K = np.log(K)

    # Precompute CF(-i) for normalisation of the P1 integrand (S-numeraire probability)
    cf_minus_i = _heston_cf(-1j, S, K, T, r, q, params)

    def integrand_p1(phi: float) -> float:
        # P1 = Prob(S_T > K) under the S-numeraire (forward measure equivalent).
        # Uses the Radon-Nikodym derivative: CF_S(phi) = CF_Q(phi - i) / CF_Q(-i)
        # where CF_Q(u) = E^Q[exp(i*u*ln(S_T))].
        cf_val = _heston_cf(phi - 1j, S, K, T, r, q, params)
        num = np.exp(-1j * phi * log_K) * cf_val
        return float(np.real(num / (1j * phi * cf_minus_i)))

    def integrand_p2(phi: float) -> float:
        # P2 = Prob(S_T > K) under the risk-neutral measure Q.
        cf_val = _heston_cf(phi, S, K, T, r, q, params)
        num = np.exp(-1j * phi * log_K) * cf_val
        return float(np.real(num / (1j * phi)))

    # Gauss-Legendre quadrature (avoid singularity at phi=0)
    eps = 1e-6
    p1, _ = quad(integrand_p1, eps, integration_limit, limit=n_points)
    p2, _ = quad(integrand_p2, eps, integration_limit, limit=n_points)

    P1 = 0.5 + p1 / np.pi
    P2 = 0.5 + p2 / np.pi

    call_price = S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
    call_price = float(call_price)

    if option_type == "call":
        return max(call_price, 0.0)
    else:
        # Put via put-call parity
        return float(call_price - S * np.exp(-q * T) + K * np.exp(-r * T))


# ------------------------------------------------------------------ #
# Monte Carlo reference implementation
# ------------------------------------------------------------------ #


def heston_price_mc(
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    q: float = 0.0,
    option_type: str = "call",
    *,
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int | None = None,
    return_stderr: bool = False,
    scheme: str = "qe",
) -> Union[float, tuple[float, float]]:
    """Monte Carlo Heston price using the Quadratic-Exponential (QE) discretisation scheme.

    The QE scheme (Andersen 2008) provides high accuracy for the CIR variance
    process, virtually eliminating the discretisation bias of Euler-Maruyama.

    Args:
        S: Spot.
        K: Strike.
        T: Time to expiry (years).
        r: Risk-free rate.
        params: HestonParams.
        q: Dividend yield.
        option_type: 'call' or 'put'.
        n_paths: Number of simulation paths (default 100k).
        n_steps: Time steps (default 252 = daily).
        seed: Random seed.
        return_stderr: If True, returns (price, stderr) tuple.
        scheme: Discretisation scheme — 'qe' (default, Andersen 2008) or
                'euler' (Euler full-truncation, faster but biased).

    Returns:
        Price, or (price, stderr) if return_stderr=True.

    References:
        Andersen, L.B.G. (2008). Efficient Simulation of the Heston Stochastic
        Volatility Model. Journal of Computational Finance, 11(3).

    Example:
        >>> p = HestonParams()
        >>> price, se = heston_price_mc(100, 100, 1.0, 0.05, p, return_stderr=True, seed=42)
        >>> 5.0 < price < 20.0
        True
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    kappa, theta, xi, rho, v0 = params.kappa, params.theta, params.xi, params.rho, params.v0

    S_t = np.full(n_paths, S, dtype=float)
    V_t = np.full(n_paths, v0, dtype=float)

    if scheme == "euler":
        # Euler-Maruyama with full-truncation for V (fast but biased)
        for _ in range(n_steps):
            z1 = rng.standard_normal(n_paths)
            z2 = rng.standard_normal(n_paths)
            zv = rho * z1 + np.sqrt(1.0 - rho**2) * z2

            V_t_pos = np.maximum(V_t, 0.0)
            sqrt_V = np.sqrt(V_t_pos)

            S_t = S_t * np.exp((r - q - 0.5 * V_t_pos) * dt + sqrt_V * z1 * sqrt_dt)
            V_t = V_t + kappa * (theta - V_t_pos) * dt + xi * sqrt_V * zv * sqrt_dt
            V_t = np.maximum(V_t, 0.0)
    else:
        # Quadratic-Exponential (QE) scheme (Andersen 2008) for variance
        # Precompute constants
        exp_kdt = np.exp(-kappa * dt)
        xi2 = xi**2
        K0 = -rho * kappa * theta * dt / xi
        K1 = 0.5 * dt * (kappa * rho / xi - 0.5) - rho / xi
        K2 = 0.5 * dt * (kappa * rho / xi - 0.5) + rho / xi
        K3 = 0.5 * dt * (1.0 - rho**2)
        K4 = K3  # same by symmetry

        # QE critical ratio threshold (Andersen suggests psi_c = 1.5)
        psi_c = 1.5

        for _ in range(n_steps):
            z1 = rng.standard_normal(n_paths)
            u_v = rng.uniform(0.0, 1.0, n_paths)  # for V update

            # QE step for V_t+dt given V_t
            # Conditional mean and variance of V(t+dt)|V(t) for CIR process
            m = theta + (V_t - theta) * exp_kdt
            s2 = (
                V_t * xi2 * exp_kdt / kappa * (1.0 - exp_kdt)
                + theta * xi2 / (2.0 * kappa) * (1.0 - exp_kdt) ** 2
            )
            s2 = np.maximum(s2, 0.0)
            psi = s2 / (m**2)

            # High-psi regime (exponential mixture): psi > psi_c
            # Low-psi regime (quadratic normal): psi <= psi_c
            hi = psi > psi_c

            # --- Low-psi branch: quadratic approximation ---
            inv_psi = 1.0 / np.where(hi, 1.0, psi)  # safe denominator
            b2 = np.maximum(
                2.0 * inv_psi - 1.0 + np.sqrt(2.0 * inv_psi) * np.sqrt(2.0 * inv_psi - 1.0), 0.0
            )
            b = np.sqrt(b2)
            a = m / (1.0 + b2)
            z_v = norm_ppf(u_v)  # standard normal quantile
            V_lo = a * (b + z_v) ** 2

            # --- High-psi branch: exponential distribution ---
            p_exp = (psi - 1.0) / (psi + 1.0)
            beta = (1.0 - p_exp) / np.where(hi, m, 1.0)  # avoid /0 in lo branch
            # Inversion of mixed distribution
            # u ~ [0, p] -> V = 0; u ~ (p, 1] -> V = ln(...) / beta
            v_hi_safe = np.where(
                u_v > p_exp,
                np.log((1.0 - p_exp) / np.maximum(1.0 - u_v, 1e-300))
                / np.where(beta > 0, beta, 1.0),
                0.0,
            )
            V_hi = np.where(u_v > p_exp, v_hi_safe, 0.0)

            V_next = np.where(hi, V_hi, V_lo)

            # Log-asset update using V_t and V_next (trapezoidal integral approximation)
            S_t = S_t * np.exp(
                (r - q) * dt + K0 + K1 * V_t + K2 * V_next + np.sqrt(K3 * V_t + K4 * V_next) * z1
            )

            V_t = V_next

    if option_type == "call":
        payoffs = np.maximum(S_t - K, 0.0)
    else:
        payoffs = np.maximum(K - S_t, 0.0)

    discount = np.exp(-r * T)
    price = discount * float(np.mean(payoffs))
    stderr = discount * float(np.std(payoffs) / np.sqrt(n_paths))

    if return_stderr:
        return price, stderr
    return price


# ------------------------------------------------------------------ #
# Greeks (finite differences on the analytic price)
# ------------------------------------------------------------------ #


def heston_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    q: float = 0.0,
    option_type: str = "call",
    *,
    dS: float = 0.5,
    dT: float = 1.0 / 365.0,
    dv0: float = 0.001,
) -> dict[str, float]:
    """Compute Heston Greeks via finite differences on the semi-analytic price.

    Args:
        S: Spot price.
        K, T, r, q, option_type: Option parameters.
        params: HestonParams instance.
        dS: Bump size for Delta/Gamma (default 0.5).
        dT: Bump size for Theta (default 1/365 year).
        dv0: Bump size for Vega (relative to v0, default 0.001).

    Returns:
        Dictionary with keys: 'delta', 'gamma', 'theta', 'vega'.
        Vega is reported per unit change in v0 (not in sigma).

    Example:
        >>> p = HestonParams()
        >>> g = heston_greeks(100.0, 100.0, 1.0, 0.05, p)
        >>> 0.0 < g['delta'] < 1.0
        True
    """
    price_0 = heston_price(S, K, T, r, params, q=q, option_type=option_type)
    price_up = heston_price(S + dS, K, T, r, params, q=q, option_type=option_type)
    price_dn = heston_price(S - dS, K, T, r, params, q=q, option_type=option_type)

    delta = (price_up - price_dn) / (2.0 * dS)
    gamma = (price_up - 2.0 * price_0 + price_dn) / (dS**2)

    price_T = heston_price(S, K, T - dT, r, params, q=q, option_type=option_type)
    theta = (price_T - price_0) / dT  # per year, negative = time decay

    p_vega = HestonParams(
        kappa=params.kappa,
        theta=params.theta,
        xi=params.xi,
        rho=params.rho,
        v0=params.v0 + dv0,
    )
    price_v = heston_price(S, K, T, r, p_vega, q=q, option_type=option_type)
    vega = (price_v - price_0) / dv0  # per unit v0 bump

    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


# ------------------------------------------------------------------ #
# Calibration
# ------------------------------------------------------------------ #


@dataclass
class HestonCalibrator:
    """Calibrate Heston model parameters to a market implied-volatility surface.

    Usage::

        from tensorquantlib.finance.heston import HestonCalibrator, HestonParams
        import numpy as np

        cal = HestonCalibrator(S=100.0, r=0.05)
        K_grid = np.array([90., 95., 100., 105., 110.])
        T_grid = np.array([0.5, 1.0, 1.5])
        # Build synthetic market IV (20% flat)
        iv_mkt = np.full((len(K_grid), len(T_grid)), 0.20)
        cal.fit(iv_mkt, K_grid, T_grid)
        print(cal.params_)

    Attributes:
        S: Spot price (fixed during calibration).
        r: Risk-free rate.
        q: Dividend yield.
        params_: Calibrated HestonParams (set after fit()).
        rmse_: Root-mean-square error in IV after fit.
    """

    S: float
    r: float
    q: float = 0.0
    option_type: str = "call"
    params_: HestonParams = field(default_factory=HestonParams)
    rmse_: float = field(default=float("nan"), init=False, repr=False)

    def fit(
        self,
        iv_market: np.ndarray,
        K_grid: np.ndarray,
        T_grid: np.ndarray,
        *,
        n_restarts: int = 3,
        maxiter: int = 500,
        verbose: bool = False,
    ) -> HestonCalibrator:
        """Fit Heston parameters to market IV surface.

        Args:
            iv_market: 2-D array of market implied vols, shape (len(K_grid), len(T_grid)).
            K_grid: 1-D array of strikes.
            T_grid: 1-D array of expiries.
            n_restarts: Number of random re-starts for the optimiser.
            maxiter: Maximum optimiser iterations per restart.
            verbose: Print progress.

        Returns:
            self
        """
        from tensorquantlib.finance.implied_vol import implied_vol  # avoid circular import

        def _objective(x: np.ndarray) -> float:
            kappa, theta, xi, rho, v0 = x
            params = HestonParams(kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0)
            sq_err = 0.0
            for i, K in enumerate(K_grid):
                for j, T in enumerate(T_grid):
                    try:
                        model_price = heston_price(
                            self.S,
                            float(K),
                            float(T),
                            self.r,
                            params,
                            q=self.q,
                            option_type=self.option_type,
                        )
                        # Convert model price to IV
                        model_iv = implied_vol(
                            model_price,
                            self.S,
                            float(K),
                            float(T),
                            self.r,
                            q=self.q,
                            option_type=self.option_type,
                        )
                        sq_err += (model_iv - iv_market[i, j]) ** 2
                    except Exception:
                        sq_err += 1.0
            return sq_err

        bounds = [
            (0.1, 10.0),  # kappa
            (0.001, 0.5),  # theta
            (0.1, 2.0),  # xi
            (-0.99, 0.99),  # rho
            (0.001, 0.5),  # v0
        ]

        best_result = None
        rng = np.random.default_rng(42)

        for restart in range(n_restarts):
            if restart == 0:
                x0 = np.array([2.0, 0.04, 0.3, -0.5, 0.04])
            else:
                # Random start within bounds
                x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])

            result = minimize(
                _objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8},
            )

            if best_result is None or result.fun < best_result.fun:
                best_result = result
                if verbose:
                    print(
                        f"Restart {restart + 1}/{n_restarts}: RMSE={np.sqrt(result.fun / (len(K_grid) * len(T_grid))):.6f}"
                    )

        if best_result is not None:
            self.params_ = HestonParams.from_array(best_result.x)
            n_obs = len(K_grid) * len(T_grid)
            self.rmse_ = float(np.sqrt(best_result.fun / n_obs))

        return self

    def implied_vol_surface(
        self,
        K_grid: np.ndarray,
        T_grid: np.ndarray,
    ) -> np.ndarray:
        """Compute the model IV surface for the calibrated parameters.

        Args:
            K_grid: 1-D array of strikes.
            T_grid: 1-D array of expiries.

        Returns:
            2-D array of model-implied volatilities, shape (len(K_grid), len(T_grid)).
        """
        from tensorquantlib.finance.implied_vol import implied_vol

        surface = np.full((len(K_grid), len(T_grid)), np.nan)
        for i, K in enumerate(K_grid):
            for j, T in enumerate(T_grid):
                try:
                    price = heston_price(
                        self.S,
                        float(K),
                        float(T),
                        self.r,
                        self.params_,
                        q=self.q,
                        option_type=self.option_type,
                    )
                    surface[i, j] = implied_vol(
                        price,
                        self.S,
                        float(K),
                        float(T),
                        self.r,
                        q=self.q,
                        option_type=self.option_type,
                    )
                except Exception:
                    surface[i, j] = np.nan
        return surface
