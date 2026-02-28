"""Jump-diffusion models for option pricing.

- Merton (1976) jump-diffusion: analytic + MC
- Kou (2002) double-exponential jump-diffusion: MC
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.special import factorial


# ---------------------------------------------------------------------------
# Merton jump-diffusion (1976)
# ---------------------------------------------------------------------------

def merton_jump_price(S: float, K: float, T: float, r: float, sigma: float,
                      lam: float, mu_j: float, sigma_j: float,
                      option_type: str = "call", n_terms: int = 50) -> float:
    """Merton (1976) jump-diffusion price via series expansion.

    The stock follows dS/S = (r - lambda*k)*dt + sigma*dW + J*dN
    where N is Poisson(lambda*T), J ~ LogNormal(mu_j, sigma_j^2),
    and k = E[e^J - 1] = exp(mu_j + sigma_j^2/2) - 1.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to expiry (years).
    r : float
        Risk-free rate.
    sigma : float
        Diffusion volatility.
    lam : float
        Jump intensity (expected number of jumps per year).
    mu_j : float
        Mean of log-jump size.
    sigma_j : float
        Std dev of log-jump size.
    option_type : str
        ``'call'`` or ``'put'``.
    n_terms : int
        Number of terms in the series expansion.

    Returns
    -------
    float
        Option price.
    """
    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    lam_prime = lam * (1 + k)
    price = 0.0

    for n in range(n_terms):
        sigma_n = np.sqrt(sigma**2 + n * sigma_j**2 / T)
        r_n = r - lam * k + n * np.log(1 + k) / T
        # Standard BS price with (r_n, sigma_n)
        d1 = (np.log(S / K) + (r_n + 0.5 * sigma_n**2) * T) / (sigma_n * np.sqrt(T))
        d2 = d1 - sigma_n * np.sqrt(T)

        if option_type == "call":
            bs_n = S * norm.cdf(d1) - K * np.exp(-r_n * T) * norm.cdf(d2)
        else:
            bs_n = K * np.exp(-r_n * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        weight = np.exp(-lam_prime * T) * (lam_prime * T)**n / factorial(n, exact=True)
        price += weight * bs_n

    return float(price)


def merton_jump_price_mc(S: float, K: float, T: float, r: float, sigma: float,
                         lam: float, mu_j: float, sigma_j: float,
                         option_type: str = "call", n_paths: int = 100_000,
                         n_steps: int = 252, seed: int | None = None) -> float:
    """Merton jump-diffusion price via Monte Carlo.

    Parameters
    ----------
    S, K, T, r, sigma, lam, mu_j, sigma_j, option_type
        Same as :func:`merton_jump_price`.
    n_paths : int
        Number of MC paths.
    n_steps : int
        Time steps per path.
    seed : int, optional
        Random seed.

    Returns
    -------
    float
        MC estimate of the option price.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0

    log_S = np.full(n_paths, np.log(S))
    drift = (r - 0.5 * sigma**2 - lam * k) * dt

    for _ in range(n_steps):
        z = rng.standard_normal(n_paths)
        # Poisson jumps in [t, t+dt]
        n_jumps = rng.poisson(lam * dt, n_paths)
        # Total jump size: sum of n_jumps LogNormal jumps
        jump = np.zeros(n_paths)
        has_jumps = n_jumps > 0
        if np.any(has_jumps):
            for i in np.where(has_jumps)[0]:
                jump[i] = np.sum(rng.normal(mu_j, sigma_j, n_jumps[i]))
        log_S += drift + sigma * np.sqrt(dt) * z + jump

    S_T = np.exp(log_S)
    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    else:
        payoff = np.maximum(K - S_T, 0.0)

    return float(np.exp(-r * T) * np.mean(payoff))


# ---------------------------------------------------------------------------
# Kou double-exponential jump-diffusion (2002)
# ---------------------------------------------------------------------------

def kou_jump_price_mc(S: float, K: float, T: float, r: float, sigma: float,
                      lam: float, p: float, eta1: float, eta2: float,
                      option_type: str = "call", n_paths: int = 100_000,
                      n_steps: int = 252, seed: int | None = None) -> float:
    """Kou (2002) double-exponential jump-diffusion via Monte Carlo.

    Jump sizes J follow a double-exponential distribution:
        f(x) = p * eta1 * exp(-eta1*x) * 1_{x>=0}
             + (1-p) * eta2 * exp(eta2*x) * 1_{x<0}

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to expiry.
    r : float
        Risk-free rate.
    sigma : float
        Diffusion volatility.
    lam : float
        Jump intensity.
    p : float
        Probability of upward jump (0 < p < 1).
    eta1 : float
        Rate parameter for upward jumps (eta1 > 1 for finite expectation).
    eta2 : float
        Rate parameter for downward jumps (eta2 > 0).
    option_type : str
        ``'call'`` or ``'put'``.
    n_paths : int
        Number of MC paths.
    n_steps : int
        Time steps.
    seed : int, optional
        Random seed.

    Returns
    -------
    float
        MC estimate of option price.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    # E[e^J] = p*eta1/(eta1-1) + (1-p)*eta2/(eta2+1)
    k = p * eta1 / (eta1 - 1) + (1 - p) * eta2 / (eta2 + 1) - 1.0

    log_S = np.full(n_paths, np.log(S))
    drift = (r - 0.5 * sigma**2 - lam * k) * dt

    for _ in range(n_steps):
        z = rng.standard_normal(n_paths)
        n_jumps = rng.poisson(lam * dt, n_paths)
        jump = np.zeros(n_paths)
        has_jumps = n_jumps > 0
        if np.any(has_jumps):
            for i in np.where(has_jumps)[0]:
                for _ in range(n_jumps[i]):
                    u = rng.random()
                    if u < p:
                        jump[i] += rng.exponential(1.0 / eta1)
                    else:
                        jump[i] -= rng.exponential(1.0 / eta2)
        log_S += drift + sigma * np.sqrt(dt) * z + jump

    S_T = np.exp(log_S)
    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    else:
        payoff = np.maximum(K - S_T, 0.0)

    return float(np.exp(-r * T) * np.mean(payoff))
