"""Market data wrappers using yfinance.

All functions require the ``yfinance`` package, which is an optional
dependency installable via ``pip install tensorquantlib[data]``.
"""

from __future__ import annotations

import numpy as np


def _import_yfinance():
    """Lazy import of yfinance with a helpful error message."""
    try:
        import yfinance as yf

        return yf
    except ImportError:
        raise ImportError(
            "yfinance is required for market data functions. "
            "Install it with: pip install tensorquantlib[data]"
        ) from None


def get_stock_price(ticker: str) -> float:
    """Get the latest closing price for a ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. ``'AAPL'``).

    Returns
    -------
    float
        Most recent closing price.
    """
    yf = _import_yfinance()
    t = yf.Ticker(ticker)
    hist = t.history(period="1d")
    if hist.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")
    return float(hist["Close"].iloc[-1])


def get_historical_prices(ticker: str, start: str, end: str) -> dict:
    """Fetch historical OHLCV data.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    start : str
        Start date ``'YYYY-MM-DD'``.
    end : str
        End date ``'YYYY-MM-DD'``.

    Returns
    -------
    dict
        Keys: ``'dates'``, ``'open'``, ``'high'``, ``'low'``, ``'close'``,
        ``'volume'`` — each a numpy array.
    """
    yf = _import_yfinance()
    t = yf.Ticker(ticker)
    hist = t.history(start=start, end=end)
    if hist.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")
    return {
        "dates": hist.index.to_numpy(),
        "open": hist["Open"].to_numpy(),
        "high": hist["High"].to_numpy(),
        "low": hist["Low"].to_numpy(),
        "close": hist["Close"].to_numpy(),
        "volume": hist["Volume"].to_numpy(),
    }


def get_options_chain(ticker: str, expiry: str | None = None) -> dict:
    """Fetch options chain data.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    expiry : str, optional
        Expiry date ``'YYYY-MM-DD'``. If *None*, uses the nearest expiry.

    Returns
    -------
    dict
        Keys: ``'calls'`` and ``'puts'``, each a dict with numpy arrays for
        ``'strike'``, ``'lastPrice'``, ``'bid'``, ``'ask'``,
        ``'impliedVolatility'``, ``'volume'``, ``'openInterest'``.
    """
    yf = _import_yfinance()
    t = yf.Ticker(ticker)
    if expiry is None:
        expiries = t.options
        if not expiries:
            raise ValueError(f"No options available for '{ticker}'")
        expiry = expiries[0]
    chain = t.option_chain(expiry)

    def _extract(df):
        cols = ["strike", "lastPrice", "bid", "ask", "impliedVolatility", "volume", "openInterest"]
        return {c: df[c].to_numpy() for c in cols if c in df.columns}

    return {"calls": _extract(chain.calls), "puts": _extract(chain.puts)}


def historical_volatility(ticker: str, window: int = 252, annualize: bool = True) -> float:
    """Compute realised historical volatility from daily close prices.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    window : int
        Number of trading days to look back.
    annualize : bool
        If True, return annualised volatility.

    Returns
    -------
    float
        Historical volatility.
    """
    yf = _import_yfinance()
    t = yf.Ticker(ticker)
    hist = t.history(period=f"{window}d")
    if len(hist) < 2:
        raise ValueError(f"Insufficient data for ticker '{ticker}'")
    prices = hist["Close"].to_numpy()
    log_returns = np.log(prices[1:] / prices[:-1])
    vol = float(np.std(log_returns, ddof=1))
    if annualize:
        vol *= np.sqrt(252)
    return vol


def get_risk_free_rate() -> float:
    """Get approximate risk-free rate from 13-week US Treasury yield.

    Uses the ``^IRX`` ticker (CBOE 13-week T-Bill) as a proxy.

    Returns
    -------
    float
        Annualised risk-free rate (e.g. 0.05 for 5%).
    """
    yf = _import_yfinance()
    t = yf.Ticker("^IRX")
    hist = t.history(period="5d")
    if hist.empty:
        raise ValueError("Could not fetch Treasury rate data")
    # ^IRX is quoted as a percentage (e.g. 5.25 means 5.25%)
    return float(hist["Close"].iloc[-1]) / 100.0
