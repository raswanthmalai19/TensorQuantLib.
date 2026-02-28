"""Tests for market data module (fully mocked — no real API calls)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

# We patch yfinance at the module level so the lazy import picks up the mock.
MOCK_YF = "tensorquantlib.data.market.yfinance"


def _make_hist_df(close, dates=None, n=None):
    """Helper to build a minimal history DataFrame."""
    if n is None:
        n = len(close)
    if dates is None:
        dates = pd.bdate_range(end="2024-01-15", periods=n)
    return pd.DataFrame({
        "Open": close,
        "High": close,
        "Low": close,
        "Close": close,
        "Volume": [1_000_000] * n,
    }, index=dates)


@pytest.fixture
def mock_yf():
    """Fixture that patches yfinance and returns the mock module."""
    with patch.dict("sys.modules", {"yfinance": MagicMock()}) as _:
        import importlib
        import tensorquantlib.data.market as market_mod
        importlib.reload(market_mod)
        import sys
        yf_mock = sys.modules["yfinance"]
        yield yf_mock, market_mod


class TestGetStockPrice:
    def test_returns_last_close(self, mock_yf):
        yf_mock, market = mock_yf
        ticker_mock = MagicMock()
        ticker_mock.history.return_value = _make_hist_df([150.0, 152.0])
        yf_mock.Ticker.return_value = ticker_mock

        price = market.get_stock_price("AAPL")
        assert price == 152.0
        yf_mock.Ticker.assert_called_once_with("AAPL")

    def test_empty_raises(self, mock_yf):
        yf_mock, market = mock_yf
        ticker_mock = MagicMock()
        ticker_mock.history.return_value = pd.DataFrame()
        yf_mock.Ticker.return_value = ticker_mock

        with pytest.raises(ValueError, match="No data"):
            market.get_stock_price("INVALID")


class TestGetHistoricalPrices:
    def test_returns_dict_of_arrays(self, mock_yf):
        yf_mock, market = mock_yf
        ticker_mock = MagicMock()
        df = _make_hist_df([100.0, 101.0, 102.0])
        ticker_mock.history.return_value = df
        yf_mock.Ticker.return_value = ticker_mock

        result = market.get_historical_prices("AAPL", "2024-01-01", "2024-01-15")
        assert set(result.keys()) == {"dates", "open", "high", "low", "close", "volume"}
        assert len(result["close"]) == 3
        np.testing.assert_array_equal(result["close"], [100.0, 101.0, 102.0])


class TestOptionsChain:
    def test_returns_calls_and_puts(self, mock_yf):
        yf_mock, market = mock_yf
        ticker_mock = MagicMock()
        ticker_mock.options = ("2024-02-16",)
        calls_df = pd.DataFrame({
            "strike": [150.0, 155.0],
            "lastPrice": [5.0, 3.0],
            "bid": [4.9, 2.9],
            "ask": [5.1, 3.1],
            "impliedVolatility": [0.2, 0.22],
            "volume": [100, 50],
            "openInterest": [1000, 500],
        })
        puts_df = pd.DataFrame({
            "strike": [145.0, 150.0],
            "lastPrice": [2.0, 4.0],
            "bid": [1.9, 3.9],
            "ask": [2.1, 4.1],
            "impliedVolatility": [0.21, 0.23],
            "volume": [80, 60],
            "openInterest": [800, 600],
        })
        chain_mock = MagicMock()
        chain_mock.calls = calls_df
        chain_mock.puts = puts_df
        ticker_mock.option_chain.return_value = chain_mock
        yf_mock.Ticker.return_value = ticker_mock

        result = market.get_options_chain("AAPL")
        assert "calls" in result and "puts" in result
        np.testing.assert_array_equal(result["calls"]["strike"], [150.0, 155.0])
        np.testing.assert_array_equal(result["puts"]["strike"], [145.0, 150.0])


class TestHistoricalVolatility:
    def test_returns_annualised_vol(self, mock_yf):
        yf_mock, market = mock_yf
        ticker_mock = MagicMock()
        # Generate 50 prices with known daily returns
        np.random.seed(42)
        prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 50)))
        ticker_mock.history.return_value = _make_hist_df(prices, n=50)
        yf_mock.Ticker.return_value = ticker_mock

        vol = market.historical_volatility("AAPL", window=50, annualize=True)
        assert vol > 0
        # Should be roughly 0.01 * sqrt(252) ≈ 0.159
        assert 0.05 < vol < 0.5

    def test_non_annualised(self, mock_yf):
        yf_mock, market = mock_yf
        ticker_mock = MagicMock()
        np.random.seed(42)
        prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 50)))
        ticker_mock.history.return_value = _make_hist_df(prices, n=50)
        yf_mock.Ticker.return_value = ticker_mock

        vol = market.historical_volatility("AAPL", window=50, annualize=False)
        # Daily vol should be much smaller than annualised
        assert vol < 0.05


class TestGetRiskFreeRate:
    def test_returns_decimal_rate(self, mock_yf):
        yf_mock, market = mock_yf
        ticker_mock = MagicMock()
        # ^IRX is quoted as percentage
        ticker_mock.history.return_value = _make_hist_df([5.25, 5.30])
        yf_mock.Ticker.return_value = ticker_mock

        rate = market.get_risk_free_rate()
        assert abs(rate - 0.053) < 0.001
