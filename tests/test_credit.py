"""Tests for credit risk models (Merton structural + CDS pricing)."""
from __future__ import annotations

import numpy as np
import pytest

from tensorquantlib.finance.credit import (
    merton_default_prob,
    merton_credit_spread,
    survival_probability,
    hazard_rate_from_spread,
    cds_spread,
    cds_price,
)


class TestMertonStructural:
    """Tests for Merton (1974) structural credit model."""

    def test_default_prob_in_range(self):
        pd = merton_default_prob(V=100, D=80, T=1.0, r=0.05, sigma_V=0.2)
        assert 0 < pd < 1

    def test_default_prob_decreases_with_leverage(self):
        """Lower D/V → lower default probability."""
        pd_high = merton_default_prob(V=100, D=90, T=1.0, r=0.05, sigma_V=0.2)
        pd_low = merton_default_prob(V=100, D=50, T=1.0, r=0.05, sigma_V=0.2)
        assert pd_high > pd_low

    def test_default_prob_near_zero_for_safe_firm(self):
        """Very low leverage → ~0 default probability."""
        pd = merton_default_prob(V=1000, D=10, T=1.0, r=0.05, sigma_V=0.1)
        assert pd < 1e-6

    def test_default_prob_increases_with_vol(self):
        pd_low = merton_default_prob(V=100, D=80, T=1.0, r=0.05, sigma_V=0.1)
        pd_high = merton_default_prob(V=100, D=80, T=1.0, r=0.05, sigma_V=0.4)
        assert pd_high > pd_low

    def test_credit_spread_positive(self):
        s = merton_credit_spread(V=100, D=80, T=1.0, r=0.05, sigma_V=0.2)
        assert s > 0

    def test_credit_spread_increases_with_leverage(self):
        s1 = merton_credit_spread(V=100, D=50, T=1.0, r=0.05, sigma_V=0.2)
        s2 = merton_credit_spread(V=100, D=90, T=1.0, r=0.05, sigma_V=0.2)
        assert s2 > s1

    def test_credit_spread_near_zero_for_safe_firm(self):
        s = merton_credit_spread(V=1000, D=10, T=1.0, r=0.05, sigma_V=0.1)
        assert s < 1e-4

    def test_credit_spread_increases_with_vol(self):
        s1 = merton_credit_spread(V=100, D=80, T=1.0, r=0.05, sigma_V=0.1)
        s2 = merton_credit_spread(V=100, D=80, T=1.0, r=0.05, sigma_V=0.4)
        assert s2 > s1


class TestSurvivalHazard:
    """Tests for survival probability and hazard rate utilities."""

    def test_survival_exponential(self):
        """Q(T) = exp(-lambda*T) for constant hazard."""
        q = survival_probability(hazard_rate=0.02, T=5.0)
        expected = np.exp(-0.02 * 5.0)
        assert abs(q - expected) < 1e-12

    def test_survival_at_zero(self):
        assert survival_probability(hazard_rate=0.02, T=0.0) == 1.0

    def test_survival_decreases_with_time(self):
        q1 = survival_probability(0.02, 1.0)
        q5 = survival_probability(0.02, 5.0)
        assert q5 < q1

    def test_hazard_from_spread_roundtrip(self):
        """lambda = spread / (1 - R)."""
        spread, R = 0.01, 0.4
        lam = hazard_rate_from_spread(spread, R)
        expected = 0.01 / 0.6
        assert abs(lam - expected) < 1e-12

    def test_hazard_from_spread_zero_recovery(self):
        lam = hazard_rate_from_spread(0.01, 0.0)
        assert abs(lam - 0.01) < 1e-12


class TestCDS:
    """Tests for CDS pricing."""

    def test_par_spread_positive(self):
        s = cds_spread(hazard_rate=0.02, T=5.0, recovery=0.4, r=0.05)
        assert s > 0

    def test_cds_mtm_zero_at_par(self):
        """At inception (spread = par), MTM = 0."""
        par = cds_spread(hazard_rate=0.02, T=5.0, recovery=0.4, r=0.05)
        mtm = cds_price(hazard_rate=0.02, T=5.0, recovery=0.4, r=0.05,
                        spread=par, notional=1e6)
        assert abs(mtm) < 1.0  # <$1 on 1M notional

    def test_cds_mtm_positive_when_spread_widens(self):
        """Protection buyer profits when credit deteriorates."""
        par = cds_spread(hazard_rate=0.02, T=5.0, recovery=0.4, r=0.05)
        # Now hazard rate doubles → protection more valuable
        mtm = cds_price(hazard_rate=0.04, T=5.0, recovery=0.4, r=0.05,
                        spread=par, notional=1e6)
        assert mtm > 0

    def test_cds_mtm_negative_when_spread_tightens(self):
        """Protection buyer loses when credit improves."""
        par = cds_spread(hazard_rate=0.04, T=5.0, recovery=0.4, r=0.05)
        mtm = cds_price(hazard_rate=0.02, T=5.0, recovery=0.4, r=0.05,
                        spread=par, notional=1e6)
        assert mtm < 0

    def test_spread_roundtrip_via_hazard(self):
        """spread → hazard → cds_spread should approximately recover original."""
        original_spread = 0.01
        R = 0.4
        lam = hazard_rate_from_spread(original_spread, R)
        recovered = cds_spread(lam, T=5.0, recovery=R, r=0.05)
        assert abs(recovered - original_spread) < 0.0005

    def test_higher_hazard_higher_spread(self):
        s1 = cds_spread(hazard_rate=0.01, T=5.0, recovery=0.4, r=0.05)
        s2 = cds_spread(hazard_rate=0.05, T=5.0, recovery=0.4, r=0.05)
        assert s2 > s1

    def test_zero_hazard_zero_spread(self):
        s = cds_spread(hazard_rate=0.0, T=5.0, recovery=0.4, r=0.05)
        assert abs(s) < 1e-12
