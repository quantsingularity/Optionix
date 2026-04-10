"""
PricingEngine and RiskCalculator unit tests.
"""

import numpy as np
import pytest
from app.services.pricing_engine import PricingEngine
from app.services.risk_assessment import RiskCalculator


@pytest.fixture
def engine():
    return PricingEngine()


@pytest.fixture
def risk():
    return RiskCalculator()


# ---------------------------------------------------------------------------
# PricingEngine – Greeks
# ---------------------------------------------------------------------------


class TestPricingEngineGreeks:
    ATM_PARAMS = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20)

    def test_returns_required_keys(self, engine):
        g = engine.calculate_greeks(**self.ATM_PARAMS)
        assert "delta" in g
        assert "gamma" in g
        assert "vega" in g

    def test_delta_between_0_and_1(self, engine):
        g = engine.calculate_greeks(**self.ATM_PARAMS)
        assert 0 < g["delta"] < 1

    def test_gamma_positive(self, engine):
        g = engine.calculate_greeks(**self.ATM_PARAMS)
        assert g["gamma"] > 0

    def test_vega_positive(self, engine):
        g = engine.calculate_greeks(**self.ATM_PARAMS)
        assert g["vega"] > 0

    def test_zero_time_returns_safe_defaults(self, engine):
        g = engine.calculate_greeks(S=100.0, K=100.0, T=0.0, r=0.05, sigma=0.20)
        assert g["delta"] == 0.0
        assert g["gamma"] == 0.0

    def test_deep_itm_delta_approaches_1(self, engine):
        g = engine.calculate_greeks(S=200.0, K=100.0, T=1.0, r=0.05, sigma=0.10)
        assert g["delta"] > 0.95

    def test_deep_otm_delta_approaches_0(self, engine):
        g = engine.calculate_greeks(S=50.0, K=200.0, T=1.0, r=0.05, sigma=0.10)
        assert g["delta"] < 0.05


# ---------------------------------------------------------------------------
# PricingEngine – Monte Carlo
# ---------------------------------------------------------------------------


class TestMonteCarloPricing:
    def test_price_is_positive(self, engine):
        price = engine.monte_carlo_pricing(100.0, 100.0, 1.0, 0.05, 0.20, seed=42)
        assert price > 0

    def test_intrinsic_value_floor(self, engine):
        """Deep ITM option price must exceed intrinsic value approximately"""
        price = engine.monte_carlo_pricing(150.0, 100.0, 0.001, 0.0, 0.01, seed=42)
        assert price >= 49.0  # roughly S - K

    def test_zero_time_returns_intrinsic(self, engine):
        price = engine.monte_carlo_pricing(110.0, 100.0, 0.0, 0.05, 0.20, seed=0)
        assert price == 10.0

    def test_reproducible_with_seed(self, engine):
        p1 = engine.monte_carlo_pricing(100.0, 100.0, 1.0, 0.05, 0.20, seed=123)
        p2 = engine.monte_carlo_pricing(100.0, 100.0, 1.0, 0.05, 0.20, seed=123)
        assert abs(p1 - p2) < 1e-10

    def test_price_increases_with_volatility(self, engine):
        low = engine.monte_carlo_pricing(100.0, 100.0, 1.0, 0.05, 0.10, seed=42)
        high = engine.monte_carlo_pricing(100.0, 100.0, 1.0, 0.05, 0.50, seed=42)
        assert high > low


# ---------------------------------------------------------------------------
# RiskCalculator
# ---------------------------------------------------------------------------


class TestRiskCalculator:
    RETURNS = np.array(
        [-0.05, -0.03, -0.01, 0.00, 0.01, 0.02, 0.03, -0.10, -0.08, 0.04]
    )

    def test_var_negative_for_loss(self):
        var = RiskCalculator.calculate_var(self.RETURNS, 0.95)
        assert var <= 0

    def test_var_worse_at_higher_confidence(self):
        var95 = RiskCalculator.calculate_var(self.RETURNS, 0.95)
        var99 = RiskCalculator.calculate_var(self.RETURNS, 0.99)
        assert var99 <= var95

    def test_var_empty_returns_zero(self):
        var = RiskCalculator.calculate_var(np.array([]), 0.95)
        assert var == 0.0

    def test_expected_shortfall_worse_than_var(self):
        var = RiskCalculator.calculate_var(self.RETURNS, 0.95)
        es = RiskCalculator.expected_shortfall(self.RETURNS, 0.95)
        assert es <= var

    def test_expected_shortfall_empty_returns_zero(self):
        es = RiskCalculator.expected_shortfall(np.array([]), 0.95)
        assert es == 0.0

    def test_margin_requirement_increases_with_volatility(self):
        low = RiskCalculator.margin_requirement(100_000, 0.10)
        high = RiskCalculator.margin_requirement(100_000, 0.50)
        assert high > low

    def test_margin_requirement_positive(self):
        margin = RiskCalculator.margin_requirement(100_000, 0.20)
        assert margin > 0
