"""
Additional tests for option pricing, Greeks, and model service.
Covers Black-Scholes pricing, Greeks calculations, and ModelService fallback.
"""

from decimal import Decimal

import pytest
from app.services.financial_service import FinancialCalculationService
from app.services.model_service import ModelService


@pytest.fixture
def svc():
    return FinancialCalculationService()


@pytest.fixture
def model_svc():
    return ModelService()


# ---------------------------------------------------------------------------
# Black-Scholes pricing
# ---------------------------------------------------------------------------


class TestBlackScholesPrice:
    def test_call_price_positive(self, svc):
        price = svc.calculate_black_scholes_price(
            spot_price=Decimal("100"),
            strike_price=Decimal("100"),
            time_to_expiry=Decimal("1.0"),
            volatility=Decimal("0.20"),
            is_call=True,
        )
        assert price > Decimal("0")

    def test_put_price_positive(self, svc):
        price = svc.calculate_black_scholes_price(
            spot_price=Decimal("100"),
            strike_price=Decimal("100"),
            time_to_expiry=Decimal("1.0"),
            volatility=Decimal("0.20"),
            is_call=False,
        )
        assert price > Decimal("0")

    def test_put_call_parity(self, svc):
        """C - P = S - K*e^(-rT) approximately"""
        S = Decimal("100")
        K = Decimal("100")
        T = Decimal("1.0")
        sigma = Decimal("0.20")
        r = Decimal("0.05")
        call = svc.calculate_black_scholes_price(S, K, T, sigma, r, is_call=True)
        put = svc.calculate_black_scholes_price(S, K, T, sigma, r, is_call=False)
        import math

        parity = S - K * Decimal(str(math.exp(-float(r) * float(T))))
        diff = abs((call - put) - parity)
        assert diff < Decimal("0.10")  # allow small rounding tolerance

    def test_itm_call_higher_than_otm_call(self, svc):
        itm = svc.calculate_black_scholes_price(
            Decimal("110"), Decimal("100"), Decimal("1.0"), Decimal("0.20")
        )
        otm = svc.calculate_black_scholes_price(
            Decimal("90"), Decimal("100"), Decimal("1.0"), Decimal("0.20")
        )
        assert itm > otm

    def test_price_increases_with_volatility(self, svc):
        low_vol = svc.calculate_black_scholes_price(
            Decimal("100"), Decimal("100"), Decimal("1.0"), Decimal("0.10")
        )
        high_vol = svc.calculate_black_scholes_price(
            Decimal("100"), Decimal("100"), Decimal("1.0"), Decimal("0.40")
        )
        assert high_vol > low_vol

    def test_invalid_zero_time_raises(self, svc):
        with pytest.raises(ValueError):
            svc.calculate_black_scholes_price(
                Decimal("100"), Decimal("100"), Decimal("0"), Decimal("0.20")
            )

    def test_result_is_decimal(self, svc):
        price = svc.calculate_black_scholes_price(
            Decimal("100"), Decimal("100"), Decimal("1.0"), Decimal("0.20")
        )
        assert isinstance(price, Decimal)


# ---------------------------------------------------------------------------
# Option Greeks
# ---------------------------------------------------------------------------


class TestOptionGreeks:
    PARAMS = dict(
        spot_price=Decimal("100"),
        strike_price=Decimal("100"),
        time_to_expiry=Decimal("1.0"),
        volatility=Decimal("0.20"),
    )

    def test_greeks_keys_present(self, svc):
        g = svc.calculate_option_greeks(**self.PARAMS)
        for key in ("delta", "gamma", "theta", "vega", "rho"):
            assert key in g

    def test_call_delta_between_0_and_1(self, svc):
        g = svc.calculate_option_greeks(**self.PARAMS, is_call=True)
        assert Decimal("0") < g["delta"] < Decimal("1")

    def test_put_delta_between_neg1_and_0(self, svc):
        g = svc.calculate_option_greeks(**self.PARAMS, is_call=False)
        assert Decimal("-1") < g["delta"] < Decimal("0")

    def test_gamma_positive(self, svc):
        g = svc.calculate_option_greeks(**self.PARAMS)
        assert g["gamma"] > Decimal("0")

    def test_vega_positive(self, svc):
        g = svc.calculate_option_greeks(**self.PARAMS)
        assert g["vega"] > Decimal("0")

    def test_theta_negative_for_long_call(self, svc):
        g = svc.calculate_option_greeks(**self.PARAMS, is_call=True)
        assert g["theta"] < Decimal("0")

    def test_invalid_negative_spot_raises(self, svc):
        with pytest.raises(ValueError):
            svc.calculate_option_greeks(
                spot_price=Decimal("-1"),
                strike_price=Decimal("100"),
                time_to_expiry=Decimal("1.0"),
                volatility=Decimal("0.20"),
            )


# ---------------------------------------------------------------------------
# ModelService – fallback path (no model file loaded)
# ---------------------------------------------------------------------------


class TestModelServiceFallback:
    def test_fallback_returns_volatility(self, model_svc, db):
        result = model_svc.get_volatility_prediction(
            {
                "symbol": "BTC",
                "open": 50000.0,
                "high": 52000.0,
                "low": 49000.0,
                "volume": 1000000.0,
            },
            db,
        )
        assert "volatility" in result
        assert result["volatility"] >= 0

    def test_fallback_volatility_keys(self, model_svc, db):
        result = model_svc.get_volatility_prediction(
            {"open": 100.0, "high": 110.0, "low": 95.0, "volume": 50000.0},
            db,
        )
        assert "model_version" in result
        assert "prediction_timestamp" in result

    def test_model_availability_check(self, model_svc):
        # is_model_available returns a bool regardless of whether model loaded
        result = model_svc.is_model_available()
        assert isinstance(result, bool)

    def test_invalid_negative_open_raises(self, model_svc, db):
        with pytest.raises(ValueError):
            model_svc.get_volatility_prediction(
                {"open": -1.0, "high": 110.0, "low": 90.0, "volume": 1000.0},
                db,
            )
