"""Financial calculation service unit tests."""

from decimal import Decimal

import pytest
from app.services.financial_service import FinancialCalculationService


@pytest.fixture
def svc():
    return FinancialCalculationService()


class TestLiquidationPrice:
    def test_long_position_liquidation_below_entry(self, svc):
        result = svc.calculate_liquidation_price(
            Decimal("50000"), Decimal("1.0"), True, Decimal("5000")
        )
        assert result < Decimal("50000")

    def test_short_position_liquidation_above_entry(self, svc):
        result = svc.calculate_liquidation_price(
            Decimal("50000"), Decimal("1.0"), False, Decimal("5000")
        )
        assert result > Decimal("50000")

    def test_liquidation_price_positive(self, svc):
        result = svc.calculate_liquidation_price(
            Decimal("100"), Decimal("10.0"), True, Decimal("5")
        )
        assert result > 0

    def test_liquidation_price_decimal_precision(self, svc):
        result = svc.calculate_liquidation_price(
            Decimal("50000"), Decimal("1.0"), True, Decimal("5000")
        )
        # Result should be quantized to 2 decimal places
        assert result == result.quantize(Decimal("0.01"))


class TestMarginRequirement:
    def test_default_leverage(self, svc):
        margin = svc.calculate_margin_requirement(Decimal("100000"))
        assert margin == Decimal("10000.00")

    def test_custom_leverage(self, svc):
        margin = svc.calculate_margin_requirement(
            Decimal("100000"), leverage=Decimal("5")
        )
        assert margin == Decimal("20000.00")

    def test_custom_margin_rate(self, svc):
        margin = svc.calculate_margin_requirement(
            Decimal("100000"), margin_rate=Decimal("0.1")
        )
        assert margin == Decimal("10000.00")

    def test_margin_is_decimal(self, svc):
        margin = svc.calculate_margin_requirement(Decimal("50000"), Decimal("10"))
        assert isinstance(margin, Decimal)


class TestUnrealizedPnL:
    def test_long_profit(self, svc):
        pnl = svc.calculate_unrealized_pnl(
            Decimal("50000"), Decimal("55000"), Decimal("1.0"), True
        )
        assert pnl == Decimal("5000.00")

    def test_long_loss(self, svc):
        pnl = svc.calculate_unrealized_pnl(
            Decimal("50000"), Decimal("45000"), Decimal("1.0"), True
        )
        assert pnl == Decimal("-5000.00")

    def test_short_profit(self, svc):
        pnl = svc.calculate_unrealized_pnl(
            Decimal("50000"), Decimal("45000"), Decimal("1.0"), False
        )
        assert pnl == Decimal("5000.00")

    def test_short_loss(self, svc):
        pnl = svc.calculate_unrealized_pnl(
            Decimal("50000"), Decimal("55000"), Decimal("1.0"), False
        )
        assert pnl == Decimal("-5000.00")

    def test_pnl_at_entry_price_is_zero(self, svc):
        pnl = svc.calculate_unrealized_pnl(
            Decimal("50000"), Decimal("50000"), Decimal("1.0"), True
        )
        assert pnl == Decimal("0.00")


class TestTradingFees:
    def test_taker_fee(self, svc):
        fee = svc.calculate_trading_fees(Decimal("10000"), is_maker=False)
        assert fee == Decimal("10.00")

    def test_maker_fee_is_half_taker(self, svc):
        taker = svc.calculate_trading_fees(Decimal("10000"), is_maker=False)
        maker = svc.calculate_trading_fees(Decimal("10000"), is_maker=True)
        assert maker == taker / 2

    def test_custom_fee_rate(self, svc):
        fee = svc.calculate_trading_fees(
            Decimal("10000"), fee_rate=Decimal("0.002"), is_maker=False
        )
        assert fee == Decimal("20.00")

    def test_zero_trade_value(self, svc):
        fee = svc.calculate_trading_fees(Decimal("0"))
        assert fee == Decimal("0.00")


class TestPositionHealthRatio:
    def test_healthy_position(self, svc):
        ratio = svc.calculate_position_health_ratio(
            Decimal("10000"), Decimal("500"), Decimal("2000")
        )
        assert ratio > Decimal("1.0")

    def test_zero_margin_returns_max(self, svc):
        ratio = svc.calculate_position_health_ratio(
            Decimal("10000"), Decimal("0"), Decimal("0")
        )
        assert ratio == Decimal("999.99")

    def test_underwater_position(self, svc):
        ratio = svc.calculate_position_health_ratio(
            Decimal("1000"), Decimal("-5000"), Decimal("2000")
        )
        assert ratio < Decimal("1.0")


class TestPositionLimitsValidation:
    def test_valid_position(self, svc):
        result = svc.validate_position_limits(
            Decimal("1.0"), Decimal("50000"), Decimal("100000")
        )
        assert "valid" in result
        assert "violations" in result
        assert "required_margin" in result

    def test_small_valid_position(self, svc):
        result = svc.validate_position_limits(
            Decimal("0.01"), Decimal("100"), Decimal("10000")
        )
        assert result["valid"] is True

    def test_excessive_position_invalid(self, svc):
        result = svc.validate_position_limits(
            Decimal("100"), Decimal("999999999"), Decimal("10000")
        )
        assert result["valid"] is False
        assert len(result["violations"]) > 0


class TestVaR:
    def test_var_positive(self, svc):
        var = svc.calculate_var(Decimal("1000000"), Decimal("0.20"), Decimal("0.95"), 1)
        assert var > Decimal("0")

    def test_var_increases_with_horizon(self, svc):
        var_1day = svc.calculate_var(Decimal("1000000"), Decimal("0.20"))
        var_5day = svc.calculate_var(
            Decimal("1000000"), Decimal("0.20"), time_horizon=5
        )
        assert var_5day > var_1day

    def test_var_increases_with_confidence(self, svc):
        var_95 = svc.calculate_var(Decimal("1000000"), Decimal("0.20"), Decimal("0.95"))
        var_99 = svc.calculate_var(Decimal("1000000"), Decimal("0.20"), Decimal("0.99"))
        assert var_99 > var_95
