"""
Risk assessment utilities for Optionix platform.
Provides VaR, Expected Shortfall, and margin requirement calculations.
"""

from typing import Any

import numpy as np


class RiskCalculator:

    @staticmethod
    def calculate_var(returns: Any, confidence: Any = 0.95) -> Any:
        returns = np.asarray(returns, dtype=float)
        if len(returns) == 0:
            return 0.0
        return float(np.percentile(returns, 100 * (1 - confidence)))

    @staticmethod
    def expected_shortfall(returns: Any, confidence: Any = 0.95) -> Any:
        returns = np.asarray(returns, dtype=float)
        if len(returns) == 0:
            return 0.0
        var = RiskCalculator.calculate_var(returns, confidence)
        tail = returns[returns <= var]
        if len(tail) == 0:
            return float(var)
        return float(tail.mean())

    @staticmethod
    def margin_requirement(portfolio_value: Any, volatility: Any) -> Any:
        return portfolio_value * (0.1 + 0.9 * (volatility / 0.3))
