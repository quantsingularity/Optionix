"""
Pricing engine service for Optionix platform.
Wraps the quantitative library for option pricing and Greeks calculation.
"""

import numpy as np
from scipy.stats import norm


class PricingEngine:
    """Option pricing engine using Black-Scholes model and Monte Carlo simulation"""

    def calculate_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> dict:
        """
        Calculate option Greeks using Black-Scholes.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry in years
            r: Risk-free rate
            sigma: Volatility (annualized)

        Returns:
            Dictionary with delta, gamma, vega
        """
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0}
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        delta = float(norm.cdf(d1))
        gamma = float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))
        vega = float(S * norm.pdf(d1) * np.sqrt(T))
        return {"delta": delta, "gamma": gamma, "vega": vega}

    def monte_carlo_pricing(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        iterations: int = 100_000,
        seed: int = None,
    ) -> float:
        """
        Price a European call option using Monte Carlo simulation.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry in years
            r: Risk-free rate
            sigma: Volatility (annualized)
            iterations: Number of simulation paths
            seed: Random seed for reproducibility

        Returns:
            Option price
        """
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return max(S - K, 0.0)
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(iterations)
        ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
        payoffs = np.maximum(ST - K, 0)
        return float(np.exp(-r * T) * np.mean(payoffs))
