"""
Monte Carlo Simulation Module for Optionix Platform
Implements comprehensive Monte Carlo methods for:
- Option pricing (European, Asian, Barrier, Exotic)
- Risk management (VaR, CVaR, stress testing)
- Variance reduction techniques
- Parallel processing for performance
"""

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats.qmc import Sobol

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ProcessType(str, Enum):
    """Stochastic process types"""

    GEOMETRIC_BROWNIAN_MOTION = "gbm"
    MEAN_REVERTING = "mean_reverting"
    JUMP_DIFFUSION = "jump_diffusion"
    HESTON = "heston"
    VASICEK = "vasicek"
    CIR = "cir"


class VarianceReduction(str, Enum):
    """Variance reduction techniques"""

    ANTITHETIC = "antithetic"
    CONTROL_VARIATE = "control_variate"
    QUASI_RANDOM = "quasi_random"


@dataclass
class SimulationParameters:
    """Monte Carlo simulation parameters"""

    initial_price: float
    risk_free_rate: float
    volatility: float
    time_horizon: float
    time_steps: int
    num_simulations: int
    process_type: ProcessType = ProcessType.GEOMETRIC_BROWNIAN_MOTION
    variance_reduction: Optional[VarianceReduction] = None
    random_seed: Optional[int] = None
    parallel: bool = False
    mean_reversion_speed: Optional[float] = None
    long_term_mean: Optional[float] = None
    jump_intensity: Optional[float] = None
    jump_mean: Optional[float] = None
    jump_volatility: Optional[float] = None
    initial_variance: Optional[float] = None
    variance_mean_reversion: Optional[float] = None
    long_term_variance: Optional[float] = None
    vol_of_vol: Optional[float] = None
    correlation: Optional[float] = None


@dataclass
class OptionPayoff:
    """Option payoff specification"""

    option_style: str
    option_type: str
    strike_price: float
    barrier_level: Optional[float] = None
    barrier_type: Optional[str] = None
    averaging_start_step: int = 0
    lookback_start_step: int = 0


@dataclass
class SimulationResult:
    """Monte Carlo simulation result"""

    option_price: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    paths: Optional[np.ndarray] = field(default=None, repr=False)
    payoffs: Optional[np.ndarray] = field(default=None, repr=False)
    greeks: Dict[str, float] = field(default_factory=lambda: {})
    convergence_data: Dict[str, List[float]] = field(default_factory=lambda: {})
    computation_time: float = 0.0
    method_used: str = ""


class MCSimulator:
    """Monte Carlo simulator with advanced features"""

    def __init__(self, params: SimulationParameters) -> None:
        """Initialize Monte Carlo simulator"""
        self.params = params
        self.dt = params.time_horizon / params.time_steps
        self.validate_parameters()
        if params.random_seed is not None:
            np.random.seed(params.random_seed)

    def validate_parameters(self) -> None:
        """Validate simulation parameters and financial constraints"""
        if self.params.initial_price <= 0:
            raise ValueError("Initial price must be positive")
        if self.params.volatility < 0:
            raise ValueError("Volatility must be non-negative")
        if self.params.time_horizon <= 0:
            raise ValueError("Time horizon must be positive")
        if self.params.time_steps <= 0:
            raise ValueError("Time steps must be positive")
        if self.params.num_simulations <= 0:
            raise ValueError("Number of simulations must be positive")
        if self.params.risk_free_rate is None:
            raise ValueError("Risk-free rate must be provided for pricing")
        if self.params.process_type == ProcessType.MEAN_REVERTING and (
            self.params.mean_reversion_speed is None
            or self.params.long_term_mean is None
        ):
            raise ValueError("Mean reversion parameters required")

    def geometric_brownian_motion(
        self, n_simulations: Optional[int] = None
    ) -> np.ndarray:
        """Generate paths using Geometric Brownian Motion (Vectorized)"""
        n_sims = n_simulations or self.params.num_simulations
        if self.params.variance_reduction == VarianceReduction.ANTITHETIC:
            n_base = n_sims // 2
            random_base = np.random.standard_normal((self.params.time_steps, n_base))
            random_normals = np.concatenate([random_base, -random_base], axis=1)
            if n_sims % 2 != 0:
                random_normals = np.concatenate(
                    [
                        random_normals,
                        np.random.standard_normal((self.params.time_steps, 1)),
                    ],
                    axis=1,
                )
        elif self.params.variance_reduction == VarianceReduction.QUASI_RANDOM:
            sobol = Sobol(d=self.params.time_steps, scramble=True)
            uniform_randoms = sobol.random(n=n_sims)
            random_normals = stats.norm.ppf(uniform_randoms).T
        else:
            random_normals = np.random.standard_normal((self.params.time_steps, n_sims))
        drift_term = (
            self.params.risk_free_rate - 0.5 * self.params.volatility**2
        ) * self.dt
        diffusion_term = self.params.volatility * np.sqrt(self.dt) * random_normals
        log_returns = drift_term + diffusion_term
        cumulative_log_returns = np.cumsum(log_returns, axis=0)
        paths = np.zeros((self.params.time_steps + 1, n_sims))
        paths[0, :] = self.params.initial_price
        paths[1:] = self.params.initial_price * np.exp(cumulative_log_returns)
        return paths

    def mean_reverting_process(self, n_simulations: Optional[int] = None) -> np.ndarray:
        """Generate paths using mean-reverting process (Ornstein-Uhlenbeck)"""
        n_sims = n_simulations or self.params.num_simulations
        dt = self.dt
        kappa = self.params.mean_reversion_speed
        theta = self.params.long_term_mean
        sigma = self.params.volatility
        paths = np.zeros((self.params.time_steps + 1, n_sims))
        paths[0] = self.params.initial_price
        random_normals = np.random.standard_normal((self.params.time_steps, n_sims))
        for t in range(1, self.params.time_steps + 1):
            drift_term = kappa * (theta - paths[t - 1]) * dt
            diffusion_term = sigma * np.sqrt(dt) * random_normals[t - 1]
            paths[t] = paths[t - 1] + drift_term + diffusion_term
            paths[t] = np.maximum(paths[t], 0.001)
        return paths

    def heston_model(
        self, n_simulations: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate paths using Heston stochastic volatility model"""
        n_sims = n_simulations or self.params.num_simulations
        dt = self.dt
        v0 = self.params.initial_variance
        kappa = self.params.variance_mean_reversion
        theta = self.params.long_term_variance
        sigma_v = self.params.vol_of_vol
        rho = self.params.correlation
        price_paths = np.zeros((self.params.time_steps + 1, n_sims))
        variance_paths = np.zeros((self.params.time_steps + 1, n_sims))
        price_paths[0] = self.params.initial_price
        variance_paths[0] = v0
        random_normals = np.random.standard_normal((2, self.params.time_steps, n_sims))
        z1 = random_normals[0]
        z2 = rho * random_normals[0] + np.sqrt(1 - rho**2) * random_normals[1]
        for t in range(1, self.params.time_steps + 1):
            V_prev = np.maximum(variance_paths[t - 1], 0)
            variance_drift = kappa * (theta - V_prev) * dt
            variance_diffusion = sigma_v * np.sqrt(V_prev * dt) * z2[t - 1]
            variance_paths[t] = np.maximum(
                variance_paths[t - 1] + variance_drift + variance_diffusion, 0
            )
            price_drift_log = (self.params.risk_free_rate - 0.5 * V_prev) * dt
            price_diffusion_log = np.sqrt(V_prev * dt) * z1[t - 1]
            price_paths[t] = price_paths[t - 1] * np.exp(
                price_drift_log + price_diffusion_log
            )
        return (price_paths, variance_paths)

    def _generate_paths_chunk(self, n_simulations: int) -> np.ndarray:
        """Generate a chunk of paths for parallel processing"""
        if self.params.random_seed is not None:
            np.random.seed(self.params.random_seed + mp.current_process().pid)
        if self.params.process_type == ProcessType.GEOMETRIC_BROWNIAN_MOTION:
            paths = self.geometric_brownian_motion(n_simulations)
        elif self.params.process_type == ProcessType.MEAN_REVERTING:
            paths = self.mean_reverting_process(n_simulations)
        elif self.params.process_type == ProcessType.HESTON:
            price_paths, _ = self.heston_model(n_simulations)
            paths = price_paths
        else:
            paths = self.geometric_brownian_motion(n_simulations)
        return paths

    def generate_paths(self) -> np.ndarray:
        """Generate price paths based on specified process"""
        if self.params.parallel:
            num_cores = mp.cpu_count()
            sims_per_core = self.params.num_simulations // num_cores
            remainder = self.params.num_simulations % num_cores
            sims_list = [sims_per_core] * num_cores
            for i in range(remainder):
                sims_list[i] += 1
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                paths_chunks = list(executor.map(self._generate_paths_chunk, sims_list))
            paths = np.concatenate(paths_chunks, axis=1)
            return paths
        else:
            if self.params.process_type == ProcessType.GEOMETRIC_BROWNIAN_MOTION:
                return self.geometric_brownian_motion()
            elif self.params.process_type == ProcessType.MEAN_REVERTING:
                return self.mean_reverting_process()
            elif self.params.process_type == ProcessType.HESTON:
                price_paths, _ = self.heston_model()
                return price_paths
            else:
                raise ValueError(
                    f"Unsupported process type: {self.params.process_type}"
                )

    def calculate_payoff(
        self, paths: np.ndarray, payoff_spec: OptionPayoff
    ) -> np.ndarray:
        """Calculate option payoffs from price paths (undiscounted)"""
        option_style = payoff_spec.option_style.lower()
        option_type = payoff_spec.option_type.lower()
        strike = payoff_spec.strike_price

        if option_type not in ["call", "put"]:
            raise ValueError(f"Unsupported option type: {option_type}")

        if option_style == "european":
            final_prices = paths[-1]
            payoffs = np.maximum(
                (
                    final_prices - strike
                    if option_type == "call"
                    else strike - final_prices
                ),
                0.0,
            )
        elif option_style == "asian":
            # Asian option: payoff depends on the average price over a period
            avg_prices = np.mean(paths[payoff_spec.averaging_start_step :, :], axis=0)
            payoffs = np.maximum(
                (avg_prices - strike if option_type == "call" else strike - avg_prices),
                0.0,
            )
        elif option_style == "barrier":
            # Barrier option: Knock-out (Down-and-Out Call and Up-and-Out Put)
            barrier = payoff_spec.barrier_level
            barrier_type = payoff_spec.barrier_type.lower()
            final_prices = paths[-1]
            payoffs = np.maximum(
                (
                    final_prices - strike
                    if option_type == "call"
                    else strike - final_prices
                ),
                0.0,
            )
            if barrier_type == "down_and_out":
                # Check if the price ever hit or went below the barrier
                hit_barrier = np.any(paths <= barrier, axis=0)
                payoffs[hit_barrier] = 0.0
            elif barrier_type == "up_and_out":
                # Check if the price ever hit or went above the barrier
                hit_barrier = np.any(paths >= barrier, axis=0)
                payoffs[hit_barrier] = 0.0
            else:
                logger.warning(f"Unsupported barrier type for MC: {barrier_type}")
                payoffs = np.zeros_like(payoffs)
        elif option_style == "lookback":
            # Lookback option: payoff depends on the maximum or minimum price over a period
            if option_type == "call":
                # Floating strike lookback call: max(S_T - min(S_t))
                min_prices = np.min(paths[payoff_spec.lookback_start_step :, :], axis=0)
                payoffs = np.maximum(paths[-1] - min_prices, 0.0)
            else:
                # Floating strike lookback put: max(max(S_t) - S_T)
                max_prices = np.max(paths[payoff_spec.lookback_start_step :, :], axis=0)
                payoffs = np.maximum(max_prices - paths[-1], 0.0)
        else:
            raise ValueError(f"Unsupported option style: {option_style}")

        return payoffs

    def _price_only(self, payoff_spec: OptionPayoff) -> float:
        """Price an option without computing Greeks (avoids recursion)."""
        paths = self.generate_paths()
        payoffs = self.calculate_payoff(paths, payoff_spec)
        discounted_payoffs = payoffs * np.exp(
            -self.params.risk_free_rate * self.params.time_horizon
        )
        return float(np.mean(discounted_payoffs))

    def price_option(
        self, payoff_spec: OptionPayoff, paths: Optional[np.ndarray] = None
    ) -> SimulationResult:
        """
        Price an option using Monte Carlo simulation.
        """
        start_time = datetime.now()
        if paths is None:
            paths = self.generate_paths()

        payoffs = self.calculate_payoff(paths, payoff_spec)
        discounted_payoffs = payoffs * np.exp(
            -self.params.risk_free_rate * self.params.time_horizon
        )
        option_price = np.mean(discounted_payoffs)
        std_dev = np.std(discounted_payoffs)
        standard_error = std_dev / np.sqrt(self.params.num_simulations)

        z_score = stats.norm.ppf(0.975)
        conf_interval = (
            option_price - z_score * standard_error,
            option_price + z_score * standard_error,
        )

        greeks = self._calculate_greeks_finite_difference(payoff_spec)

        computation_time = (datetime.now() - start_time).total_seconds()

        return SimulationResult(
            option_price=option_price,
            standard_error=standard_error,
            confidence_interval=conf_interval,
            paths=paths,
            payoffs=payoffs,
            greeks=greeks,
            computation_time=computation_time,
            method_used=f"MC-{self.params.process_type.value}",
        )

    def _calculate_greeks_finite_difference(
        self, payoff_spec: OptionPayoff
    ) -> Dict[str, float]:
        """
        Calculate option Greeks using finite difference method.
        Uses proper parameter copies to avoid mutating shared state.
        """
        import copy

        greeks = {}
        h = 0.01

        def _price_with(field: str, delta: float) -> float:
            p = copy.copy(self.params)
            setattr(p, field, getattr(p, field) + delta)
            return MCSimulator(p)._price_only(payoff_spec)

        price_up = _price_with("initial_price", h)
        price_down = _price_with("initial_price", -h)
        price_center = self._price_only(payoff_spec)
        greeks["delta"] = (price_up - price_down) / (2 * h)
        greeks["gamma"] = (price_up - 2 * price_center + price_down) / (h**2)

        h_vol = 0.001
        greeks["vega"] = (
            _price_with("volatility", h_vol) - _price_with("volatility", -h_vol)
        ) / (2 * h_vol)

        h_time = 1 / 365
        greeks["theta"] = -(
            _price_with("time_horizon", h_time) - _price_with("time_horizon", -h_time)
        ) / (2 * h_time)

        h_rate = 0.0001
        greeks["rho"] = (
            _price_with("risk_free_rate", h_rate)
            - _price_with("risk_free_rate", -h_rate)
        ) / (2 * h_rate)

        return greeks

    def calculate_risk_metrics(
        self, paths: np.ndarray, confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR).
        """
        # Assuming paths represent the portfolio value at the end of the horizon
        # We are interested in the loss, so we look at the negative returns
        returns = paths[-1] / paths[0] - 1
        losses = -returns
        losses = np.sort(losses)
        alpha = 1 - confidence_level
        var_index = int(np.floor(alpha * self.params.num_simulations))
        var = losses[var_index]
        cvar = np.mean(losses[var_index:])
        return {"VaR": var, "CVaR": cvar}

    def run_stress_test(
        self,
        payoff_spec: OptionPayoff,
        stress_scenarios: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Run stress tests by modifying simulation parameters.
        """
        results = {}
        original_params = self.params
        for scenario_name, changes in stress_scenarios.items():
            # Create a copy of parameters and apply changes
            stressed_params = SimulationParameters(**original_params.__dict__)
            for param, value in changes.items():
                setattr(stressed_params, param, value)

            # Run simulation with stressed parameters
            stressed_simulator = MCSimulator(stressed_params)
            stressed_price = stressed_simulator.price_option(payoff_spec).option_price
            results[scenario_name] = stressed_price

        # Restore original parameters
        self.params = original_params
        return results

    def plot_paths(self, paths: np.ndarray, num_paths_to_plot: int = 10) -> None:
        """
        Plot a subset of the simulated price paths.
        """
        time_grid = np.linspace(0, self.params.time_horizon, self.params.time_steps + 1)
        plt.figure(figsize=(10, 6))
        for i in range(min(num_paths_to_plot, paths.shape[1])):
            plt.plot(time_grid, paths[:, i])
        plt.title(f"Monte Carlo Simulation Paths ({self.params.process_type.value})")
        plt.xlabel("Time (Years)")
        plt.ylabel("Asset Price")
        plt.grid(True)
        plt.show()

    def plot_convergence(self, payoff_spec: OptionPayoff) -> Dict[str, List[float]]:
        """
        Plot the convergence of the option price as the number of simulations increases.
        """
        num_sims_list = np.logspace(
            1, np.log10(self.params.num_simulations), 20, dtype=int
        )
        prices = []
        std_errors = []
        for num_sims in num_sims_list:
            temp_params = SimulationParameters(**self.params.__dict__)
            temp_params.num_simulations = num_sims
            temp_simulator = MCSimulator(temp_params)
            result = temp_simulator.price_option(payoff_spec)
            prices.append(result.option_price)
            std_errors.append(result.standard_error)

        plt.figure(figsize=(10, 6))
        plt.plot(num_sims_list, prices, label="Option Price")
        plt.fill_between(
            num_sims_list,
            np.array(prices) - 1.96 * np.array(std_errors),
            np.array(prices) + 1.96 * np.array(std_errors),
            alpha=0.2,
            label="95% Confidence Interval",
        )
        plt.xscale("log")
        plt.title("Monte Carlo Convergence")
        plt.xlabel("Number of Simulations (log scale)")
        plt.ylabel("Option Price")
        plt.legend()
        plt.grid(True)
        plt.show()

        return {"num_simulations": num_sims_list.tolist(), "prices": prices}


try:
    mc_simulator = MCSimulator(
        SimulationParameters(
            initial_price=100.0,
            risk_free_rate=0.05,
            volatility=0.2,
            time_horizon=1.0,
            time_steps=252,
            num_simulations=10000,
        )
    )
except Exception:
    mc_simulator = None
