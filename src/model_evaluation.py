"""
Model Evaluation and Statistical Testing for TimesFM VN30
Implements comprehensive validation framework with Diebold-Mariano tests
Following pfnet-research validation methodology
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import yaml
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments/model_evaluation.log'),
        logging.StreamHandler()
    ]
)


class TimesFMModelEvaluator:
    """
    Comprehensive model evaluation with statistical testing

    Financial Validation Framework:
    - Standard metrics: QLIKE, R², RMSE, MSE
    - Statistical significance: Diebold-Mariano tests
    - Financial metrics: Sharpe ratio, maximum drawdown
    """

    def __init__(self, config_path: str = 'configs/config.yaml'):
        """
        Initialize model evaluator

        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Evaluation results storage
        self.evaluation_results = {}

    # MANDATORY METRIC FUNCTIONS (exact names from CLAUDE.md)

    @staticmethod
    def calculate_qlike(actuals: np.ndarray, predictions: np.ndarray) -> float:
        """
        Calculate QLIKE metric for volatility forecasting

        Args:
            actuals: Ground truth volatility values
            predictions: Predicted volatility values

        Returns:
            QLIKE score (lower is better)

        Financial Note:
        QLIKE is the standard metric for volatility forecasting evaluation.
        It properly handles heteroskedasticity in financial time series.

        QLIKE = mean(actual/pred + log(pred) - 1)
        """
        # Prevent division by zero and log of zero
        predictions_safe = np.maximum(predictions, 1e-8)
        actuals_safe = np.maximum(actuals, 1e-8)

        qlike = np.mean(actuals_safe / predictions_safe + np.log(predictions_safe) - 1)

        return float(qlike)

    @staticmethod
    def calculate_r2(actuals: np.ndarray, predictions: np.ndarray) -> float:
        """
        Calculate R² score (coefficient of determination)

        Args:
            actuals: Ground truth values
            predictions: Predicted values

        Returns:
            R² score (higher is better, max 1.0)

        Financial Note:
        R² measures the proportion of variance explained by the model.
        Target: R² > 0.5 (model explains >50% of variance)
        """
        # Calculate residual sum of squares
        ss_res = np.sum((actuals - predictions) ** 2)

        # Calculate total sum of squares
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)

        # Handle edge case
        if ss_tot == 0:
            return -float('inf')

        r2 = 1 - (ss_res / ss_tot)

        return float(r2)

    @staticmethod
    def calculate_rmse(actuals: np.ndarray, predictions: np.ndarray) -> float:
        """
        Calculate Root Mean Square Error

        Args:
            actuals: Ground truth values
            predictions: Predicted values

        Returns:
            RMSE score (lower is better)

        Financial Note:
        RMSE is in the same units as the target variable (volatility).
        More interpretable than MSE for financial applications.
        """
        mse = np.mean((actuals - predictions) ** 2)
        rmse = np.sqrt(mse)

        return float(rmse)

    @staticmethod
    def calculate_mse(actuals: np.ndarray, predictions: np.ndarray) -> float:
        """
        Calculate Mean Square Error

        Args:
            actuals: Ground truth values
            predictions: Predicted values

        Returns:
            MSE score (lower is better)

        Financial Note:
        MSE is the standard loss function for regression.
        Penalizes larger errors more heavily (squared term).
        """
        mse = np.mean((actuals - predictions) ** 2)

        return float(mse)

    @staticmethod
    def calculate_mae(actuals: np.ndarray, predictions: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error

        Args:
            actuals: Ground truth values
            predictions: Predicted values

        Returns:
            MAE score (lower is better)

        Financial Note:
        MAE is more robust to outliers than MSE.
        Useful supplementary metric for volatility forecasting.
        """
        mae = np.mean(np.abs(actuals - predictions))

        return float(mae)

    # STATISTICAL TESTING FUNCTIONS

    def diebold_mariano_test(self, actual: np.ndarray,
                           model_pred: np.ndarray,
                           bench_pred: np.ndarray) -> Dict[str, float]:
        """
        Diebold-Mariano test for forecast accuracy comparison

        Args:
            actual: Ground truth values
            model_pred: Fine-tuned model predictions
            bench_pred: Baseline/benchmark predictions

        Returns:
            Dictionary with DM test results

        Financial Note:
        Diebold-Mariano test determines whether one forecast is significantly
        better than another. Critical for validating fine-tuning improvements.

        H0: Both forecasts have equal accuracy
        H1: Model forecast is significantly better

        Target: p-value < 0.05 (statistically significant improvement)
        """
        # Calculate loss differentials (squared error loss)
        loss_model = (actual - model_pred) ** 2
        loss_bench = (actual - bench_pred) ** 2

        loss_diff = loss_model - loss_bench

        # Calculate mean and variance of loss differential
        mean_loss_diff = np.mean(loss_diff)
        var_loss_diff = np.var(loss_diff, ddof=1)

        # Calculate DM statistic
        n = len(loss_diff)
        if var_loss_diff == 0 or n == 0:
            dm_statistic = 0.0
            p_value = 1.0
        else:
            dm_statistic = mean_loss_diff / np.sqrt(var_loss_diff / n)
            p_value = 2 * (1 - stats.norm.cdf(abs(dm_statistic)))

        result = {
            'dm_statistic': float(dm_statistic),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'mean_loss_diff': float(mean_loss_diff),
            'interpretation': self._interpret_dm_test(p_value, mean_loss_diff)
        }

        return result

    def _interpret_dm_test(self, p_value: float, mean_loss_diff: float) -> str:
        """Interpret Diebold-Mariano test results"""
        if p_value >= 0.05:
            return "Not statistically significant (p >= 0.05)"
        elif mean_loss_diff < 0:
            return "Model significantly better than baseline (p < 0.05)"
        else:
            return "Baseline significantly better than model (p < 0.05)"

    def calculate_sharpe_ratio(self, returns: np.ndarray,
                             risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio for volatility-based trading strategy

        Args:
            returns: Portfolio returns from volatility-based strategy
            risk_free_rate: Risk-free rate (default: 0)

        Returns:
            Sharpe ratio (higher is better)

        Financial Note:
        Sharpe ratio measures risk-adjusted returns.
        Target: 0.8-1.5 vs baseline 0.42

        Sharpe = (mean(returns) - risk_free_rate) / std(returns)
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate
        sharpe = np.mean(excess_returns) / np.std(returns)

        # Annualize (assuming daily returns)
        sharpe_annualized = sharpe * np.sqrt(252)  # 252 trading days per year

        return float(sharpe_annualized)

    def calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown from cumulative returns

        Args:
            cumulative_returns: Cumulative portfolio returns

        Returns:
            Maximum drawdown (lower is better, expressed as negative percentage)

        Financial Note:
        Maximum drawdown measures the largest peak-to-trough decline.
        Critical for risk assessment in trading strategies.
        """
        if len(cumulative_returns) == 0:
            return 0.0

        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)

        # Calculate drawdown at each point
        drawdown = (cumulative_returns - running_max) / running_max

        # Maximum drawdown
        max_drawdown = np.min(drawdown)

        return float(max_drawdown)

    def evaluate_model_predictions(self, actuals: np.ndarray,
                                  predictions: np.ndarray,
                                  baseline_predictions: Optional[np.ndarray] = None) -> Dict:
        """
        Comprehensive model evaluation with all required metrics

        Args:
            actuals: Ground truth values
            predictions: Model predictions
            baseline_predictions: Baseline/benchmark predictions (optional)

        Returns:
            Dictionary with all evaluation metrics
        """
        self.logger.info("=" * 70)
        self.logger.info("[COMPREHENSIVE MODEL EVALUATION]")
        self.logger.info("=" * 70)

        # Calculate all mandatory metrics
        qlike = self.calculate_qlike(actuals, predictions)
        r2 = self.calculate_r2(actuals, predictions)
        rmse = self.calculate_rmse(actuals, predictions)
        mse = self.calculate_mse(actuals, predictions)
        mae = self.calculate_mae(actuals, predictions)

        results = {
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(actuals),
            'metrics': {
                'qlike': qlike,
                'r2': r2,
                'rmse': rmse,
                'mse': mse,
                'mae': mae
            }
        }

        # Log results
        self.logger.info(f"QLIKE: {qlike:.6f}")
        self.logger.info(f"R²: {r2:.4f}")
        self.logger.info(f"RMSE: {rmse:.6f}")
        self.logger.info(f"MSE: {mse:.6f}")
        self.logger.info(f"MAE: {mae:.6f}")

        # Diebold-Mariano test (if baseline provided)
        if baseline_predictions is not None:
            dm_results = self.diebold_mariano_test(actuals, predictions, baseline_predictions)

            results['statistical_tests'] = {
                'diebold_mariano': dm_results
            }

            self.logger.info("=" * 70)
            self.logger.info("[STATISTICAL SIGNIFICANCE TESTING]")
            self.logger.info("=" * 70)
            self.logger.info(f"DM Statistic: {dm_results['dm_statistic']:.4f}")
            self.logger.info(f"P-value: {dm_results['p_value']:.4f}")
            self.logger.info(f"Significant: {dm_results['significant']}")
            self.logger.info(f"Interpretation: {dm_results['interpretation']}")

        # Success criteria check
        results['success_criteria'] = {
            'r2_target_0.5': r2 > 0.5,
            'significance_target_0.05': dm_results['p_value'] < 0.05 if baseline_predictions is not None else False
        }

        self.logger.info("=" * 70)
        self.logger.info("[SUCCESS CRITERIA CHECK]")
        self.logger.info("=" * 70)
        self.logger.info(f"R² > 0.5: {r2 > 0.5} (achieved: {r2:.4f})")

        if baseline_predictions is not None:
            self.logger.info(f"p < 0.05: {dm_results['p_value'] < 0.05} (achieved: {dm_results['p_value']:.4f})")

        return results

    def backtest_volatility_strategy(self, actual_volatility: np.ndarray,
                                   predicted_volatility: np.ndarray,
                                   price_data: Optional[np.ndarray] = None) -> Dict:
        """
        Simple backtesting framework for volatility-based trading strategy

        Args:
            actual_volatility: Actual volatility values
            predicted_volatility: Predicted volatility values
            price_data: Optional price data for more realistic backtesting

        Returns:
            Dictionary with backtesting results

        Financial Note:
        Simple volatility-based strategy:
        - High predicted volatility → reduce position size
        - Low predicted volatility → increase position size
        """
        self.logger.info("=" * 70)
        self.logger.info("[VOLATILITY-BASED STRATEGY BACKTESTING]")
        self.logger.info("=" * 70)

        # Simple strategy: volatility-adjusted returns
        # Normalize predictions
        pred_norm = (predicted_volatility - np.mean(predicted_volatility)) / np.std(predicted_volatility)

        # Generate simple trading signals based on volatility
        # Low volatility → buy signal, High volatility → sell signal
        signals = -pred_norm  # Negative because high vol = reduce position

        # Normalize signals to [-1, 1]
        signals = np.clip(signals / np.std(signals), -1, 1)

        # If price data provided, calculate actual returns
        if price_data is not None and len(price_data) == len(predicted_volatility):
            # Calculate price returns
            price_returns = np.diff(price_data) / price_data[:-1]

            # Trading strategy returns
            strategy_returns = signals[:-1] * price_returns
        else:
            # Simplified: use actual volatility as proxy for market returns
            # (not realistic, but provides framework)
            market_returns = np.diff(actual_volatility) / (actual_volatility[:-1] + 1e-8)
            strategy_returns = signals[:-1] * market_returns

        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + strategy_returns) - 1

        # Calculate financial metrics
        sharpe_ratio = self.calculate_sharpe_ratio(strategy_returns)
        max_drawdown = self.calculate_max_drawdown(cumulative_returns)
        total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0.0

        results = {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'cumulative_returns': cumulative_returns.tolist()
        }

        self.logger.info(f"Total Return: {total_return:.4f}")
        self.logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        self.logger.info(f"Max Drawdown: {max_drawdown:.4f}")

        return results

    def save_evaluation_results(self, results: Dict,
                               filename: str = "model_evaluation_results.json") -> None:
        """
        Save evaluation results to JSON

        Args:
            results: Evaluation results dictionary
            filename: Output filename
        """
        experiments_dir = Path(self.config['experiment_tracking']['experiments_dir'])
        experiments_dir.mkdir(parents=True, exist_ok=True)

        output_path = experiments_dir / filename

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"[SAVE] Evaluation results: {output_path}")

    def generate_evaluation_report(self, results: Dict) -> str:
        """
        Generate human-readable evaluation report

        Args:
            results: Evaluation results dictionary

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("TIMESFM VN30 MODEL EVALUATION REPORT")
        report.append("=" * 70)
        report.append(f"Timestamp: {results['timestamp']}")
        report.append(f"Number of Samples: {results['num_samples']}")
        report.append("")

        report.append("PERFORMANCE METRICS")
        report.append("-" * 70)
        metrics = results['metrics']
        report.append(f"QLIKE: {metrics['qlike']:.6f}")
        report.append(f"R² Score: {metrics['r2']:.4f}")
        report.append(f"RMSE: {metrics['rmse']:.6f}")
        report.append(f"MSE: {metrics['mse']:.6f}")
        report.append(f"MAE: {metrics['mae']:.6f}")
        report.append("")

        if 'statistical_tests' in results:
            report.append("STATISTICAL SIGNIFICANCE")
            report.append("-" * 70)
            dm = results['statistical_tests']['diebold_mariano']
            report.append(f"Diebold-Mariano Statistic: {dm['dm_statistic']:.4f}")
            report.append(f"P-value: {dm['p_value']:.4f}")
            report.append(f"Significant: {dm['significant']}")
            report.append(f"Interpretation: {dm['interpretation']}")
            report.append("")

        report.append("SUCCESS CRITERIA")
        report.append("-" * 70)
        success = results['success_criteria']
        report.append(f"R² > 0.5: {success['r2_target_0.5']}")
        if 'significance_target_0.05' in success:
            report.append(f"Statistical Significance (p < 0.05): {success['significance_target_0.05']}")
        report.append("")

        if 'backtesting' in results:
            report.append("BACKTESTING RESULTS")
            report.append("-" * 70)
            bt = results['backtesting']
            report.append(f"Total Return: {bt['total_return']:.4f}")
            report.append(f"Sharpe Ratio: {bt['sharpe_ratio']:.4f}")
            report.append(f"Max Drawdown: {bt['max_drawdown']:.4f}")
            report.append("")

        report.append("=" * 70)

        return "\n".join(report)


def main():
    """Main execution function for model evaluation"""
    import sys

    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize evaluator
    evaluator = TimesFMModelEvaluator()

    # Example evaluation (would be called with actual model predictions)
    # This is a placeholder demonstrating the evaluation framework

    logging.info("[INFO] Model evaluation framework ready")
    logging.info("[INFO] Use with actual model predictions from training")

    return 0


if __name__ == "__main__":
    sys.exit(main())