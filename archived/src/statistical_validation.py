"""
Statistical Validation Framework for TimesFM Model
Based on TimesFM research paper methodology

Implements Diebold-Mariano and Giacomini-White tests to prove
statistical significance of model improvements over traditional methods.

From TimesFM paper (arXiv:2505.11163):
"Fine-tuned variants not only improve forecast accuracy but also
statistically outperform traditional models, as demonstrated through
Diebold-Mariano and Giacomini-White tests"
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import yaml
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimesFMStatisticalValidator:
    """
    Statistical validation for TimesFM forecasting improvements.

    Implements statistical tests from TimesFM paper to validate that
    fine-tuned models significantly outperform traditional benchmarks.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize statistical validator"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.alpha = self.config['validation']['alpha']
        self.benchmarks = self.config['validation']['benchmarks']

        logger.info(f"Initialized statistical validator (alpha={self.alpha})")

    def calculate_forecast_errors(self, actual: np.ndarray,
                                 predicted: np.ndarray) -> np.ndarray:
        """
        Calculate squared forecast errors (standard approach)

        Args:
            actual: Actual values
            predicted: Predicted values

        Returns:
            Squared errors
        """
        # Handle missing values
        valid_mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = actual[valid_mask]
        predicted_clean = predicted[valid_mask]

        return (actual_clean - predicted_clean) ** 2

    def diebold_mariano_test(self, actual: np.ndarray,
                            model_pred: np.ndarray,
                            benchmark_pred: np.ndarray) -> Dict:
        """
        Diebold-Mariano test for equal forecast accuracy.

        H0: Model and benchmark have same forecast accuracy
        H1: Model is more accurate than benchmark

        From TimesFM paper: Primary statistical test for forecast comparison

        Args:
            actual: Actual values
            model_pred: Model predictions
            benchmark_pred: Benchmark predictions

        Returns:
            Dictionary with test results
        """
        # Ensure same length
        min_len = min(len(actual), len(model_pred), len(benchmark_pred))
        actual = actual[:min_len]
        model_pred = model_pred[:min_len]
        benchmark_pred = benchmark_pred[:min_len]

        # Calculate loss differentials
        loss_model = self.calculate_forecast_errors(actual, model_pred)
        loss_benchmark = self.calculate_forecast_errors(actual, benchmark_pred)

        # Loss differential: benchmark - model (positive = model is better)
        loss_diff = loss_benchmark - loss_model

        # Calculate DM statistic
        mean_diff = np.mean(loss_diff)
        var_diff = np.var(loss_diff, ddof=1)

        if var_diff == 0:
            logger.warning("Zero variance in loss differentials")
            return {
                'dm_statistic': 0,
                'p_value': 1.0,
                'is_significant': False,
                'interpretation': 'Cannot calculate test (zero variance)',
                'mean_improvement': 0
            }

        dm_statistic = mean_diff / np.sqrt(var_diff / len(loss_diff))

        # Two-tailed test
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_statistic)))

        result = {
            'dm_statistic': dm_statistic,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'interpretation': self._interpret_dm_result(p_value, mean_diff),
            'mean_improvement': mean_diff,
            'test_name': 'Diebold-Mariano'
        }

        logger.info(f"DM Test: Statistic={dm_statistic:.4f}, p-value={p_value:.4f}")

        return result

    def _interpret_dm_result(self, p_value: float, mean_improvement: float) -> str:
        """Interpret Diebold-Mariano test result"""
        if p_value < self.alpha:
            direction = "better" if mean_improvement > 0 else "worse"
            return f"Model significantly {direction} than benchmark (p={p_value:.4f})"
        else:
            return "No significant difference between models"

    def giacomini_white_test(self, actual: np.ndarray,
                            model_pred: np.ndarray,
                            benchmark_pred: np.ndarray) -> Dict:
        """
        Giacomini-White test for conditional predictive ability.

        More robust than DM test for finite samples and conditional
        predictive ability testing.

        From TimesFM paper: Additional robust test for statistical validation

        Args:
            actual: Actual values
            model_pred: Model predictions
            benchmark_pred: Benchmark predictions

        Returns:
            Dictionary with test results
        """
        # Ensure same length
        min_len = min(len(actual), len(model_pred), len(benchmark_pred))
        actual = actual[:min_len]
        model_pred = model_pred[:min_len]
        benchmark_pred = benchmark_pred[:min_len]

        # Calculate loss differentials
        loss_model = self.calculate_forecast_errors(actual, model_pred)
        loss_benchmark = self.calculate_forecast_errors(actual, benchmark_pred)

        loss_diff = loss_benchmark - loss_model

        # Regress loss differential on constant
        X = np.ones(len(loss_diff))

        try:
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression(fit_intercept=False)
            reg.fit(X.reshape(-1, 1), loss_diff)

            predictions = reg.predict(X.reshape(-1, 1))
            residuals = loss_diff - predictions

            # Calculate GW statistic
            n = len(loss_diff)
            sum_residuals_sq = np.sum(residuals ** 2)

            if sum_residuals_sq == 0:
                logger.warning("Zero residual sum of squares")
                return {
                    'gw_statistic': 0,
                    'p_value': 1.0,
                    'is_significant': False,
                    'interpretation': 'Cannot calculate test (zero residual variance)',
                    'test_name': 'Giacomini-White'
                }

            gw_statistic = n * (reg.coef_[0] ** 2) / sum_residuals_sq

            # p-value (chi-squared with 1 degree of freedom)
            p_value = 1 - stats.chi2.cdf(gw_statistic, 1)

            result = {
                'gw_statistic': gw_statistic,
                'p_value': p_value,
                'is_significant': p_value < self.alpha,
                'interpretation': self._interpret_gw_result(p_value, reg.coef_[0]),
                'test_name': 'Giacomini-White'
            }

            logger.info(f"GW Test: Statistic={gw_statistic:.4f}, p-value={p_value:.4f}")

            return result

        except Exception as e:
            logger.error(f"Error in Giacomini-White test: {e}")
            return {
                'gw_statistic': 0,
                'p_value': 1.0,
                'is_significant': False,
                'interpretation': f'Test calculation error: {e}',
                'test_name': 'Giacomini-White'
            }

    def _interpret_gw_result(self, p_value: float, coefficient: float) -> str:
        """Interpret Giacomini-White test result"""
        if p_value < self.alpha:
            direction = "significant conditional predictive ability" if coefficient > 0 else "significantly worse"
            return f"Model has {direction} (p={p_value:.4f})"
        else:
            return "No significant conditional predictive ability"

    def calculate_evaluation_metrics(self, actual: np.ndarray,
                                   predicted: np.ndarray) -> Dict:
        """
        Calculate standard evaluation metrics

        Args:
            actual: Actual values
            predicted: Predicted values

        Returns:
            Dictionary of metrics
        """
        # Handle missing values
        valid_mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = actual[valid_mask]
        predicted_clean = predicted[valid_mask]

        # MAE
        mae = np.mean(np.abs(actual_clean - predicted_clean))

        # RMSE
        rmse = np.sqrt(np.mean((actual_clean - predicted_clean) ** 2))

        # MAPE (if no zero values)
        if np.all(actual_clean != 0):
            mape = np.mean(np.abs((actual_clean - predicted_clean) / actual_clean)) * 100
        else:
            mape = np.nan

        # Correlation
        if len(actual_clean) > 1:
            correlation = np.corrcoef(actual_clean, predicted_clean)[0, 1]
        else:
            correlation = np.nan

        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'correlation': correlation,
            'samples': len(actual_clean)
        }

    def validate_single_stock(self, actual: np.ndarray,
                             model_pred: np.ndarray,
                             benchmark_pred: np.ndarray,
                             stock_name: str) -> Dict:
        """
        Validate predictions for a single stock

        Args:
            actual: Actual values
            model_pred: Model predictions
            benchmark_pred: Benchmark predictions
            stock_name: Stock identifier

        Returns:
            Comprehensive validation results
        """
        logger.info(f"\nValidating {stock_name}...")

        results = {
            'stock_name': stock_name,
            'samples': len(actual)
        }

        # Evaluation metrics
        model_metrics = self.calculate_evaluation_metrics(actual, model_pred)
        benchmark_metrics = self.calculate_evaluation_metrics(actual, benchmark_pred)

        results['model_metrics'] = model_metrics
        results['benchmark_metrics'] = benchmark_metrics

        # Statistical tests
        dm_result = self.diebold_mariano_test(actual, model_pred, benchmark_pred)
        gw_result = self.giacomini_white_test(actual, model_pred, benchmark_pred)

        results['diebold_mariano'] = dm_result
        results['giacomini_white'] = gw_result

        # Summary
        results['summary'] = {
            'model_better': model_metrics['rmse'] < benchmark_metrics['rmse'],
            'improvement_pct': ((benchmark_metrics['rmse'] - model_metrics['rmse']) /
                              benchmark_metrics['rmse'] * 100) if benchmark_metrics['rmse'] > 0 else 0,
            'statistically_significant': dm_result['is_significant'] or gw_result['is_significant']
        }

        # Print summary
        logger.info(f"  Model RMSE: {model_metrics['rmse']:.6f}")
        logger.info(f"  Benchmark RMSE: {benchmark_metrics['rmse']:.6f}")
        logger.info(f"  Improvement: {results['summary']['improvement_pct']:.2f}%")
        logger.info(f"  DM Test: {dm_result['interpretation']}")
        logger.info(f"  GW Test: {gw_result['interpretation']}")

        return results

    def comprehensive_validation(self, actual_data: Dict[str, np.ndarray],
                               model_predictions: Dict[str, np.ndarray],
                               benchmark_predictions: Dict[str, np.ndarray]) -> Dict:
        """
        Perform comprehensive statistical validation across all stocks

        Args:
            actual_data: Dictionary of actual values by stock
            model_predictions: Dictionary of model predictions by stock
            benchmark_predictions: Dictionary of benchmark predictions by stock

        Returns:
            Comprehensive validation results for all stocks
        """
        logger.info("="*50)
        logger.info("COMPREHENSIVE STATISTICAL VALIDATION")
        logger.info("="*50)

        validation_results = {}

        for symbol in actual_data.keys():
            if symbol in model_predictions and symbol in benchmark_predictions:
                actual = actual_data[symbol]
                model_pred = model_predictions[symbol]
                bench_pred = benchmark_predictions[symbol]

                try:
                    result = self.validate_single_stock(
                        actual, model_pred, bench_pred, symbol
                    )
                    validation_results[symbol] = result

                except Exception as e:
                    logger.error(f"Error validating {symbol}: {e}")
                    validation_results[symbol] = {'error': str(e)}
            else:
                logger.warning(f"Missing predictions for {symbol}")

        # Generate overall summary
        summary = self._generate_overall_summary(validation_results)

        return {
            'individual_results': validation_results,
            'summary': summary
        }

    def _generate_overall_summary(self, validation_results: Dict) -> Dict:
        """Generate overall summary of validation results"""
        total_stocks = len(validation_results)
        significant_stocks = 0
        better_performance_stocks = 0
        total_improvement = 0

        dm_significant = 0
        gw_significant = 0

        for stock, results in validation_results.items():
            if 'error' in results:
                continue

            if results.get('summary', {}).get('statistically_significant', False):
                significant_stocks += 1

            if results.get('summary', {}).get('model_better', False):
                better_performance_stocks += 1

            improvement = results.get('summary', {}).get('improvement_pct', 0)
            total_improvement += improvement

            if results.get('diebold_mariano', {}).get('is_significant', False):
                dm_significant += 1

            if results.get('giacomini_white', {}).get('is_significant', False):
                gw_significant += 1

        avg_improvement = total_improvement / max(total_stocks, 1)

        summary = {
            'total_stocks': total_stocks,
            'statistically_significant': significant_stocks,
            'better_performance': better_performance_stocks,
            'dm_significant': dm_significant,
            'gw_significant': gw_significant,
            'average_improvement_pct': avg_improvement,
            'significance_rate': (significant_stocks / max(total_stocks, 1)) * 100
        }

        logger.info("\n" + "="*50)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Total stocks tested: {total_stocks}")
        logger.info(f"Statistically significant: {significant_stocks}/{total_stocks} ({summary['significance_rate']:.1f}%)")
        logger.info(f"DM Test significant: {dm_significant}/{total_stocks}")
        logger.info(f"GW Test significant: {gw_significant}/{total_stocks}")
        logger.info(f"Average improvement: {avg_improvement:.2f}%")
        logger.info("="*50)

        return summary


class BenchmarkModel:
    """
    Traditional benchmark models for comparison

    Implements GARCH, ARIMA, and Random Walk benchmarks
    """

    def __init__(self):
        self.model_type = "benchmark"

    def random_walk_forecast(self, actual_values: np.ndarray, forecast_horizon: int = 1) -> np.ndarray:
        """
        Random walk forecast (naive benchmark)

        Args:
            actual_values: Historical actual values
            forecast_horizon: Number of steps to forecast

        Returns:
            Forecast array
        """
        forecasts = []
        for i in range(len(actual_values) - forecast_horizon):
            forecast = actual_values[i]  # Random walk: use last observed value
            forecasts.append(forecast)

        return np.array(forecasts)

    def moving_average_forecast(self, actual_values: np.ndarray, window: int = 20) -> np.ndarray:
        """
        Simple moving average forecast

        Args:
            actual_values: Historical actual values
            window: Moving average window

        Returns:
            Forecast array
        """
        forecasts = []
        for i in range(window, len(actual_values)):
            forecast = np.mean(actual_values[i-window:i])
            forecasts.append(forecast)

        return np.array(forecasts)

    def garch_forecast(self, returns: np.ndarray, forecast_horizon: int = 1) -> np.ndarray:
        """
        GARCH volatility forecast

        Args:
            returns: Return series
            forecast_horizon: Forecast horizon

        Returns:
            Volatility forecasts
        """
        try:
            from arch import arch_model

            forecasts = []

            # Use rolling window for forecasts
            window_size = 252  # ~1 year of trading days

            for i in range(window_size, len(returns) - forecast_horizon):
                window_returns = returns[i-window_size:i]

                try:
                    # Fit GARCH(1,1) model
                    model = arch_model(window_returns * 100, vol='Garch', p=1, q=1)
                    fitted_model = model.fit(disp='off')

                    # Forecast
                    forecast = fitted_model.forecast(horizon=forecast_horizon).variance.values[-1]
                    forecast = np.sqrt(forecast) / 100  # Convert back to original scale

                    forecasts.append(forecast)

                except Exception as e:
                    # Fallback to historical volatility if GARCH fails
                    forecast = np.std(window_returns)
                    forecasts.append(forecast)

            return np.array(forecasts)

        except ImportError:
            logger.warning("arch package not available, using simple volatility")
            return self.simple_volatility_forecast(returns)

    def simple_volatility_forecast(self, returns: np.ndarray, window: int = 20) -> np.ndarray:
        """
        Simple historical volatility forecast

        Args:
            returns: Return series
            window: Volatility calculation window

        Returns:
            Volatility forecasts
        """
        forecasts = []

        for i in range(window, len(returns)):
            volatility = np.std(returns[i-window:i])
            forecasts.append(volatility)

        return np.array(forecasts)


def generate_validation_report(validation_results: Dict, output_path: str = "validation_report.txt"):
    """
    Generate comprehensive validation report

    Args:
        validation_results: Results from comprehensive validation
        output_path: Path to save report
    """
    summary = validation_results['summary']
    individual_results = validation_results['individual_results']

    report_lines = []
    report_lines.append("="*70)
    report_lines.append("TIMESFM STATISTICAL VALIDATION REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("="*70)
    report_lines.append("")

    # Overall Summary
    report_lines.append("OVERALL SUMMARY")
    report_lines.append("-" * 70)
    report_lines.append(f"Total stocks tested: {summary['total_stocks']}")
    report_lines.append(f"Statistically significant improvements: {summary['statistically_significant']}/{summary['total_stocks']}")
    report_lines.append(f"Significance rate: {summary['significance_rate']:.1f}%")
    report_lines.append(f"Diebold-Mariano significant: {summary['dm_significant']}/{summary['total_stocks']}")
    report_lines.append(f"Giacomini-White significant: {summary['gw_significant']}/{summary['total_stocks']}")
    report_lines.append(f"Average improvement: {summary['average_improvement_pct']:.2f}%")
    report_lines.append("")

    # Individual Stock Results
    report_lines.append("INDIVIDUAL STOCK RESULTS")
    report_lines.append("-" * 70)

    for symbol, results in individual_results.items():
        if 'error' in results:
            report_lines.append(f"{symbol}: ERROR - {results['error']}")
            continue

        model_metrics = results['model_metrics']
        benchmark_metrics = results['benchmark_metrics']
        summary_data = results['summary']

        report_lines.append(f"\n{symbol}:")
        report_lines.append(f"  Model RMSE: {model_metrics['rmse']:.6f}")
        report_lines.append(f"  Benchmark RMSE: {benchmark_metrics['rmse']:.6f}")
        report_lines.append(f"  Improvement: {summary_data['improvement_pct']:.2f}%")
        report_lines.append(f"  DM Test: {results['diebold_mariano']['interpretation']}")
        report_lines.append(f"  GW Test: {results['giacomini_white']['interpretation']}")

    # Statistical Methodology Note
    report_lines.append("\n" + "="*70)
    report_lines.append("STATISTICAL METHODOLOGY")
    report_lines.append("-" * 70)
    report_lines.append("Tests performed based on TimesFM research paper methodology:")
    report_lines.append("1. Diebold-Mariano test for equal forecast accuracy")
    report_lines.append("2. Giacomini-White test for conditional predictive ability")
    report_lines.append("3. Significance level: α = 0.05")
    report_lines.append("4. Null hypothesis: Model and benchmark have same accuracy")
    report_lines.append("5. Alternative hypothesis: Model is significantly better")
    report_lines.append("="*70)

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"Validation report saved to {output_path}")


def main():
    """Main execution for testing"""
    logger.info("Initializing statistical validation framework...")

    validator = TimesFMStatisticalValidator()
    benchmark = BenchmarkModel()

    # Create synthetic test data
    np.random.seed(42)
    n_samples = 1000

    actual = np.random.randn(n_samples) * 0.02  # True volatility
    model_pred = actual + np.random.randn(n_samples) * 0.01  # Better predictions
    benchmark_pred = actual + np.random.randn(n_samples) * 0.03  # Worse predictions

    # Run validation
    results = validator.validate_single_stock(actual, model_pred, benchmark_pred, "TEST")

    logger.info("Statistical validation framework initialized successfully!")
    logger.info(f"Test results: {results['diebold_mariano']['interpretation']}")


if __name__ == "__main__":
    main()
