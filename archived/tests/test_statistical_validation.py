"""
Test statistical validation framework
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.statistical_validation import (
    TimesFMStatisticalValidator,
    BenchmarkModel,
    generate_validation_report
)
import logging

logging.basicConfig(level=logging.INFO)

def create_test_predictions():
    """Create synthetic test predictions"""
    np.random.seed(42)

    n_samples = 1000
    n_stocks = 5

    stocks = ['VCB', 'VIC', 'VNM', 'FPT', 'HPG']

    # Create data
    actual_data = {}
    model_predictions = {}
    benchmark_predictions = {}

    for symbol in stocks:
        # Actual volatility
        actual = np.abs(np.random.randn(n_samples) * 0.02)

        # Model predictions (better than benchmark)
        model_pred = actual + np.random.randn(n_samples) * 0.01

        # Benchmark predictions (worse)
        benchmark_pred = actual + np.random.randn(n_samples) * 0.03

        actual_data[symbol] = actual
        model_predictions[symbol] = model_pred
        benchmark_predictions[symbol] = benchmark_pred

    return actual_data, model_predictions, benchmark_predictions


def test_diebold_mariano():
    """Test Diebold-Mariano test"""
    print("\n" + "="*50)
    print("TEST: Diebold-Mariano Test")
    print("="*50)

    validator = TimesFMStatisticalValidator()

    # Create test data
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])  # Better predictions
    benchmark_pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5])  # Worse predictions

    # Run test
    result = validator.diebold_mariano_test(actual, model_pred, benchmark_pred)

    print(f"✓ Diebold-Mariano test completed")
    print(f"  Statistic: {result['dm_statistic']:.4f}")
    print(f"  P-value: {result['p_value']:.4f}")
    print(f"  Significant: {result['is_significant']}")
    print(f"  Interpretation: {result['interpretation']}")

    return result


def test_giacomini_white():
    """Test Giacomini-White test"""
    print("\n" + "="*50)
    print("TEST: Giacomini-White Test")
    print("="*50)

    validator = TimesFMStatisticalValidator()

    # Create test data
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    model_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1])
    benchmark_pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])

    # Run test
    result = validator.giacomini_white_test(actual, model_pred, benchmark_pred)

    print(f"✓ Giacomini-White test completed")
    print(f"  Statistic: {result['gw_statistic']:.4f}")
    print(f"  P-value: {result['p_value']:.4f}")
    print(f"  Significant: {result['is_significant']}")
    print(f"  Interpretation: {result['interpretation']}")

    return result


def test_evaluation_metrics():
    """Test evaluation metrics calculation"""
    print("\n" + "="*50)
    print("TEST: Evaluation Metrics")
    print("="*50)

    validator = TimesFMStatisticalValidator()

    # Create test data
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predicted = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

    # Calculate metrics
    metrics = validator.calculate_evaluation_metrics(actual, predicted)

    print(f"✓ Evaluation metrics calculated")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAPE: {metrics['mape']:.6f}")
    print(f"  Correlation: {metrics['correlation']:.6f}")
    print(f"  Samples: {metrics['samples']}")

    return metrics


def test_single_stock_validation():
    """Test single stock validation"""
    print("\n" + "="*50)
    print("TEST: Single Stock Validation")
    print("="*50)

    validator = TimesFMStatisticalValidator()

    # Create test data
    np.random.seed(42)
    n_samples = 1000

    actual = np.abs(np.random.randn(n_samples) * 0.02)
    model_pred = actual + np.random.randn(n_samples) * 0.01
    benchmark_pred = actual + np.random.randn(n_samples) * 0.03

    # Run validation
    result = validator.validate_single_stock(actual, model_pred, benchmark_pred, "VCB")

    print(f"✓ Single stock validation completed")
    print(f"  Stock: {result['stock_name']}")
    print(f"  Samples: {result['samples']}")
    print(f"  Model RMSE: {result['model_metrics']['rmse']:.6f}")
    print(f"  Benchmark RMSE: {result['benchmark_metrics']['rmse']:.6f}")
    print(f"  Improvement: {result['summary']['improvement_pct']:.2f}%")
    print(f"  Statistically significant: {result['summary']['statistically_significant']}")

    return result


def test_comprehensive_validation():
    """Test comprehensive validation across multiple stocks"""
    print("\n" + "="*50)
    print("TEST: Comprehensive Validation")
    print("="*50)

    validator = TimesFMStatisticalValidator()

    # Create test data
    actual_data, model_predictions, benchmark_predictions = create_test_predictions()

    # Run comprehensive validation
    results = validator.comprehensive_validation(actual_data, model_predictions, benchmark_predictions)

    print(f"✓ Comprehensive validation completed")
    print(f"  Total stocks: {results['summary']['total_stocks']}")
    print(f"  Statistically significant: {results['summary']['statistically_significant']}")
    print(f"  Significance rate: {results['summary']['significance_rate']:.1f}%")
    print(f"  Average improvement: {results['summary']['average_improvement_pct']:.2f}%")

    # Show individual results
    print(f"\n  Individual Results:")
    for symbol, result in results['individual_results'].items():
        if 'error' not in result:
            print(f"    {symbol}: {result['summary']['improvement_pct']:.1f}% improvement, " +
                  f"significant={result['summary']['statistically_significant']}")

    return results


def test_benchmark_models():
    """Test benchmark model predictions"""
    print("\n" + "="*50)
    print("TEST: Benchmark Models")
    print("="*50)

    benchmark = BenchmarkModel()

    # Create test data
    np.random.seed(42)
    returns = np.random.randn(1000) * 0.02

    # Test different benchmark methods
    print("  Testing Random Walk:")
    rw_forecast = benchmark.random_walk_forecast(returns[:100])
    print(f"    Forecast shape: {rw_forecast.shape}")
    print(f"    Sample forecast: {rw_forecast[0]:.6f}")

    print("\n  Testing Moving Average:")
    ma_forecast = benchmark.moving_average_forecast(returns, window=20)
    print(f"    Forecast shape: {ma_forecast.shape}")
    print(f"    Sample forecast: {ma_forecast[0]:.6f}")

    print("\n  Testing Simple Volatility:")
    vol_forecast = benchmark.simple_volatility_forecast(returns, window=20)
    print(f"    Forecast shape: {vol_forecast.shape}")
    print(f"    Sample forecast: {vol_forecast[0]:.6f}")

    # Try GARCH if available
    print("\n  Testing GARCH:")
    try:
        garch_forecast = benchmark.garch_forecast(returns)
        print(f"    Forecast shape: {garch_forecast.shape}")
        print(f"    Sample forecast: {garch_forecast[0]:.6f}")
    except Exception as e:
        print(f"    GARCH not available or failed: {e}")

    print("✓ Benchmark models tested")

    return benchmark


def test_validation_report():
    """Test validation report generation"""
    print("\n" + "="*50)
    print("TEST: Validation Report Generation")
    print("="*50)

    validator = TimesFMStatisticalValidator()

    # Create test data
    actual_data, model_predictions, benchmark_predictions = create_test_predictions()

    # Run validation
    results = validator.comprehensive_validation(actual_data, model_predictions, benchmark_predictions)

    # Generate report
    output_path = "test_validation_report.txt"
    generate_validation_report(results, output_path)

    print(f"✓ Validation report generated")
    print(f"  Output path: {output_path}")

    # Read and display first few lines
    with open(output_path, 'r') as f:
        lines = f.readlines()
        print(f"\n  Report preview (first 10 lines):")
        for i, line in enumerate(lines[:10]):
            print(f"    {line.rstrip()}")

    return results


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "="*50)
    print("TEST: Edge Cases and Error Handling")
    print("="*50)

    validator = TimesFMStatisticalValidator()

    # Test with NaN values
    print("  Testing with NaN values:")
    actual = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    model_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    benchmark_pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

    try:
        result = validator.diebold_mariano_test(actual, model_pred, benchmark_pred)
        print(f"    Handled NaN values: ✓")
    except Exception as e:
        print(f"    Error with NaN values: {e}")

    # Test with different length arrays
    print("\n  Testing with different length arrays:")
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model_pred = np.array([1.1, 2.1, 3.1])  # Shorter
    benchmark_pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

    try:
        result = validator.diebold_mariano_test(actual, model_pred, benchmark_pred)
        print(f"    Handled different lengths: ✓")
    except Exception as e:
        print(f"    Error with different lengths: {e}")

    # Test with perfect predictions
    print("\n  Testing with perfect predictions:")
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Perfect
    benchmark_pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

    try:
        result = validator.diebold_mariano_test(actual, model_pred, benchmark_pred)
        print(f"    Perfect predictions - Significant: {result['is_significant']}")
    except Exception as e:
        print(f"    Error with perfect predictions: {e}")

    print("✓ Edge cases tested")


def main():
    """Main test execution"""
    print("\n" + "="*50)
    print("STATISTICAL VALIDATION FRAMEWORK TESTS")
    print("="*50)

    try:
        # Test 1: Diebold-Mariano Test
        test_diebold_mariano()

        # Test 2: Giacomini-White Test
        test_giacomini_white()

        # Test 3: Evaluation Metrics
        test_evaluation_metrics()

        # Test 4: Single Stock Validation
        test_single_stock_validation()

        # Test 5: Comprehensive Validation
        test_comprehensive_validation()

        # Test 6: Benchmark Models
        test_benchmark_models()

        # Test 7: Validation Report
        test_validation_report()

        # Test 8: Edge Cases
        test_edge_cases()

        print("\n" + "="*50)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("="*50)

    except Exception as e:
        print(f"\nERROR: Test execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
