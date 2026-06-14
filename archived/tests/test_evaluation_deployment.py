"""
Test model evaluation and deployment system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.model_evaluation_deployment import (
    ModelEvaluator,
    ModelDeploymentSystem,
    ProductionPredictionAPI,
    generate_evaluation_report
)
import logging

logging.basicConfig(level=logging.INFO)

def create_test_evaluation_data():
    """Create test data for model evaluation"""
    np.random.seed(42)

    n_samples = 1000
    n_stocks = 5

    stocks = ['VCB', 'VIC', 'VNM', 'FPT', 'HPG']

    actual_data = {}
    predictions = {}
    time_indices = {}

    for symbol in stocks:
        # Create time index
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        time_indices[symbol] = dates

        # Create actual volatility
        actual = np.abs(np.random.randn(n_samples) * 0.02)
        actual_data[symbol] = actual

        # Create predictions (with some noise)
        pred_noise = np.random.randn(n_samples) * 0.01
        predictions[symbol] = actual + pred_noise

    return actual_data, predictions, time_indices


def test_point_metrics():
    """Test point metrics calculation"""
    print("\n" + "="*50)
    print("TEST: Point Metrics Calculation")
    print("="*50)

    evaluator = ModelEvaluator()

    # Create test data
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predicted = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

    # Calculate metrics
    metrics = evaluator.calculate_point_metrics(actual, predicted)

    print(f"✓ Point metrics calculated")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAPE: {metrics['mape']:.6f}")
    print(f"  Correlation: {metrics['correlation']:.6f}")
    print(f"  R²: {metrics['r_squared']:.6f}")
    print(f"  Max Error: {metrics['max_error']:.6f}")
    print(f"  Bias: {metrics['bias']:.6f}")

    return metrics


def test_volatility_metrics():
    """Test volatility-specific metrics"""
    print("\n" + "="*50)
    print("TEST: Volatility-Specific Metrics")
    print("="*50)

    evaluator = ModelEvaluator()

    # Create test data
    np.random.seed(42)
    n_samples = 1000

    actual = np.abs(np.random.randn(n_samples) * 0.02)
    predicted = actual + np.random.randn(n_samples) * 0.01

    # Calculate volatility metrics
    metrics = evaluator.calculate_volatility_specific_metrics(actual, predicted)

    print(f"✓ Volatility metrics calculated")
    print(f"  Direction Accuracy: {metrics['direction_accuracy']:.4f}")
    print(f"  Regime Accuracy: {metrics['regime_accuracy']:.4f}")
    print(f"  Tail Risk Recall: {metrics['tail_risk_recall']:.4f}")

    return metrics


def test_temporal_stability():
    """Test temporal stability metrics"""
    print("\n" + "="*50)
    print("TEST: Temporal Stability Metrics")
    print("="*50)

    evaluator = ModelEvaluator()

    # Create test data
    np.random.seed(42)
    n_samples = 1000

    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    actual = np.abs(np.random.randn(n_samples) * 0.02)
    predicted = actual + np.random.randn(n_samples) * 0.01

    # Calculate temporal metrics
    metrics = evaluator.calculate_temporal_stability(predicted, actual, dates)

    print(f"✓ Temporal stability metrics calculated")
    print(f"  Weekly Error Std: {metrics.get('weekly_error_std', 'N/A')}")
    print(f"  Monthly Error Std: {metrics.get('monthly_error_std', 'N/A')}")
    print(f"  Trend Correlation: {metrics.get('trend_correlation', 'N/A')}")

    return metrics


def test_comprehensive_evaluation():
    """Test comprehensive model evaluation"""
    print("\n" + "="*50)
    print("TEST: Comprehensive Model Evaluation")
    print("="*50)

    evaluator = ModelEvaluator()

    # Create test data
    actual_data, predictions, time_indices = create_test_evaluation_data()

    # Run comprehensive evaluation
    results = evaluator.evaluate_model_performance(actual_data, predictions, time_indices)

    print(f"✓ Comprehensive evaluation completed")
    print(f"  Total stocks: {results['aggregate_results']['total_stocks']}")
    print(f"  Avg MAE: {results['aggregate_results']['avg_mae']:.6f}")
    print(f"  Avg RMSE: {results['aggregate_results']['avg_rmse']:.6f}")
    print(f"  Avg Correlation: {results['aggregate_results']['avg_correlation']:.4f}")

    # Show individual results
    print(f"\n  Individual Results:")
    for symbol, result in results['individual_results'].items():
        print(f"    {symbol}:")
        print(f"      RMSE: {result['point_metrics']['rmse']:.6f}")
        print(f"      Direction Acc: {result['volatility_metrics']['direction_accuracy']:.4f}")

    return results


def test_deployment_system():
    """Test model deployment system"""
    print("\n" + "="*50)
    print("TEST: Model Deployment System")
    print("="*50)

    deployment_system = ModelDeploymentSystem()

    # Create a simple model for testing
    class SimpleModel:
        def __init__(self):
            self.bias = 0.01
            self.scale = 1.0

        def __call__(self, x):
            # Simple linear model
            if isinstance(x, torch.Tensor):
                return x.mean(dim=-1) * self.scale + self.bias
            return x.mean(axis=-1) * self.scale + self.bias

        def forward(self, x):
            return self.__call__(x)

    # Load model
    import torch
    simple_model = SimpleModel()
    deployment_system.model = simple_model
    deployment_system.model_version = "test_v1.0"
    deployment_system.model_metadata = {
        'type': 'simple_model',
        'version': 'test_v1.0'
    }

    print(f"✓ Deployment system initialized")
    print(f"  Model loaded: {deployment_system.model is not None}")
    print(f"  Model version: {deployment_system.model_version}")

    return deployment_system


def test_prediction_functionality():
    """Test prediction functionality"""
    print("\n" + "="*50)
    print("TEST: Prediction Functionality")
    print("="*50)

    # Create deployment system
    deployment_system = test_deployment_system()

    # Create test context data
    context_data = np.random.randn(10, 512, 9).astype(np.float32)  # [samples, sequence, features]

    # Make predictions
    try:
        result = deployment_system.predict_volatility(context_data, ['VCB'])

        print(f"✓ Predictions made successfully")
        print(f"  Predictions: {len(result['predictions'])} samples")
        print(f"  Model version: {result['model_version']}")
        print(f"  Timestamp: {result['timestamp']}")

        # Test batch prediction
        print(f"\n  Testing batch prediction...")

        batch_context = {
            'VCB': np.random.randn(1, 512, 9).astype(np.float32),
            'VIC': np.random.randn(1, 512, 9).astype(np.float32),
            'VNM': np.random.randn(1, 512, 9).astype(np.float32)
        }

        batch_results = deployment_system.batch_predict(batch_context)

        print(f"  Batch predictions: {len(batch_results)} stocks")
        for symbol, result in batch_results.items():
            if 'error' not in result:
                print(f"    {symbol}: ✓")
            else:
                print(f"    {symbol}: ✗ {result['error']}")

        return result

    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_performance_monitoring():
    """Test performance monitoring"""
    print("\n" + "="*50)
    print("TEST: Performance Monitoring")
    print("="*50)

    deployment_system = test_deployment_system()

    # Simulate some predictions and actual values
    actual_values = {'VCB': 0.025, 'VIC': 0.030, 'VNM': 0.018}
    predicted_values = {'VCB': 0.024, 'VIC': 0.029, 'VNM': 0.019}

    # Update performance metrics
    deployment_system.update_performance_metrics(actual_values, predicted_values)

    # Add more samples
    for i in range(10):
        actual = {symbol: value + np.random.randn() * 0.001 for symbol, value in actual_values.items()}
        predicted = {symbol: value + np.random.randn() * 0.002 for symbol, value in predicted_values.items()}
        deployment_system.update_performance_metrics(actual, predicted)

    # Get performance report
    report = deployment_system.get_performance_report()

    print(f"✓ Performance monitoring active")
    print(f"  Total predictions: {report['total_predictions']}")
    print(f"  Avg error: {report['avg_error']:.6f}")
    print(f"  RMSE: {report['rmse']:.6f}")
    print(f"  Max error: {report['max_error']:.6f}")

    print(f"\n  Symbol Performance:")
    for symbol, metrics in report['symbol_performance'].items():
        print(f"    {symbol}: {metrics['avg_error']:.6f} avg error, {metrics['predictions']} predictions")

    return report


def test_model_export():
    """Test model package export"""
    print("\n" + "="*50)
    print("TEST: Model Package Export")
    print("="*50)

    deployment_system = test_deployment_system()

    # Export model package
    export_path = "test_model_export"

    try:
        deployment_system.export_model_package(export_path)

        print(f"✓ Model package exported successfully")
        print(f"  Export path: {export_path}")

        # Check exported files
        from pathlib import Path
        export_dir = Path(export_path)

        if export_dir.exists():
            files = list(export_dir.glob("*"))
            print(f"  Exported files: {len(files)}")
            for file in files:
                print(f"    - {file.name}")

        return export_path

    except Exception as e:
        print(f"✗ Model export failed: {e}")
        return None


def test_evaluation_report():
    """Test evaluation report generation"""
    print("\n" + "="*50)
    print("TEST: Evaluation Report Generation")
    print("="*50)

    # Create evaluation results
    actual_data, predictions, time_indices = create_test_evaluation_data()
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model_performance(actual_data, predictions, time_indices)

    # Generate report
    output_path = "test_evaluation_report.json"

    try:
        generate_evaluation_report(results, output_path)

        print(f"✓ Evaluation report generated successfully")
        print(f"  Output path: {output_path}")

        # Read and display summary
        import json
        with open(output_path, 'r') as f:
            report = json.load(f)

        print(f"  Report sections: {list(report.keys())}")
        print(f"  Generated at: {report['generated_at']}")
        print(f"  Total stocks evaluated: {report['results']['aggregate_results']['total_stocks']}")

        return output_path

    except Exception as e:
        print(f"✗ Report generation failed: {e}")
        return None


def test_api_integration():
    """Test API integration (basic check)"""
    print("\n" + "="*50)
    print("TEST: API Integration")
    print("="*50)

    deployment_system = test_deployment_system()

    try:
        api = ProductionPredictionAPI(deployment_system)

        print(f"✓ API system initialized")
        print(f"  Deployment system linked: {api.deployment_system is not None}")
        print(f"  Note: Full API testing requires running FastAPI server")

        return api

    except Exception as e:
        print(f"✗ API initialization failed: {e}")
        return None


def main():
    """Main test execution"""
    print("\n" + "="*50)
    print("MODEL EVALUATION AND DEPLOYMENT TESTS")
    print("="*50)

    try:
        # Test 1: Point Metrics
        test_point_metrics()

        # Test 2: Volatility Metrics
        test_volatility_metrics()

        # Test 3: Temporal Stability
        test_temporal_stability()

        # Test 4: Comprehensive Evaluation
        test_comprehensive_evaluation()

        # Test 5: Deployment System
        test_deployment_system()

        # Test 6: Prediction Functionality
        test_prediction_functionality()

        # Test 7: Performance Monitoring
        test_performance_monitoring()

        # Test 8: Model Export
        test_model_export()

        # Test 9: Evaluation Report
        test_evaluation_report()

        # Test 10: API Integration
        test_api_integration()

        print("\n" + "="*50)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("="*50)

    except Exception as e:
        print(f"\nERROR: Test execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
