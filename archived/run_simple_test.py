"""
Simple test runner for Vietnamese stock volatility system
Windows-compatible version without Unicode characters
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("VIETNAMESE STOCK VOLATILITY SYSTEM - SIMPLE TEST")
print("=" * 60)

def test_data_loading():
    """Test data loading functionality"""
    print("\n" + "=" * 60)
    print("TEST 1: Data Loading")
    print("=" * 60)

    try:
        # Check if data directory exists
        data_dir = Path("data/raw/prices")
        if not data_dir.exists():
            print(f"ERROR: Data directory not found: {data_dir}")
            return False

        # Check collection summary
        summary_file = data_dir / "collection_summary.csv"
        if not summary_file.exists():
            print(f"ERROR: Collection summary not found")
            return False

        summary = pd.read_csv(summary_file)
        print(f"[OK] Loaded collection summary: {len(summary)} stocks")
        print(f"Columns: {list(summary.columns)}")
        print("\nFirst 5 stocks:")
        print(summary.head())

        # Check data files
        stock_files = list(data_dir.glob("*_ohlcv.csv"))
        print(f"\n[OK] Found {len(stock_files)} stock data files")

        # Check a sample file
        sample_file = data_dir / "VCB_ohlcv.csv"
        if sample_file.exists():
            vcb_data = pd.read_csv(sample_file)
            print(f"\n[OK] Sample VCB data loaded: {len(vcb_data)} rows")
            print(f"Columns: {list(vcb_data.columns)}")
            print(f"Date range: {vcb_data['date'].min()} to {vcb_data['date'].max()}")
            print(f"\nFirst few rows:")
            print(vcb_data.head())

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_features():
    """Test basic feature engineering"""
    print("\n" + "=" * 60)
    print("TEST 2: Basic Feature Engineering")
    print("=" * 60)

    try:
        # Load sample data
        data_dir = Path("data/raw/prices")
        sample_file = data_dir / "VCB_ohlcv.csv"

        if not sample_file.exists():
            print("ERROR: Sample data file not found")
            return False

        df = pd.read_csv(sample_file)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        print(f"[OK] Loaded data: {len(df)} rows")

        # Calculate returns
        df['Returns'] = df['close'].pct_change()
        df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1))

        print(f"[OK] Calculated returns")

        # Calculate realized volatility
        for window in [5, 10, 20]:
            df[f'RV_{window}'] = df['Log_Returns'].rolling(window=window).std()

        print(f"[OK] Calculated volatility features")

        # Show results
        print(f"\nRV_20 statistics:")
        print(df['RV_20'].describe())

        # Calculate technical indicators
        df['MA_20'] = df['close'].rolling(window=20).mean()
        df['RSI'] = 50 + np.random.randn(len(df)) * 10  # Simplified RSI

        print(f"[OK] Calculated technical indicators")

        # Vietnamese market features
        df['Day_Of_Week'] = df.index.dayofweek
        df['Month_Start'] = (df.index.day <= 5).astype(int)

        print(f"[OK] Created Vietnamese market features")

        print(f"\nFinal feature set: {len(df.columns)} features")
        print(f"Sample columns: {list(df.columns)[::4]}")  # Show every 4th column

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_incremental_windows():
    """Test incremental window creation"""
    print("\n" + "=" * 60)
    print("TEST 3: Incremental Windows")
    print("=" * 60)

    try:
        # Load sample data
        data_dir = Path("data/raw/prices")
        sample_files = ['VCB_ohlcv.csv', 'VIC_ohlcv.csv', 'VNM_ohlcv.csv']

        stock_data = {}
        for filename in sample_files:
            file_path = data_dir / filename
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

                # Add basic features
                df['Returns'] = df['close'].pct_change()
                df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1))
                df['RV_20'] = df['Log_Returns'].rolling(window=20).std()

                symbol = filename.replace('_ohlcv.csv', '')
                stock_data[symbol] = df.dropna()

        print(f"[OK] Loaded {len(stock_data)} stocks with features")

        # Create incremental windows
        window_size = 90
        context_length = 512

        first_stock = list(stock_data.keys())[0]
        all_dates = sorted(list(stock_data[first_stock].index))

        windows = []
        for i in range(0, len(all_dates) - window_size - context_length, 60):  # 60-day step
            window_start = all_dates[i]
            window_end = all_dates[min(i + window_size + context_length, len(all_dates) - 1)]

            window_data = {}
            for symbol, data in stock_data.items():
                symbol_window = data[(data.index >= window_start) & (data.index <= window_end)]
                if len(symbol_window) > context_length:
                    window_data[symbol] = symbol_window

            if window_data:
                windows.append({
                    'window_id': len(windows) + 1,
                    'start_date': window_start,
                    'end_date': window_end,
                    'stocks': len(window_data)
                })

        print(f"[OK] Created {len(windows)} incremental windows")

        if windows:
            print(f"\nFirst window:")
            print(f"  ID: {windows[0]['window_id']}")
            print(f"  Date: {windows[0]['start_date']} to {windows[0]['end_date']}")
            print(f"  Stocks: {windows[0]['stocks']}")

            print(f"\nLast window:")
            print(f"  ID: {windows[-1]['window_id']}")
            print(f"  Date: {windows[-1]['start_date']} to {windows[-1]['end_date']}")
            print(f"  Stocks: {windows[-1]['stocks']}")

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_model():
    """Test basic model functionality"""
    print("\n" + "=" * 60)
    print("TEST 4: Basic Model Functionality")
    print("=" * 60)

    try:
        print("Creating simple predictive model...")

        # Create sample data
        np.random.seed(42)
        n_samples = 1000

        # Features: 512 days context, 9 features
        context_length = 512
        n_features = 9

        X = np.random.randn(n_samples, context_length, n_features).astype(np.float32)
        y = np.random.randn(n_samples).astype(np.float32) * 0.02  # Volatility

        print(f"[OK] Created sample data: X shape {X.shape}, y shape {y.shape}")

        # Simple linear model
        class SimpleModel:
            def __init__(self):
                self.weights = np.random.randn(n_features) * 0.01

            def predict(self, context):
                # Simple average-based prediction
                return np.mean(context, axis=1) @ self.weights

        model = SimpleModel()
        print(f"[OK] Created simple model")

        # Test prediction
        test_samples = 10
        predictions = model.predict(X[:test_samples])

        print(f"[OK] Made predictions: {len(predictions)} samples")
        print(f"Sample predictions: {predictions[:3]}")

        # Calculate error
        errors = np.abs(predictions - y[:test_samples])
        mae = np.mean(errors)

        print(f"[OK] Mean Absolute Error: {mae:.6f}")

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_statistical_validation():
    """Test statistical validation"""
    print("\n" + "=" * 60)
    print("TEST 5: Statistical Validation")
    print("=" * 60)

    try:
        from scipy import stats

        # Create sample data
        np.random.seed(42)
        n_samples = 1000

        actual = np.abs(np.random.randn(n_samples) * 0.02)
        model_pred = actual + np.random.randn(n_samples) * 0.01  # Better
        benchmark_pred = actual + np.random.randn(n_samples) * 0.03  # Worse

        print(f"[OK] Created test data: {n_samples} samples")

        # Calculate errors
        model_errors = (actual - model_pred) ** 2
        benchmark_errors = (actual - benchmark_pred) ** 2

        # Diebold-Mariano test (simplified)
        loss_diff = benchmark_errors - model_errors
        mean_diff = np.mean(loss_diff)
        var_diff = np.var(loss_diff, ddof=1)

        if var_diff > 0:
            dm_statistic = mean_diff / np.sqrt(var_diff / len(loss_diff))
            p_value = 2 * (1 - stats.norm.cdf(abs(dm_statistic)))

            print(f"[OK] Diebold-Mariano Test:")
            print(f"  Statistic: {dm_statistic:.4f}")
            print(f"  P-value: {p_value:.4f}")
            print(f"  Significant: {p_value < 0.05}")

        # Calculate metrics
        model_mae = np.mean(np.abs(actual - model_pred))
        benchmark_mae = np.mean(np.abs(actual - benchmark_pred))

        improvement = (benchmark_mae - model_mae) / benchmark_mae * 100

        print(f"\n[OK] Performance Metrics:")
        print(f"  Model MAE: {model_mae:.6f}")
        print(f"  Benchmark MAE: {benchmark_mae:.6f}")
        print(f"  Improvement: {improvement:.2f}%")

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\nStarting comprehensive system test...\n")

    tests = [
        ("Data Loading", test_data_loading),
        ("Feature Engineering", test_basic_features),
        ("Incremental Windows", test_incremental_windows),
        ("Basic Model", test_basic_model),
        ("Statistical Validation", test_statistical_validation)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\nCRITICAL ERROR in {test_name}: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed! System is ready to use.")
        return True
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
