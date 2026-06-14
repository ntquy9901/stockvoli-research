"""
Test data preprocessing module
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.data_preprocessing import (
    VietnameseDataLoader,
    VietnameseVolatilityFeatures,
    IncrementalDataWindowManager
)
from src.utils import (
    validate_data_quality,
    align_data_dates,
    log_data_summary
)
import logging

logging.basicConfig(level=logging.INFO)

def test_data_loading():
    """Test data loading functionality"""
    print("\n" + "="*50)
    print("TEST: Data Loading")
    print("="*50)

    loader = VietnameseDataLoader()

    # Test collection summary loading
    summary = loader.load_collection_summary()
    print(f"✓ Loaded collection summary: {len(summary)} stocks")
    print(summary.head())

    # Test loading a specific stock
    test_stock = "VCB"
    vcb_data = loader.load_stock_data(test_stock)
    print(f"✓ Loaded {test_stock}: {len(vcb_data)} days")

    if not vcb_data.empty:
        print(f"  Date range: {vcb_data.index.min()} to {vcb_data.index.max()}")
        print(f"  Columns: {list(vcb_data.columns)}")

    # Test loading all stocks
    all_stocks = loader.load_all_stocks()
    print(f"✓ Loaded {len(all_stocks)} stocks total")

    return all_stocks


def test_feature_engineering(stock_data):
    """Test feature engineering"""
    print("\n" + "="*50)
    print("TEST: Feature Engineering")
    print("="*50)

    feature_engineer = VietnameseVolatilityFeatures()

    # Test on one stock first
    test_symbol = list(stock_data.keys())[0]
    test_data = stock_data[test_symbol].copy()

    print(f"Testing on {test_symbol}")
    print(f"Original shape: {test_data.shape}")

    # Test each feature type
    print("\n1. Volatility Features:")
    vol_features = feature_engineer.create_volatility_features(test_data)
    print(f"   Shape after volatility features: {vol_features.shape}")
    vol_cols = [col for col in vol_features.columns if col.startswith('RV_')]
    print(f"   Volatility columns: {vol_cols[:5]}...")

    print("\n2. Technical Indicators:")
    tech_features = feature_engineer.create_technical_indicators(vol_features)
    print(f"   Shape after technical indicators: {tech_features.shape}")
    tech_cols = [col for col in tech_features.columns if col in ['MA_10', 'MA_20', 'RSI', 'MACD']]
    print(f"   Technical columns: {tech_cols}")

    print("\n3. Vietnamese Market Features:")
    viet_features = feature_engineer.create_vietnamese_market_features(tech_features)
    print(f"   Shape after Vietnamese features: {viet_features.shape}")
    viet_cols = [col for col in viet_features.columns if 'Vietnamese' in col or 'Tet' in col]
    print(f"   Vietnamese columns: {[col for col in viet_features.columns if col in ['Is_Monday', 'Is_Tet_Period']]}")

    print("\n4. Market Regime Features:")
    regime_features = feature_engineer.create_market_regime_features(viet_features)
    print(f"   Final shape: {regime_features.shape}")
    print(f"   Total features: {len(regime_features.columns)}")

    # Test on all stocks
    print(f"\nProcessing features for all {len(stock_data)} stocks...")
    processed_data = {}

    for symbol, data in stock_data.items():
        try:
            processed_data[symbol] = feature_engineer.process_all_features(data)
            print(f"  ✓ {symbol}: {len(processed_data[symbol])} rows, {len(processed_data[symbol].columns)} features")
        except Exception as e:
            print(f"  ✗ {symbol}: Error - {e}")

    print(f"\n✓ Successfully processed {len(processed_data)} stocks")

    return processed_data


def test_data_validation(stock_data):
    """Test data validation utilities"""
    print("\n" + "="*50)
    print("TEST: Data Validation")
    print("="*50)

    from src.utils import validate_data_quality, calculate_data_statistics

    # Test individual stock validation
    test_symbol = list(stock_data.keys())[0]
    quality_report = validate_data_quality(stock_data[test_symbol], test_symbol)

    print(f"Quality Report for {test_symbol}:")
    for key, value in quality_report.items():
        print(f"  {key}: {value}")

    # Test overall statistics
    print("\nCalculating statistics for all stocks...")
    stats_df = calculate_data_statistics(stock_data)
    print(stats_df)

    # Test data alignment
    print("\nTesting data alignment...")
    aligned_data = align_data_dates(stock_data)
    print(f"✓ Aligned data to common dates")

    return aligned_data


def test_incremental_windows(processed_data):
    """Test incremental window creation"""
    print("\n" + "="*50)
    print("TEST: Incremental Windows")
    print("="*50)

    window_manager = IncrementalDataWindowManager()

    # Create windows
    windows = window_manager.create_incremental_windows(processed_data)

    print(f"✓ Created {len(windows)} incremental windows")

    if windows:
        # Show first window details
        first_window = windows[0]
        print(f"\nFirst Window:")
        print(f"  Window ID: {first_window['window_id']}")
        print(f"  Date Range: {first_window['start_date']} to {first_window['end_date']}")
        print(f"  Stocks: {len(first_window['data'])}")

        # Show last window details
        last_window = windows[-1]
        print(f"\nLast Window:")
        print(f"  Window ID: {last_window['window_id']}")
        print(f"  Date Range: {last_window['start_date']} to {last_window['end_date']}")
        print(f"  Stocks: {len(last_window['data'])}")

        # Show window distribution
        window_sizes = [len(window['data']) for window in windows]
        print(f"\nWindow Statistics:")
        print(f"  Avg stocks per window: {np.mean(window_sizes):.1f}")
        print(f"  Min stocks per window: {np.min(window_sizes)}")
        print(f"  Max stocks per window: {np.max(window_sizes)}")

    return windows


def test_data_summary(processed_data):
    """Test data summary logging"""
    print("\n" + "="*50)
    print("TEST: Data Summary")
    print("="*50)

    log_data_summary(processed_data)

    # Show feature statistics
    test_symbol = list(processed_data.keys())[0]
    test_data = processed_data[test_symbol]

    print(f"\nFeature Statistics for {test_symbol}:")
    print(f"  Total features: {len(test_data.columns)}")
    print(f"  Total rows: {len(test_data)}")

    # Show some key features
    key_features = ['RV_5', 'RV_20', 'MA_20', 'RSI', 'MACD', 'Vol_Regime']
    available_key_features = [f for f in key_features if f in test_data.columns]

    print(f"\nKey Features (first 5 rows):")
    print(test_data[available_key_features].head())


def main():
    """Main test execution"""
    print("\n" + "="*50)
    print("VIETNAMESE STOCK DATA PREPROCESSING TESTS")
    print("="*50)

    try:
        # Test 1: Data Loading
        stock_data = test_data_loading()

        if not stock_data:
            print("ERROR: No stock data loaded. Exiting tests.")
            return

        # Test 2: Feature Engineering
        processed_data = test_feature_engineering(stock_data)

        if not processed_data:
            print("ERROR: Feature engineering failed. Exiting tests.")
            return

        # Test 3: Data Validation
        validated_data = test_data_validation(processed_data)

        # Test 4: Incremental Windows
        windows = test_incremental_windows(processed_data)

        # Test 5: Data Summary
        test_data_summary(processed_data)

        print("\n" + "="*50)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("="*50)

    except Exception as e:
        print(f"\nERROR: Test execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
