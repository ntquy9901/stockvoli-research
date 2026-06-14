"""
Test TimesFM incremental learning module
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.timesfm_incremental import (
    TimesFMDataset,
    TimesFMIncrementalLearner,
    run_incremental_learning
)
import logging

logging.basicConfig(level=logging.INFO)

def create_test_data():
    """Create synthetic test data for testing"""
    np.random.seed(42)

    # Create sample data for 3 stocks over 1000 days
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')

    test_data = {}

    for symbol in ['VCB', 'VIC', 'VNM']:
        df = pd.DataFrame(index=dates)

        # Basic OHLCV data
        df['close'] = 100 + np.cumsum(np.random.randn(1000) * 0.02)
        df['open'] = df['close'].shift(1) * (1 + np.random.randn(1000) * 0.001)
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + abs(np.random.randn(1000) * 0.005))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - abs(np.random.randn(1000) * 0.005))
        df['volume'] = 1000000 + np.random.randint(0, 500000, 1000)

        # Calculate returns
        df['Returns'] = df['close'].pct_change()
        df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1))

        # Create volatility features
        df['RV_5'] = df['Log_Returns'].rolling(window=5).std()
        df['RV_10'] = df['Log_Returns'].rolling(window=10).std()
        df['RV_20'] = df['Log_Returns'].rolling(window=20).std()
        df['RV_30'] = df['Log_Returns'].rolling(window=30).std()

        # Technical indicators
        df['MA_10'] = df['close'].rolling(window=10).mean()
        df['MA_20'] = df['close'].rolling(window=20).mean()
        df['RSI'] = 50 + np.random.randn(1000) * 10

        # Vietnamese features
        df['Day_Of_Week'] = df.index.dayofweek
        df['Month_Start'] = (df.index.day <= 5).astype(int)

        test_data[symbol] = df.dropna()

    return test_data


def test_dataset_creation():
    """Test TimesFM dataset creation"""
    print("\n" + "="*50)
    print("TEST: TimesFM Dataset Creation")
    print("="*50)

    # Create test data
    test_data = create_test_data()

    # Define feature columns
    feature_cols = ['RV_5', 'RV_10', 'RV_20', 'RV_30', 'MA_10', 'MA_20', 'RSI', 'Day_Of_Week', 'Month_Start']

    # Create dataset
    try:
        dataset = TimesFMDataset(
            test_data,
            feature_cols,
            target_col='RV_20',
            context_length=512
        )

        print(f"✓ Dataset created successfully")
        print(f"  Total samples: {len(dataset)}")

        if len(dataset) > 0:
            # Test sample retrieval
            sample = dataset[0]
            print(f"\n  Sample structure:")
            print(f"    Context shape: {sample['context'].shape}")
            print(f"    Target value: {sample['target']:.6f}")
            print(f"    Symbol: {sample['symbol']}")
            print(f"    Target date: {sample['target_date']}")

            # Test batch creation
            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

            batch = next(iter(dataloader))
            print(f"\n  Batch structure:")
            print(f"    Context shape: {batch['context'].shape}")
            print(f"    Target shape: {batch['target'].shape}")

        return dataset, test_data

    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_learner_initialization():
    """Test TimesFM learner initialization"""
    print("\n" + "="*50)
    print("TEST: TimesFM Learner Initialization")
    print("="*50)

    try:
        learner = TimesFMIncrementalLearner()

        print(f"✓ Learner initialized successfully")
        print(f"  Device: {learner.device}")
        print(f"  Context length: {learner.context_length}")
        print(f"  Window size: {learner.window_size}")
        print(f"  Batch size: {learner.batch_size}")
        print(f"  Learning rate: {learner.learning_rate}")

        # Test model loading
        print("\n  Loading base model...")
        learner.load_base_model()

        print(f"  ✓ Model loaded successfully")
        print(f"  Model type: {type(learner.model).__name__}")

        return learner

    except Exception as e:
        print(f"✗ Learner initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_incremental_update(learner, test_data):
    """Test incremental update process"""
    print("\n" + "="*50)
    print("TEST: Incremental Update Process")
    print("="*50)

    if learner is None or test_data is None:
        print("ERROR: Invalid inputs")
        return

    try:
        # Create a single window with test data
        window_data = test_data

        print(f"Running incremental update on test window...")
        print(f"  Stocks: {len(window_data)}")

        # Run incremental update
        metrics = learner.incremental_update(window_data, window_id=1)

        if 'error' in metrics:
            print(f"✗ Incremental update failed: {metrics['error']}")
            return None

        print(f"✓ Incremental update completed successfully")
        print(f"  Results:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"    {key}: {value:.6f}" if isinstance(value, float) else f"    {key}: {value}")

        return metrics

    except Exception as e:
        print(f"✗ Incremental update failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_prediction(learner):
    """Test prediction functionality"""
    print("\n" + "="*50)
    print("TEST: Prediction Functionality")
    print("="*50)

    if learner is None:
        print("ERROR: Invalid learner")
        return

    try:
        # Create sample context data
        context_length = 512
        n_features = 9  # Based on our feature set

        context_data = np.random.randn(context_length, n_features).astype(np.float32)

        print(f"Making prediction on context data...")
        print(f"  Context shape: {context_data.shape}")

        # Make prediction
        prediction = learner.predict(context_data)

        print(f"✓ Prediction completed successfully")
        print(f"  Predicted volatility: {prediction:.6f}")

        return prediction

    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multiple_updates(learner, test_data):
    """Test multiple incremental updates"""
    print("\n" + "="*50)
    print("TEST: Multiple Incremental Updates")
    print("="*50)

    if learner is None or test_data is None:
        print("ERROR: Invalid inputs")
        return

    try:
        # Simulate multiple windows
        results = []

        for i in range(3):
            # Create slightly different data for each window
            window_data = {}
            for symbol, data in test_data.items():
                # Use different portions of data
                start_idx = i * 100
                end_idx = start_idx + 600
                if end_idx <= len(data):
                    window_data[symbol] = data.iloc[start_idx:end_idx].copy()

            if window_data:
                print(f"\nWindow {i+1}:")
                print(f"  Date range: {window_data['VCB'].index.min()} to {window_data['VCB'].index.max()}")

                metrics = learner.incremental_update(window_data, window_id=i+1)

                if 'error' not in metrics:
                    results.append(metrics)
                    print(f"  ✓ Window {i+1} completed - Loss: {metrics['loss']:.6f}")
                else:
                    print(f"  ✗ Window {i+1} failed: {metrics['error']}")

        print(f"\n✓ Multiple updates completed")
        print(f"  Successful windows: {len(results)}/3")
        print(f"  Best loss: {learner.best_loss:.6f}")

        return results

    except Exception as e:
        print(f"✗ Multiple updates failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main test execution"""
    print("\n" + "="*50)
    print("TIMESFM INCREMENTAL LEARNING TESTS")
    print("="*50)

    try:
        # Test 1: Dataset Creation
        dataset, test_data = test_dataset_creation()

        if dataset is None:
            print("ERROR: Dataset creation failed. Exiting tests.")
            return

        # Test 2: Learner Initialization
        learner = test_learner_initialization()

        if learner is None:
            print("ERROR: Learner initialization failed. Exiting tests.")
            return

        # Test 3: Single Incremental Update
        metrics = test_incremental_update(learner, test_data)

        # Test 4: Prediction
        prediction = test_prediction(learner)

        # Test 5: Multiple Updates
        multiple_results = test_multiple_updates(learner, test_data)

        print("\n" + "="*50)
        print("ALL TESTS COMPLETED! ✓")
        print("="*50)

        # Summary
        print("\nSUMMARY:")
        print(f"  Dataset samples: {len(dataset)}")
        print(f"  Model loaded: ✓")
        print(f"  Single update: {'✓' if metrics else '✗'}")
        print(f"  Prediction: {'✓' if prediction else '✗'}")
        print(f"  Multiple updates: {'✓' if multiple_results else '✗'}")

    except Exception as e:
        print(f"\nERROR: Test execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
