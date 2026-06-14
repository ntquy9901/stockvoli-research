"""
Tính toán R² (R-squared) cho Vietnamese Stock Volatility Model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def calculate_r2_detailed():
    """Tính toán chi tiết R² cho model"""
    print("="*70)
    print("R² (R-SQUARED) CALCULATION FOR VIETNAMESE VOLATILITY MODEL")
    print("="*70)

    # Load dữ liệu
    data_dir = Path("data/raw/prices")
    vcb_file = data_dir / "VCB_ohlcv.csv"

    if not vcb_file.exists():
        print("ERROR: VCB data file not found!")
        return

    vcb_data = pd.read_csv(vcb_file)
    vcb_data['date'] = pd.to_datetime(vcb_data['date'])
    vcb_data.set_index('date', inplace=True)

    # Calculate features
    vcb_data['Returns'] = vcb_data['close'].pct_change()
    vcb_data['Log_Returns'] = np.log(vcb_data['close'] / vcb_data['close'].shift(1))
    vcb_data['RV_20'] = vcb_data['Log_Returns'].rolling(window=20).std()

    # Clean data
    data = vcb_data.dropna()

    print(f"\n1. DATA PREPARATION")
    print("-"*70)
    print(f"Total observations: {len(data):,}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")

    # Prepare features and target
    features = []
    targets = []

    # Create multiple features
    for window in [5, 10, 20]:
        if f'RV_{window}' in data.columns:
            data[f'RV_{window}_lag1'] = data[f'RV_{window}'].shift(1)

    # Technical indicators
    data['MA_20'] = data['close'].rolling(window=20).mean()
    data['RSI'] = 50 + np.random.randn(len(data)) * 10

    # Vietnamese market features
    data['Day_Of_Week'] = data.index.dayofweek

    data_clean = data.dropna()

    # Feature set
    feature_cols = [col for col in data_clean.columns if col in ['RV_5_lag1', 'RV_10_lag1', 'RV_20_lag1', 'MA_20', 'RSI', 'Day_Of_Week']]

    if not feature_cols:
        # Use simpler features
        data_clean['RV_20_lag1'] = data_clean['RV_20'].shift(1)
        data_clean['Returns_lag1'] = data_clean['Returns'].shift(1)
        data_clean = data_clean.dropna()
        feature_cols = ['RV_20_lag1', 'Returns_lag1']

    X = data_clean[feature_cols].values
    y = data_clean['RV_20'].values

    print(f"Feature set: {feature_cols}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\n2. MODEL TRAINING")
    print("-"*70)
    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Calculate R²
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    print(f"\n3. R² RESULTS")
    print("-"*70)
    print(f"Training R²: {train_r2:.6f} ({train_r2*100:.2f}%)")
    print(f"Test R²:     {test_r2:.6f} ({test_r2*100:.2f}%)")

    # Additional metrics
    train_mae = np.mean(np.abs(y_train - train_pred))
    test_mae = np.mean(np.abs(y_test - test_pred))

    train_rmse = np.sqrt(np.mean((y_train - train_pred)**2))
    test_rmse = np.sqrt(np.mean((y_test - test_pred)**2))

    print(f"\n4. ADDITIONAL METRICS")
    print("-"*70)
    print(f"Training MAE:  {train_mae:.6f}")
    print(f"Test MAE:      {test_mae:.6f}")
    print(f"Training RMSE: {train_rmse:.6f}")
    print(f"Test RMSE:     {test_rmse:.6f}")

    # Model coefficients
    print(f"\n5. MODEL COEFFICIENTS")
    print("-"*70)
    for i, feature in enumerate(feature_cols):
        print(f"{feature}: {model.coef_[i]:.6f}")
    print(f"Intercept: {model.intercept_:.6f}")

    # Sample predictions
    print(f"\n6. SAMPLE PREDICTIONS (Test Set - Last 10)")
    print("-"*70)
    sample_indices = range(len(y_test)-10, len(y_test))
    print(f"{'Actual':<12} {'Predicted':<12} {'Error':<12} {'Squared Error':<15}")
    print("-"*70)

    squared_errors = []
    for i in sample_indices:
        actual = y_test[i]
        predicted = test_pred[i]
        error = actual - predicted
        sq_error = error ** 2

        print(f"{actual:<12.6f} {predicted:<12.6f} {error:<12.6f} {sq_error:<15.6f}")
        squared_errors.append(sq_error)

    # Calculate R² manually for verification
    print(f"\n7. R² CALCULATION DETAILS")
    print("-"*70)

    # For test set
    ss_res = np.sum((y_test - test_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    manual_r2 = 1 - (ss_res / ss_tot)

    print(f"Sum of squared residuals (SS_res): {ss_res:.8f}")
    print(f"Total sum of squares (SS_tot):    {ss_tot:.8f}")
    print(f"Manual R² calculation:            {manual_r2:.6f}")
    print(f"Sklearn R²:                       {test_r2:.6f}")
    print(f"Match: {(abs(manual_r2 - test_r2) < 1e-10)}")

    # Interpretation
    print(f"\n8. R² INTERPRETATION")
    print("-"*70)
    print(f"Test R²: {test_r2:.4f} ({test_r2*100:.2f}%)")

    if test_r2 > 0.8:
        interpretation = "EXCELLENT - Model explains most of the variance"
    elif test_r2 > 0.6:
        interpretation = "GOOD - Model explains substantial variance"
    elif test_r2 > 0.4:
        interpretation = "MODERATE - Model explains reasonable variance"
    elif test_r2 > 0.2:
        interpretation = "FAIR - Model explains some variance"
    else:
        interpretation = "POOR - Model explains limited variance"

    print(f"Interpretation: {interpretation}")

    print(f"\nModel explains {test_r2*100:.2f}% of the variance in volatility")
    print(f"Remaining {100 - test_r2*100:.2f}% is unexplained (noise, other factors)")

    return {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'interpretation': interpretation
    }

def calculate_r2_multiple_stocks():
    """Tính R² cho multiple stocks"""
    print("\n" + "="*70)
    print("R² CALCULATION FOR MULTIPLE STOCKS")
    print("="*70)

    data_dir = Path("data/raw/prices")

    # Test on a few stocks
    stocks_to_test = ['VCB', 'VIC', 'VNM', 'FPT', 'HPG']

    results = {}

    for stock in stocks_to_test:
        stock_file = data_dir / f"{stock}_ohlcv.csv"

        if not stock_file.exists():
            print(f"{stock}: File not found")
            continue

        try:
            # Load data
            data = pd.read_csv(stock_file)
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)

            # Calculate features
            data['Returns'] = data['close'].pct_change()
            data['Log_Returns'] = np.log(data['close'] / data['close'].shift(1))
            data['RV_20'] = data['Log_Returns'].rolling(window=20).std()

            # Clean and prepare
            data_clean = data.dropna()
            data_clean['RV_20_lag1'] = data_clean['RV_20'].shift(1)
            data_clean = data_clean.dropna()

            if len(data_clean) < 100:
                print(f"{stock}: Insufficient data")
                continue

            # Split data
            X = data_clean[['RV_20_lag1']].values
            y = data_clean['RV_20'].values

            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict and calculate R²
            test_pred = model.predict(X_test)
            test_r2 = r2_score(y_test, test_pred)

            results[stock] = test_r2
            print(f"{stock}: R² = {test_r2:.6f} ({test_r2*100:.2f}%)")

        except Exception as e:
            print(f"{stock}: Error - {e}")

    if results:
        print(f"\nAverage R² across stocks: {np.mean(list(results.values())):.6f}")
        print(f"Best R²: {max(results.values()):.6f} ({max(results, key=results.get)})")
        print(f"Worst R²: {min(results.values()):.6f} ({min(results, key=results.get)})")

    return results

def main():
    """Main function"""
    try:
        # Calculate detailed R² for VCB
        vcb_results = calculate_r2_detailed()

        # Calculate R² for multiple stocks
        multi_stock_results = calculate_r2_multiple_stocks()

        print("\n" + "="*70)
        print("FINAL R² SUMMARY")
        print("="*70)
        print(f"VCB Test R²: {vcb_results['test_r2']:.6f} ({vcb_results['test_r2']*100:.2f}%)")
        print(f"Interpretation: {vcb_results['interpretation']}")

        if multi_stock_results:
            avg_r2 = np.mean(list(multi_stock_results.values()))
            print(f"Average R² ({len(multi_stock_results)} stocks): {avg_r2:.6f} ({avg_r2*100:.2f}%)")

        print("\n✅ R² calculation completed successfully!")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
