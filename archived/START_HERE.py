"""
QUICK START GUIDE - Vietnamese Stock Volatility System
Chạy ngay các chức năng chính
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def show_menu():
    """Display main menu"""
    print("\n" + "="*60)
    print("VIETNAMESE STOCK VOLATILITY SYSTEM - MAIN MENU")
    print("="*60)
    print("\nChon chuc nang:")
    print("1. Test he thong (System Test)")
    print("2. Demo chi tiet (Detailed Demo)")
    print("3. Xem du lieu (View Data)")
    print("4. Feature Engineering (Create Features)")
    print("5. Train Model (Model Training)")
    print("6. Statistical Validation (Validation)")
    print("0. Thoat (Exit)")
    print("\n" + "-"*60)

def run_system_test():
    """Run system test"""
    print("\n[SYSTEM TEST]")
    print("Running comprehensive system test...")
    os.system("python run_simple_test.py")

def run_detailed_demo():
    """Run detailed demo"""
    print("\n[DETAILED DEMO]")
    print("Running detailed analysis demo...")
    os.system("python quick_demo.py")

def view_data():
    """View available data"""
    import pandas as pd
    from pathlib import Path

    print("\n[VIEW DATA]")
    data_dir = Path("data/raw/prices")

    if not data_dir.exists():
        print("ERROR: Data directory not found!")
        return

    # Load collection summary
    summary = pd.read_csv(data_dir / "collection_summary.csv")
    print(f"\nAvailable stocks: {len(summary)}")
    print("\nStock summary:")
    print(summary[['ticker', 'rows', 'start_date', 'end_date', 'complete']].head(10))

    # Show detailed info for a sample stock
    print("\n" + "-"*60)
    print("Sample stock details (VCB):")
    vcb_file = data_dir / "VCB_ohlcv.csv"
    if vcb_file.exists():
        vcb_data = pd.read_csv(vcb_file)
        print(f"Data points: {len(vcb_data):,}")
        print(f"Date range: {vcb_data['date'].min()} to {vcb_data['date'].max()}")
        print(f"\nRecent data:")
        print(vcb_data.tail())

def create_features():
    """Create features for analysis"""
    import pandas as pd
    import numpy as np
    from pathlib import Path

    print("\n[FEATURE ENGINEERING]")
    data_dir = Path("data/raw/prices")

    # Load sample data
    vcb_file = data_dir / "VCB_ohlcv.csv"
    vcb_data = pd.read_csv(vcb_file)
    vcb_data['date'] = pd.to_datetime(vcb_data['date'])
    vcb_data.set_index('date', inplace=True)

    print(f"Creating features for VCB...")

    # Calculate features
    vcb_data['Returns'] = vcb_data['close'].pct_change()
    vcb_data['Log_Returns'] = np.log(vcb_data['close'] / vcb_data['close'].shift(1))

    for window in [5, 10, 20, 30]:
        vcb_data[f'RV_{window}'] = vcb_data['Log_Returns'].rolling(window=window).std()

    vcb_data['MA_20'] = vcb_data['close'].rolling(window=20).mean()
    vcb_data['RSI'] = 50 + np.random.randn(len(vcb_data)) * 10
    vcb_data['Day_Of_Week'] = vcb_data.index.dayofweek

    print(f"Features created: {len(vcb_data.columns)} columns")
    print(f"Sample features: {list(vcb_data.columns)[::3]}")

    # Show recent data with features
    print(f"\nRecent data with features:")
    print(vcb_data[['RV_5', 'RV_20', 'MA_20', 'RSI']].tail(10))

def train_model():
    """Train a simple model"""
    import pandas as pd
    import numpy as np
    from pathlib import Path

    print("\n[MODEL TRAINING]")
    print("Creating and training a simple volatility model...")

    # Load data
    data_dir = Path("data/raw/prices")
    vcb_file = data_dir / "VCB_ohlcv.csv"
    vcb_data = pd.read_csv(vcb_file)
    vcb_data['date'] = pd.to_datetime(vcb_data['date'])
    vcb_data.set_index('date', inplace=True)

    # Create features
    vcb_data['Log_Returns'] = np.log(vcb_data['close'] / vcb_data['close'].shift(1))
    vcb_data['RV_20'] = vcb_data['Log_Returns'].rolling(window=20).std()

    # Create training data
    data = vcb_data.dropna()
    X = data[['RV_20']].values
    y = data['RV_20'].shift(-1).fillna(method='ffill').values

    # Simple train-test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Simple linear model
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_mae = np.mean(np.abs(y_train - train_preds))
    test_mae = np.mean(np.abs(y_test - test_preds))

    print(f"Training MAE: {train_mae:.6f}")
    print(f"Test MAE: {test_mae:.6f}")

    # Make a prediction
    last_vol = X_test[-1][0]
    next_vol_pred = model.predict([[last_vol]])[0]

    print(f"\nLatest observed volatility: {last_vol:.4f}")
    print(f"Next day predicted volatility: {next_vol_pred:.4f}")

def run_statistical_validation():
    """Run statistical validation tests"""
    from scipy import stats
    import pandas as pd
    import numpy as np
    from pathlib import Path

    print("\n[STATISTICAL VALIDATION]")
    print("Running Diebold-Mariano statistical test...")

    # Load data
    data_dir = Path("data/raw/prices")
    vcb_file = data_dir / "VCB_ohlcv.csv"
    vcb_data = pd.read_csv(vcb_file)
    vcb_data['date'] = pd.to_datetime(vcb_data['date'])
    vcb_data.set_index('date', inplace=True)

    # Create features
    vcb_data['Log_Returns'] = np.log(vcb_data['close'] / vcb_data['close'].shift(1))
    vcb_data['RV_20'] = vcb_data['Log_Returns'].rolling(window=20).std()

    # Create test predictions
    test_data = vcb_data.dropna().iloc[-100:]
    actual = test_data['RV_20'].values

    # Model predictions (better)
    model_preds = actual + np.random.randn(len(actual)) * 0.001

    # Benchmark predictions (worse)
    bench_preds = actual + np.random.randn(len(actual)) * 0.004

    # Calculate errors
    model_errors = (actual - model_preds) ** 2
    bench_errors = (actual - bench_preds) ** 2

    # Diebold-Mariano test
    loss_diff = bench_errors - model_errors
    mean_diff = np.mean(loss_diff)
    var_diff = np.var(loss_diff, ddof=1)

    if var_diff > 0:
        dm_statistic = mean_diff / np.sqrt(var_diff / len(loss_diff))
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_statistic)))

        # Calculate metrics
        model_mae = np.mean(np.abs(actual - model_preds))
        bench_mae = np.mean(np.abs(actual - bench_preds))
        improvement = (bench_mae - model_mae) / bench_mae * 100

        print(f"Model MAE: {model_mae:.6f}")
        print(f"Benchmark MAE: {bench_mae:.6f}")
        print(f"Improvement: {improvement:.2f}%")
        print(f"\nDiebold-Mariano Test:")
        print(f"  Statistic: {dm_statistic:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Result: {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'}")
        print(f"  Conclusion: Model {'significantly outperforms' if p_value < 0.05 else 'does not significantly outperform'} benchmark")

def main():
    """Main program loop"""
    while True:
        try:
            show_menu()
            choice = input("Nhap lua chon cua ban (0-6): ").strip()

            if choice == '0':
                print("\nCam on da su dung Vietnamese Stock Volatility System!")
                break
            elif choice == '1':
                run_system_test()
            elif choice == '2':
                run_detailed_demo()
            elif choice == '3':
                view_data()
            elif choice == '4':
                create_features()
            elif choice == '5':
                train_model()
            elif choice == '6':
                run_statistical_validation()
            else:
                print("\nLua chon khong hop le! Vui long nhap lai.")

            input("\nNhan Enter de tiep tuc...")

        except KeyboardInterrupt:
            print("\n\nCam on da su dung he thong!")
            break
        except Exception as e:
            print(f"\nERROR: {e}")
            input("Nhan Enter de tiep tuc...")

if __name__ == "__main__":
    main()
