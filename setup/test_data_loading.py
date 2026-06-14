"""
Vietnamese Stock Data Validation Script
Checks data quality and availability for 30 VN30 stocks
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def validate_single_stock(file_path, stock_name):
    """Validate a single stock's data file"""
    try:
        df = pd.read_csv(file_path)

        # Check required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            return False, f"Missing columns: {missing_cols}"

        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])

        # Check for empty data
        if len(df) == 0:
            return False, "Empty dataframe"

        # Check for sufficient observations
        if len(df) < 100:
            return False, f"Insufficient data: {len(df)} observations"

        # Check for missing values
        missing_values = df[required_cols].isnull().sum()
        if missing_values.any():
            return False, f"Missing values detected: {missing_values[missing_values > 0].to_dict()}"

        # Check for valid price data
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] <= 0).any():
                return False, f"Invalid prices in {col} (must be > 0)"

        # Check for valid volume
        if (df['volume'] < 0).any():
            return False, "Invalid volume values (must be >= 0)"

        # Check date range
        date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
        observations = len(df)

        return True, {
            'observations': observations,
            'date_range': date_range,
            'missing_values': missing_values.sum(),
            'price_range': f"{df['close'].min():.2f} - {df['close'].max():.2f}"
        }

    except Exception as e:
        return False, f"Error reading file: {str(e)}"


def validate_all_stocks():
    """Validate all Vietnamese stock data"""
    print("=" * 70)
    print("🏗️  Vietnamese Stock Data Validation")
    print("=" * 70)

    data_path = Path("data/raw/prices")

    if not data_path.exists():
        print(f"❌ Data directory not found: {data_path}")
        print("   Create directory and add Vietnamese stock OHLCV data")
        return False

    # Find all stock files
    stock_files = list(data_path.glob("*_ohlcv.csv"))

    if not stock_files:
        print(f"❌ No stock files found in {data_path}")
        print("   Expected format: STOCK_ohlcv.csv (e.g., VCB_ohlcv.csv)")
        return False

    print(f"\n📋 Found {len(stock_files)} stock files")
    print("=" * 70)

    # Load configuration to check expected stocks
    try:
        import yaml
        config_path = Path("configs/config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            expected_stocks = config.get('data', {}).get('stocks', [])
            print(f"📋 Expected stocks: {len(expected_stocks)}")
        else:
            expected_stocks = []
            print("⚠️  No config file found - will validate available stocks")
    except:
        expected_stocks = []

    # Validate each stock
    results = {}
    valid_stocks = []

    for i, file_path in enumerate(stock_files, 1):
        stock_name = file_path.stem.replace('_ohlcv', '').upper()
        print(f"\n[{i}/{len(stock_files)}] Validating {stock_name}...")

        is_valid, result = validate_single_stock(file_path, stock_name)

        if is_valid:
            print(f"✅ {stock_name}: VALID")
            print(f"   Observations: {result['observations']:,}")
            print(f"   Date range: {result['date_range']}")
            print(f"   Price range: {result['price_range']}")
            results[stock_name] = {'status': 'valid', 'details': result}
            valid_stocks.append(stock_name)
        else:
            print(f"❌ {stock_name}: INVALID - {result}")
            results[stock_name] = {'status': 'invalid', 'error': result}

    # Summary
    print("\n" + "=" * 70)
    print("📊 DATA VALIDATION SUMMARY")
    print("=" * 70)

    total = len(stock_files)
    valid = len(valid_stocks)
    invalid = total - valid

    print(f"Total files: {total}")
    print(f"✅ Valid: {valid}")
    print(f"❌ Invalid: {invalid}")

    if expected_stocks:
        missing = set(expected_stocks) - set(valid_stocks)
        print(f"⚠️  Missing from expected: {len(missing)}")
        if missing:
            print(f"   Missing stocks: {sorted(missing)}")

    # Data quality checks
    print("\n📋 Data Quality Analysis:")

    if valid_stocks:
        # Get total observations
        total_obs = sum(results[s]['details']['observations'] for s in valid_stocks)
        print(f"✅ Total observations: {total_obs:,}")

        # Check date ranges
        date_ranges = [results[s]['details']['date_range'] for s in valid_stocks]
        print(f"✅ Date ranges cover: {min(date_ranges)} to {max(date_ranges)}")

        # Minimum observations check
        min_obs = min(results[s]['details']['observations'] for s in valid_stocks)
        if min_obs >= 100:
            print(f"✅ All stocks have >= 100 observations")
        else:
            print(f"⚠️  Some stocks have < 100 observations (min: {min_obs})")

    # Success criteria
    print("\n🎯 Validation Criteria:")

    success = True

    if valid < 10:
        print("❌ Need at least 10 valid stocks for multi-stock dataset")
        success = False
    else:
        print(f"✅ Sufficient stocks for multi-stock dataset ({valid} >= 10)")

    if total_obs < 1000:
        print("❌ Need at least 1,000 total observations")
        success = False
    else:
        print(f"✅ Sufficient observations ({total_obs:,} >= 1,000)")

    if expected_stocks and len(set(expected_stocks) & set(valid_stocks)) < len(expected_stocks) * 0.5:
        print("⚠️  Less than 50% of expected stocks found")
        success = False
    else:
        print("✅ Good coverage of expected stocks")

    # Save validation results
    print("\n📋 Saving validation results...")
    try:
        results_dir = Path("experiments")
        results_dir.mkdir(exist_ok=True)

        import json
        validation_report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_files': total,
            'valid_stocks': valid,
            'invalid_stocks': invalid,
            'total_observations': sum(results[s]['details']['observations'] for s in valid_stocks) if valid_stocks else 0,
            'results': results
        }

        with open(results_dir / 'data_validation_report.json', 'w') as f:
            json.dump(validation_report, f, indent=2)

        print(f"✅ Validation report saved to experiments/data_validation_report.json")

    except Exception as e:
        print(f"⚠️  Could not save validation report: {e}")

    # Final verdict
    print("\n" + "=" * 70)
    if success:
        print("🎉 DATA VALIDATION SUCCESSFUL!")
        print("=" * 70)
        print(f"✅ {valid} valid stocks with {total_obs:,} total observations")
        print("✅ Data ready for TimesFM fine-tuning")
        return True
    else:
        print("❌ DATA VALIDATION FAILED")
        print("=" * 70)
        print("Please fix data quality issues before proceeding")
        return False


def main():
    """Main function"""
    try:
        success = validate_all_stocks()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())