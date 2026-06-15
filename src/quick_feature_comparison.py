"""
Quick Feature Comparison for TimesFM VN30
Tests overnight volatility against baseline RV_20
"""

import sys
import logging
import yaml
import json
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments/feature_comparison.log'),
        logging.StreamHandler()
    ]
)

def quick_feature_test():
    """
    Quick test comparing RV_20 baseline vs overnight volatility

    This is a simplified test that:
    1. Checks data availability
    2. Creates summary statistics
    3. Prepares for model training comparison
    """
    logging.info("=" * 70)
    logging.info("[QUICK FEATURE COMPARISON TEST]")
    logging.info("=" * 70)

    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load processed data
    processed_dir = Path(config['data']['processed_path'])
    processed_files = list(processed_dir.glob("*_processed.csv"))

    logging.info(f"Found {len(processed_files)} processed stock files")

    # Analyze feature statistics
    features_to_analyze = ['RV_20', 'overnight', 'parkinson', 'gk', 'close_to_close']
    feature_stats = {feature: [] for feature in features_to_analyze}

    for file_path in processed_files:
        stock_name = file_path.stem.replace('_processed', '')

        try:
            import pandas as pd
            df = pd.read_csv(file_path)

            for feature in features_to_analyze:
                if feature in df.columns:
                    # Get basic statistics
                    feature_data = df[feature].dropna()
                    if len(feature_data) > 0:
                        feature_stats[feature].append({
                            'stock': stock_name,
                            'mean': float(feature_data.mean()),
                            'std': float(feature_data.std()),
                            'min': float(feature_data.min()),
                            'max': float(feature_data.max()),
                            'count': len(feature_data)
                        })

        except Exception as e:
            logging.warning(f"Error processing {stock_name}: {e}")
            continue

    # Generate summary report
    logging.info("\n" + "=" * 70)
    logging.info("[FEATURE STATISTICS SUMMARY]")
    logging.info("=" * 70)

    summary_report = {
        'timestamp': datetime.now().isoformat(),
        'total_stocks': len(processed_files),
        'features': {}
    }

    for feature in features_to_analyze:
        stats_list = feature_stats[feature]
        if stats_list:
            means = [s['mean'] for s in stats_list]
            stds = [s['std'] for s in stats_list]
            counts = [s['count'] for s in stats_list]

            feature_summary = {
                'stocks_available': len(stats_list),
                'overall_mean': float(sum(means) / len(means)),
                'overall_std': float(sum(stds) / len(stds)),
                'total_observations': sum(counts),
                'average_per_stock': float(sum(counts) / len(counts))
            }

            summary_report['features'][feature] = feature_summary

            logging.info(f"\n{feature}:")
            logging.info(f"  Available for: {feature_summary['stocks_available']} stocks")
            logging.info(f"  Average mean: {feature_summary['overall_mean']:.6f}")
            logging.info(f"  Average std: {feature_summary['overall_std']:.6f}")
            logging.info(f"  Total observations: {feature_summary['total_observations']:,}")

    # Save summary report
    experiments_dir = Path('experiments')
    experiments_dir.mkdir(exist_ok=True)

    with open(experiments_dir / 'feature_comparison_summary.json', 'w') as f:
        json.dump(summary_report, f, indent=2)

    logging.info("\n" + "=" * 70)
    logging.info("[COMPARISON READINESS CHECK]")
    logging.info("=" * 70)

    # Check which features are ready for training
    ready_features = []
    for feature in features_to_analyze:
        if len(feature_stats[feature]) >= 20:  # Need at least 20 stocks
            ready_features.append(feature)
            logging.info(f"✅ {feature}: READY ({len(feature_stats[feature])} stocks)")
        else:
            logging.info(f"❌ {feature}: INSUFFICIENT ({len(feature_stats[feature])} stocks)")

    logging.info(f"\n[READY] {len(ready_features)} features ready for TimesFM training:")
    for feature in ready_features:
        logging.info(f"  - {feature}")

    # Comparison plan
    logging.info("\n" + "=" * 70)
    logging.info("[RECOMMENDED COMPARISON PLAN]")
    logging.info("=" * 70)

    if 'RV_20' in ready_features and 'overnight' in ready_features:
        logging.info("✅ READY: Compare RV_20 (baseline) vs overnight volatility")
        logging.info("   Next step: Run feature comparison training")
    else:
        logging.info("❌ WAIT: Need more data for comparison")

    logging.info("\n" + "=" * 70)
    logging.info("[QUICK TEST COMPLETE]")
    logging.info("=" * 70)
    logging.info(f"Summary saved: experiments/feature_comparison_summary.json")
    logging.info("Next step: Create comparison training script")

    return ready_features

if __name__ == "__main__":
    ready_features = quick_feature_test()

    if 'RV_20' in ready_features and 'overnight' in ready_features:
        logging.info("\n🚀 SYSTEM READY FOR OVERNIGHT VOLATILITY TEST!")
        logging.info("Recommend: Proceed with TimesFM training comparison")
        sys.exit(0)
    else:
        logging.warning("\n⚠️ NEED MORE DATA FOR COMPREHENSIVE TESTING")
        sys.exit(1)