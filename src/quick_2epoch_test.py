"""
Quick 2-Epoch Feature Comparison Test
Fixed version that properly overrides epoch configuration
"""

import sys
import logging
import json
from pathlib import Path
from datetime import datetime
import time
import yaml

# Import your existing training classes
from vn30_dataset import create_vn30_dataloaders
from model_training_fixed import TimesFMVN30Finetuner

# Setup logging
Path('experiments').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments/quick_comparison.log'),
        logging.StreamHandler()
    ]
)

class QuickTimesFMTester(TimesFMVN30Finetuner):
    """
    Modified TimesFM tester that respects custom epoch counts

    This class overrides the config to use a smaller number of epochs
    for quick testing purposes.
    """

    def __init__(self, custom_epochs=2, config_path='configs/config.yaml'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Override epochs for quick testing
        self.config['training']['num_epochs'] = custom_epochs

        # Initialize parent with our modified config
        super().__init__(config_path)

def test_single_feature_fixed(feature_type: str = 'RV_20', epochs: int = 2):
    """
    Test TimesFM training with a single feature (FIXED VERSION)

    Args:
        feature_type: Feature to test ('RV_20', 'overnight', 'parkinson', 'gk')
        epochs: Number of training epochs (keep small for quick test)

    Returns:
        Dictionary with training results
    """
    logging.info("=" * 70)
    logging.info(f"[TESTING] {feature_type} Feature - {epochs} Epochs")
    logging.info("=" * 70)

    start_time = time.time()

    try:
        # Step 1: Create dataloaders for this feature
        logging.info(f"Creating dataloaders for {feature_type}...")
        train_loader, test_loader = create_vn30_dataloaders(
            config_path='configs/config.yaml',
            feature_type=feature_type
        )

        # Step 2: Initialize model with custom epochs
        logging.info(f"Initializing TimesFM model ({epochs} epochs)...")
        finetuner = QuickTimesFMTester(custom_epochs=epochs)
        finetuner.load_timesfm_model()
        finetuner.setup_lora_adapters()
        finetuner.setup_optimizer()

        # Create scheduler with correct step count
        num_training_steps = epochs * len(train_loader)
        finetuner.setup_scheduler(num_training_steps)

        # Step 3: Quick training run
        logging.info(f"Starting training ({epochs} epochs)...")
        logging.info(f"Expected time: ~{epochs * 3.7:.1f} hours")
        finetuner.train_model(train_loader, test_loader)

        # Step 4: Record results
        end_time = time.time()
        training_time = end_time - start_time

        results = {
            'feature_type': feature_type,
            'status': 'success',
            'best_val_loss': finetuner.best_val_loss,
            'training_time_minutes': training_time / 60,
            'epochs_completed': len(finetuner.training_history),
            'training_samples': len(train_loader.dataset),
            'test_samples': len(test_loader.dataset),
            'timestamp': datetime.now().isoformat()
        }

        logging.info(f"[OK] {feature_type} training complete!")
        logging.info(f"Best validation loss: {results['best_val_loss']:.6f}")
        logging.info(f"Training time: {results['training_time_minutes']:.1f} minutes")

        return results

    except Exception as e:
        end_time = time.time()
        error_results = {
            'feature_type': feature_type,
            'status': 'failed',
            'error': str(e),
            'training_time_minutes': (end_time - start_time) / 60,
            'timestamp': datetime.now().isoformat()
        }
        logging.error(f"[FAIL] {feature_type} failed: {e}")
        return error_results

def compare_features_fixed(feature_list: list = ['RV_20', 'overnight'], epochs: int = 2):
    """
    Compare multiple features against each other (FIXED VERSION)

    Args:
        feature_list: List of features to test
        epochs: Number of training epochs per feature

    Returns:
        Dictionary with comparison results
    """
    logging.info("=" * 70)
    logging.info(f"[FIXED FEATURE COMPARISON] Testing {len(feature_list)} features")
    logging.info(f"Each feature: {epochs} epochs (~{epochs * 3.7:.1f} hours per feature)")
    logging.info("=" * 70)

    all_results = {}

    for i, feature_type in enumerate(feature_list):
        logging.info(f"\n[{i+1}/{len(feature_list)}] Testing {feature_type}...")

        result = test_single_feature_fixed(feature_type, epochs)
        all_results[feature_type] = result

        # Small pause between runs (cleanup GPU memory)
        if i < len(feature_list) - 1:
            logging.info("Pausing 2 minutes for GPU cleanup...")
            time.sleep(120)

    # Generate comparison report
    logging.info("\n" + "=" * 70)
    logging.info("[COMPARISON RESULTS]")
    logging.info("=" * 70)

    # Find baseline (RV_20)
    baseline_result = all_results.get('RV_20', {})
    baseline_loss = baseline_result.get('best_val_loss', float('inf'))

    for feature_type, result in all_results.items():
        if result['status'] == 'success':
            loss = result['best_val_loss']
            if feature_type != 'RV_20' and baseline_loss != float('inf'):
                improvement = (baseline_loss - loss) / baseline_loss * 100
                logging.info(f"{feature_type}: {loss:.6f} ({improvement:+.1f}% vs baseline)")
            else:
                logging.info(f"{feature_type}: {loss:.6f} (baseline)")
        else:
            logging.info(f"{feature_type}: FAILED - {result.get('error', 'Unknown error')}")

    # Save results
    experiments_dir = Path('experiments')
    comparison_file = experiments_dir / 'feature_comparison_results_fixed.json'

    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logging.info(f"\n[OK] Comparison results saved: {comparison_file}")

    return all_results

def main():
    """Main execution function"""
    logging.info("Starting FIXED TimesFM Feature Comparison Test")
    logging.info("This will test RV_20 (baseline) vs overnight volatility")
    logging.info("FIXED: Properly uses 2 epochs per feature")

    # Quick comparison test (2 epochs each for speed)
    results = compare_features_fixed(
        feature_list=['RV_20', 'overnight'],
        epochs=2  # Exactly 2 epochs!
    )

    # Check if overnight volatility is better
    if 'RV_20' in results and 'overnight' in results:
        if results['RV_20']['status'] == 'success' and results['overnight']['status'] == 'success':
            rv20_loss = results['RV_20']['best_val_loss']
            overnight_loss = results['overnight']['best_val_loss']
            improvement = (rv20_loss - overnight_loss) / rv20_loss * 100

            logging.info("\n" + "=" * 70)
            logging.info("[FINAL RESULT]")
            logging.info("=" * 70)
            logging.info(f"RV_20 baseline loss: {rv20_loss:.6f}")
            logging.info(f"Overnight volatility loss: {overnight_loss:.6f}")
            logging.info(f"Improvement: {improvement:+.1f}%")

            if improvement > 0:
                logging.info("✅ OVERNIGHT VOLATILITY IS BETTER!")
            else:
                logging.info("❌ BASELINE RV_20 IS STILL BETTER")

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logging.warning("\n[INTERRUPTED] Training stopped by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)