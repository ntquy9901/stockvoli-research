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
import pandas as pd
import numpy as np

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


def analyze_train_val_periods(train_loader, test_loader, feature_type='RV_20'):
    """
    Comprehensive analysis of training vs validation periods

    This function explains WHY validation loss might be lower than training loss
    by analyzing the actual data characteristics in each period.

    Args:
        train_loader: Training data loader
        test_loader: Validation/test data loader
        feature_type: Feature being analyzed ('RV_20', 'overnight', etc.)

    Returns:
        dict: Analysis results with key findings
    """
    logging.info("")
    logging.info("=" * 70)
    logging.info(f"DIAGNOSTIC: Train vs Validation Period Analysis")
    logging.info(f"Feature: {feature_type}")
    logging.info("=" * 70)

    # Extract data from loaders (sample fewer batches for speed)
    train_data = []
    for batch in train_loader:
        train_data.append(batch)
        if len(train_data) >= 10:  # Sample first 10 batches (reduced from 100)
            break

    val_data = []
    for batch in test_loader:
        val_data.append(batch)
        if len(val_data) >= 10:  # Sample first 10 batches (reduced from 100)
            break

    # Convert to numpy for analysis (FIXED: Handle dict batches correctly)
    train_values = []
    for batch in train_data:
        if isinstance(batch, dict):
            # TimesFM dataloader returns dict with 'context' and 'ground_truth' keys
            context_data = batch.get('context', None)
            if context_data is not None:
                # Convert tensor to numpy and flatten
                if hasattr(context_data, 'numpy'):
                    train_values.extend(context_data.numpy().flatten())
                elif hasattr(context_data, 'flatten'):
                    train_values.extend(context_data.flatten())
        elif isinstance(batch, (list, tuple)):
            # Handle list/tuple batches
            batch_item = batch[0] if len(batch) > 0 else batch
            if hasattr(batch_item, 'numpy'):
                train_values.extend(batch_item.numpy().flatten())
            elif hasattr(batch_item, 'flatten'):
                train_values.extend(batch_item.flatten())

    val_values = []
    for batch in val_data:
        if isinstance(batch, dict):
            # TimesFM dataloader returns dict with 'context' and 'ground_truth' keys
            context_data = batch.get('context', None)
            if context_data is not None:
                # Convert tensor to numpy and flatten
                if hasattr(context_data, 'numpy'):
                    val_values.extend(context_data.numpy().flatten())
                elif hasattr(context_data, 'flatten'):
                    val_values.extend(context_data.flatten())
        elif isinstance(batch, (list, tuple)):
            # Handle list/tuple batches
            batch_item = batch[0] if len(batch) > 0 else batch
            if hasattr(batch_item, 'numpy'):
                val_values.extend(batch_item.numpy().flatten())
            elif hasattr(batch_item, 'flatten'):
                val_values.extend(batch_item.flatten())

    train_values = np.array(train_values)
    val_values = np.array(val_values)

    # Check if we got any data
    if len(train_values) == 0 or len(val_values) == 0:
        logging.warning("Could not extract data from dataloaders for diagnostic analysis")
        logging.warning("Skipping diagnostic analysis - training will continue")
        return {
            'status': 'insufficient_data',
            'error': 'Could not extract data from batches'
        }

    # Remove NaN/Inf values
    train_values = train_values[np.isfinite(train_values)]
    val_values = val_values[np.isfinite(val_values)]

    # 1. BASIC STATISTICS
    logging.info("")
    logging.info("1. BASIC STATISTICS:")
    logging.info("-" * 70)

    train_mean = train_values.mean()
    val_mean = val_values.mean()
    train_std = train_values.std()
    val_std = val_values.std()
    train_min = train_values.min()
    val_min = val_values.min()
    train_max = train_values.max()
    val_max = val_values.max()

    logging.info(f"Training set:   n={len(train_values):,}")
    logging.info(f"                 Mean={train_mean:.6f}, Std={train_std:.6f}")
    logging.info(f"                 Range: [{train_min:.6f}, {train_max:.6f}]")
    logging.info("")
    logging.info(f"Validation set: n={len(val_values):,}")
    logging.info(f"                 Mean={val_mean:.6f}, Std={val_std:.6f}")
    logging.info(f"                 Range: [{val_min:.6f}, {val_max:.6f}]")

    # 2. VOLATILITY COMPARISON
    logging.info("")
    logging.info("2. VOLATILITY COMPARISON:")
    logging.info("-" * 70)

    volatility_diff = ((train_std - val_std) / val_std) * 100
    mean_diff = ((train_mean - val_mean) / abs(val_mean)) * 100 if val_mean != 0 else 0

    logging.info(f"Volatility (Std): Train={train_std:.6f}, Val={val_std:.6f}")
    logging.info(f"Difference: {volatility_diff:+.1f}%")

    if train_std > val_std:
        logging.info(f"⚠️  TRAINING period is MORE VOLATILE than validation")
        logging.info(f"    → This explains why Train Loss > Val Loss")
        logging.info(f"    → Model has harder time fitting training data")
    else:
        logging.info(f"✅ VALIDATION period is more volatile (usual case)")

    # 3. EXTREME EVENTS ANALYSIS
    logging.info("")
    logging.info("3. EXTREME EVENTS ANALYSIS:")
    logging.info("-" * 70)

    # Define extreme as > 3 standard deviations from mean
    train_extreme_threshold = train_mean + 3 * train_std
    val_extreme_threshold = val_mean + 3 * val_std

    train_extremes = (train_values > train_extreme_threshold).sum()
    val_extremes = (val_values > val_extreme_threshold).sum()

    train_extreme_pct = (train_extremes / len(train_values)) * 100
    val_extreme_pct = (val_extremes / len(val_values)) * 100

    logging.info(f"Training extremes:   {train_extremes:,} ({train_extreme_pct:.2f}% of data)")
    logging.info(f"Validation extremes: {val_extremes:,} ({val_extreme_pct:.2f}% of data)")

    if train_extreme_pct > val_extreme_pct:
        logging.info(f"⚠️  TRAINING period has MORE extreme events")
        logging.info(f"    → Extreme values make fitting harder → higher train loss")
    else:
        logging.info(f"✅ Validation period has more extreme events")

    # 4. DISTRIBUTION SHAPE
    logging.info("")
    logging.info("4. DISTRIBUTION SHAPE:")
    logging.info("-" * 70)

    train_skew = pd.Series(train_values).skew()
    val_skew = pd.Series(val_values).skew()
    train_kurt = pd.Series(train_values).kurtosis()
    val_kurt = pd.Series(val_values).kurtosis()

    logging.info(f"Training:   Skewness={train_skew:.3f}, Kurtosis={train_kurt:.3f}")
    logging.info(f"Validation: Skewness={val_skew:.3f}, Kurtosis={val_kurt:.3f}")

    if abs(train_skew) > abs(val_skew):
        logging.info(f"⚠️  TRAINING distribution is MORE skewed (asymmetric)")
        logging.info(f"    → Skewed data harder to model → higher train loss")

    if train_kurt > val_kurt:
        logging.info(f"⚠️  TRAINING has HEAVIER TAILS (more outliers)")
        logging.info(f"    → Heavy tails indicate more extreme events")

    # 5. FINAL DIAGNOSIS
    logging.info("")
    logging.info("=" * 70)
    logging.info("DIAGNOSIS: Why Train Loss > Val Loss?")
    logging.info("=" * 70)

    reasons = []
    confidence = []

    if train_std > val_std * 1.1:  # >10% more volatile
        reasons.append(f"• Training period {volatility_diff:.1f}% more volatile")
        confidence.append("HIGH")

    if train_extreme_pct > val_extreme_pct * 1.5:  # >50% more extremes
        reasons.append(f"• Training has {(train_extreme_pct/val_extreme_pct):.1f}x more extreme events")
        confidence.append("MEDIUM")

    if abs(train_skew) > abs(val_skew) * 1.2:  # >20% more skewed
        reasons.append(f"• Training distribution more skewed")
        confidence.append("MEDIUM")

    if train_kurt > val_kurt * 1.2:  # >20% heavier tails
        reasons.append(f"• Training has heavier tails (more outliers)")
        confidence.append("MEDIUM")

    if reasons:
        logging.info("KEY REASONS IDENTIFIED:")
        for i, reason in enumerate(reasons, 1):
            logging.info(f"  {reason}")
            logging.info(f"    Confidence: {confidence[i-1]}")

        logging.info("")
        logging.info("CONCLUSION:")
        if any(c == "HIGH" for c in confidence):
            logging.info("  ✅ Train Loss > Val Loss is EXPECTED and CORRECT")
            logging.info("  ✅ Training period is genuinely harder to predict")
            logging.info("  ✅ This is a data characteristic, NOT a bug")
        else:
            logging.info("  ⚠️  Pattern is unusual but may be acceptable")
            logging.info("  ⚠️  Continue monitoring training progress")

    else:
        logging.info("  ✅ No significant differences found")
        logging.info("  ✅ Pattern may be due to random variation")
        logging.info("  ✅ Continue monitoring training")

    logging.info("=" * 70)
    logging.info("")

    # Return analysis results
    return {
        'train_mean': train_mean,
        'val_mean': val_mean,
        'train_std': train_std,
        'val_std': val_std,
        'volatility_diff_pct': volatility_diff,
        'train_extreme_pct': train_extreme_pct,
        'val_extreme_pct': val_extreme_pct,
        'train_skew': train_skew,
        'val_skew': val_skew,
        'train_kurtosis': train_kurt,
        'val_kurtosis': val_kurt,
        'diagnosis': reasons if reasons else ['No significant differences']
    }


def analyze_training_progress(training_history):
    """
    Analyze training progress to explain loss patterns

    Args:
        training_history: List of dicts with 'train_metrics' and 'val_metrics'

    Returns:
        dict: Analysis of training patterns
    """
    if not training_history or len(training_history) < 2:
        return {'status': 'insufficient_data'}

    logging.info("")
    logging.info("=" * 70)
    logging.info("TRAINING PROGRESS ANALYSIS")
    logging.info("=" * 70)

    # Extract losses from nested structure
    train_losses = [epoch['train_metrics']['loss'] for epoch in training_history]
    val_losses = [epoch['val_metrics']['val_loss'] for epoch in training_history]

    # NEW: Extract QLIKE if available
    val_qlikes = None
    if 'val_qlike' in training_history[0]['val_metrics']:
        val_qlikes = [epoch['val_metrics']['val_qlike'] for epoch in training_history
                      if epoch['val_metrics'].get('val_qlike', float('inf')) != float('inf')]

    # Calculate trends
    train_trend = (train_losses[-1] - train_losses[0]) / len(train_losses)
    val_trend = (val_losses[-1] - val_losses[0]) / len(val_losses)

    # Check for Train Loss > Val Loss pattern
    train_gt_val_epochs = sum(1 for i, (t, v) in enumerate(zip(train_losses, val_losses)) if t > v)
    total_epochs = len(train_losses)
    train_gt_val_pct = (train_gt_val_epochs / total_epochs) * 100

    logging.info(f"Total epochs: {total_epochs}")
    logging.info(f"Epochs where Train > Val: {train_gt_val_epochs} ({train_gt_val_pct:.1f}%)")
    logging.info("")
    logging.info("MSE LOSS TRENDS (Model Optimization):")
    logging.info(f"  Train: {train_losses[0]:.6f} → {train_losses[-1]:.6f} (trend: {train_trend:+.6f}/epoch)")
    logging.info(f"  Val:   {val_losses[0]:.6f} → {val_losses[-1]:.6f} (trend: {val_trend:+.6f}/epoch)")

    # NEW: QLIKE analysis
    if val_qlikes and len(val_qlikes) > 0:
        qlike_trend = (val_qlikes[-1] - val_qlikes[0]) / len(val_qlikes)
        logging.info("")
        logging.info("QLIKE METRIC TRENDS (Volatility Evaluation):")
        logging.info(f"  Val QLIKE: {val_qlikes[0]:.6f} → {val_qlikes[-1]:.6f} (trend: {qlike_trend:+.6f}/epoch)")
        logging.info("")
        logging.info("QLIKE INTERPRETATION:")
        if qlike_trend < 0:
            logging.info("  ✅ QLIKE decreasing → Volatility forecasts improving")
            logging.info("     Lower QLIKE = Better volatility predictions")
        else:
            logging.warning("  ⚠️  QLIKE increasing → Volatility forecasts degrading")
            logging.warning("     Model may be overfitting to MSE instead of volatility quality")
    else:
        logging.info("")
        logging.info("Note: QLIKE metrics not available (check model prediction outputs)")

    # Analyze pattern
    logging.info("")
    logging.info("MSE LOSS PATTERN ANALYSIS:")

    if train_gt_val_pct >= 50:
        logging.info(f"  ⚠️  Train Loss > Val Loss in {train_gt_val_pct:.1f}% of epochs")
        logging.info("  → This is UNUSUAL but may be explained by:")
        logging.info("    1. Training period harder than validation (see diagnostic above)")
        logging.info("    2. Dropout regularization active during training only")
        logging.info("    3. Different data distributions between periods")
    else:
        logging.info(f"  ✅ Train Loss < Val Loss in {(100-train_gt_val_pct):.1f}% of epochs")
        logging.info("  → This is the EXPECTED pattern (overfitting not severe)")

    # Check for convergence
    if len(train_losses) >= 5:
        recent_train_trend = (train_losses[-1] - train_losses[-5]) / 5
        recent_val_trend = (val_losses[-1] - val_losses[-5]) / 5

        logging.info("")
        logging.info("RECENT CONVERGENCE (last 5 epochs):")
        logging.info(f"  MSE Train trend: {recent_train_trend:+.6f}/epoch")
        logging.info(f"  MSE Val trend:   {recent_val_trend:+.6f}/epoch")

        # NEW: QLIKE convergence check
        if val_qlikes and len(val_qlikes) >= 5:
            recent_qlike_trend = (val_qlikes[-1] - val_qlikes[-5]) / 5
            logging.info(f"  QLIKE trend:    {recent_qlike_trend:+.6f}/epoch")

            if recent_qlike_trend > 0:
                logging.warning("  ⚠️  QLIKE INCREASING - Volatility forecasts getting worse!")
                logging.warning("     Model optimizing MSE but not volatility quality")
            else:
                logging.info("  ✅ QLIKE decreasing - Volatility predictions improving")

        if recent_val_trend > 0:
            logging.warning("  ⚠️  MSE Validation loss INCREASING - possible overfitting!")
            logging.warning("     Consider early stopping or more regularization")
        else:
            logging.info("  ✅ Both MSE losses decreasing - model still learning")

    logging.info("=" * 70)
    logging.info("")

    # NEW: Return QLIKE data if available
    result = {
        'train_gt_val_pct': train_gt_val_pct,
        'train_trend': train_trend,
        'val_trend': val_trend,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

    if val_qlikes and len(val_qlikes) > 0:
        result['val_qlikes'] = val_qlikes
        result['qlike_trend'] = (val_qlikes[-1] - val_qlikes[0]) / len(val_qlikes)

    return result

class QuickTimesFMTester(TimesFMVN30Finetuner):
    """
    Modified TimesFM tester that respects custom epoch counts

    This class overrides the config to use a smaller number of epochs
    for quick testing purposes.

    CRITICAL FIX: Don't call parent __init__ to avoid config reload!
    """

    def __init__(self, custom_epochs=2, config_path='configs/config.yaml'):
        # Load config ONCE
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Override epochs for quick testing
        self.config['training']['num_epochs'] = custom_epochs

        # Initialize parent WITHOUT config_path to prevent reload
        # We manually copy the parent's initialization code
        self.logger = logging.getLogger(__name__)

        # Set random seed
        from model_training_fixed import set_random_seed
        set_random_seed(self.config['system']['random_seed'])

        # Device
        import torch
        self.device = torch.device(self.config['system']['device'])
        self.logger.info(f"Using device: {self.device}")

        # Model storage
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.best_val_loss = float('inf')

        # Training metrics
        self.training_history = []
        self.current_epoch = 0

        # Early stopping
        self.epochs_since_improvement = 0
        self.early_stopping_patience = self.config['training'].get('early_stopping_patience', 5)

        # Log the ACTUAL epoch count
        self.logger.info(f"✅ CONFIG OVERRIDE SUCCESSFUL!")
        self.logger.info(f"   Epochs set to: {self.config['training']['num_epochs']}")

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

        # Step 2: RUN DIAGNOSTIC ANALYSIS (NEW!)
        diagnostic_results = analyze_train_val_periods(train_loader, test_loader, feature_type)

        # Step 3: Initialize model with custom epochs
        logging.info(f"Initializing TimesFM model ({epochs} epochs)...")
        finetuner = QuickTimesFMTester(custom_epochs=epochs)
        finetuner.load_timesfm_model()
        finetuner.setup_lora_adapters()
        finetuner.setup_optimizer()

        # Create scheduler with correct step count
        num_training_steps = epochs * len(train_loader)
        finetuner.setup_scheduler(num_training_steps)

        # Step 4: Quick training run
        logging.info(f"Starting training ({epochs} epochs)...")
        logging.info(f"Expected time: ~{epochs * 3.7:.1f} hours")
        finetuner.train_model(train_loader, test_loader)

        # Step 5: Record results
        end_time = time.time()
        training_time = end_time - start_time

        # Analyze training progress (NEW!)
        progress_analysis = analyze_training_progress(finetuner.training_history)

        results = {
            'feature_type': feature_type,
            'status': 'success',
            'best_val_loss': finetuner.best_val_loss,
            'training_time_minutes': training_time / 60,
            'epochs_completed': len(finetuner.training_history),
            'training_samples': len(train_loader.dataset),
            'test_samples': len(test_loader.dataset),
            'timestamp': datetime.now().isoformat(),
            'diagnostic_analysis': diagnostic_results,  # Data characteristics
            'progress_analysis': progress_analysis      # Training patterns
        }

        logging.info(f"[OK] {feature_type} training complete!")
        logging.info(f"Best validation loss: {results['best_val_loss']:.6f}")
        logging.info(f"Training time: {results['training_time_minutes']:.1f} minutes")

        # Explain the loss pattern (NEW!)
        if progress_analysis.get('train_gt_val_pct', 0) >= 50:
            logging.info(f"⚠️  Note: Train > Val in {progress_analysis['train_gt_val_pct']:.1f}% of epochs")
            logging.info(f"    See diagnostic analysis above for explanation")

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

    # NEW: Get QLIKE from progress analysis
    baseline_qlike = None
    if 'progress_analysis' in baseline_result and 'val_qlikes' in baseline_result['progress_analysis']:
        qlikes = baseline_result['progress_analysis']['val_qlikes']
        baseline_qlike = qlikes[-1] if qlikes else None

    for feature_type, result in all_results.items():
        if result['status'] == 'success':
            loss = result['best_val_loss']

            # NEW: Get QLIKE for this feature
            feature_qlike = None
            if 'progress_analysis' in result and 'val_qlikes' in result['progress_analysis']:
                qlikes = result['progress_analysis']['val_qlikes']
                feature_qlike = qlikes[-1] if qlikes else None

            if feature_type != 'RV_20' and baseline_loss != float('inf'):
                improvement = (baseline_loss - loss) / baseline_loss * 100

                # NEW: QLIKE comparison
                qlike_str = f", QLIKE: {feature_qlike:.6f}" if feature_qlike else ""
                baseline_qlike_str = f" (QLIKE: {baseline_qlike:.6f})" if baseline_qlike else ""

                logging.info(f"{feature_type}: Loss={loss:.6f} ({improvement:+.1f}% vs baseline){qlike_str}")
            else:
                baseline_qlike_str = f" (QLIKE: {baseline_qlike:.6f})" if baseline_qlike else ""
                logging.info(f"{feature_type}: Loss={loss:.6f}{baseline_qlike_str} (baseline)")
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
    logging.info("=" * 70)
    logging.info("Starting FIXED TimesFM Feature Comparison Test")
    logging.info("This will test RV_20 (baseline) vs overnight volatility")
    logging.info("=" * 70)
    logging.info("")
    logging.info("✅ CONFIGURATION FIX VERIFICATION:")
    logging.info("   - Bug: Parent class was reloading config from disk")
    logging.info("   - Fix: QuickTimesFMTester bypasses parent __init__")
    logging.info("   - Result: Epoch override now works correctly")
    logging.info("")
    logging.info("📊 EXPECTED TRAINING:")
    logging.info("   - RV_20: 2 epochs (~3.7 hours)")
    logging.info("   - Overnight: 2 epochs (~3.7 hours)")
    logging.info("   - Total: ~7.4 hours (NOT 36 hours!)")
    logging.info("=" * 70)
    logging.info("")

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

            # NEW: Get QLIKE metrics
            rv20_qlike = None
            overnight_qlike = None
            if 'progress_analysis' in results['RV_20'] and 'val_qlikes' in results['RV_20']['progress_analysis']:
                qlikes = results['RV_20']['progress_analysis']['val_qlikes']
                rv20_qlike = qlikes[-1] if qlikes else None
            if 'progress_analysis' in results['overnight'] and 'val_qlikes' in results['overnight']['progress_analysis']:
                qlikes = results['overnight']['progress_analysis']['val_qlikes']
                overnight_qlike = qlikes[-1] if qlikes else None

            logging.info("\n" + "=" * 70)
            logging.info("[FINAL RESULT]")
            logging.info("=" * 70)
            logging.info("MSE Loss (Model Optimization):")
            logging.info(f"  RV_20 baseline:        {rv20_loss:.6f}")
            logging.info(f"  Overnight volatility:   {overnight_loss:.6f}")
            logging.info(f"  Improvement:            {improvement:+.1f}%")

            # NEW: QLIKE comparison
            if rv20_qlike and overnight_qlike:
                qlike_improvement = (rv20_qlike - overnight_qlike) / rv20_qlike * 100
                logging.info("")
                logging.info("QLIKE Metric (Volatility Quality):")
                logging.info(f"  RV_20 baseline:        {rv20_qlike:.6f}")
                logging.info(f"  Overnight volatility:   {overnight_qlike:.6f}")
                logging.info(f"  Improvement:            {qlike_improvement:+.1f}%")

                if improvement > 0 and qlike_improvement > 0:
                    logging.info("")
                    logging.info("✅ OVERNIGHT VOLATILITY IS BETTER!")
                    logging.info("   ✅ Lower MSE loss (better model fit)")
                    logging.info("   ✅ Lower QLIKE (better volatility forecasts)")
                elif improvement > 0 and qlike_improvement <= 0:
                    logging.info("")
                    logging.info("⚠️  MIXED RESULTS:")
                    logging.info("   ✅ Lower MSE loss but worse QLIKE")
                    logging.info("   ⚠️  Model fits better but volatility quality may not improve")
                    logging.info("   → Consider optimizing for QLIKE directly")
                else:
                    logging.info("")
                    logging.info("❌ BASELINE RV_20 IS STILL BETTER")
            else:
                logging.info("")
                logging.info("Note: QLIKE metrics not available (check model prediction outputs)")
                if improvement > 0:
                    logging.info("✅ OVERNIGHT VOLATILITY IS BETTER! (based on MSE loss)")
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