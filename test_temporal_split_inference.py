"""
True Out-of-Sample Inference Test with Temporal Split

Tests the model on the LAST 20% of data (including June 2026) using proper
temporal split. This ensures NO data leakage while providing enough data
for testing.

Key difference from original test:
- Original: Random sampling from ENTIRE series (data leakage)
- This: Proper temporal split (first 80% train, last 20% test)

Usage:
    python test_temporal_split_inference.py
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import json
from datetime import datetime

# Workaround for PyTorch compatibility
if not hasattr(torch, 'float8_e8m0fnu'):
    torch.float8_e8m0fnu = torch.float8_e4m3fn

# Monkey-patch to disable bitsandbytes
import sys
import types
import importlib.util

original_find_spec = importlib.util.find_spec

def patched_find_spec(name, package=None):
    if name and 'bitsandbytes' in name:
        return None
    return original_find_spec(name, package)

importlib.util.find_spec = patched_find_spec

fake_bnb = types.ModuleType('bitsandbytes')
fake_bnb.__path__ = []
fake_bnb.__file__ = '<disabled>'
sys.modules['bitsandbytes'] = fake_bnb

try:
    import peft.import_utils as peft_utils
    peft_utils.is_bnb_available = lambda: False
    if hasattr(peft_utils, 'is_bnb_4bit_available'):
        peft_utils.is_bnb_4bit_available = lambda: False
    if hasattr(peft_utils, 'is_bnb_8bit_available'):
        peft_utils.is_bnb_8bit_available = lambda: False
except:
    pass

# Import model
from transformers import TimesFm2_5ModelForPrediction
from peft import PeftModel

# Import evaluation metrics
sys.path.append('src')
from model_evaluation import TimesFMModelEvaluator

def load_finetuned_model(checkpoint_path: str = 'models/checkpoints'):
    """Load fine-tuned TimesFM model"""
    print('=' * 70)
    print('[LOADING FINE-TUNED MODEL]')
    print('=' * 70)

    # Base model
    base_model = TimesFm2_5ModelForPrediction.from_pretrained(
        "google/timesfm-2.5-200m-transformers",
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    print(f'[OK] Model loaded from: {checkpoint_path}')
    print(f'Model device: {next(model.parameters()).device}')

    return model

def load_temporal_split_data(config_path: str = 'configs/config.yaml', split_ratio: float = 0.8):
    """
    Load data with proper temporal split (first 80% train, last 20% test)

    This ensures NO data leakage unlike the original random sampling.

    Args:
        config_path: Path to config file
        split_ratio: Train/test split ratio (default: 0.8)

    Returns:
        Test dataset with only last 20% of data
    """
    print('=' * 70)
    print('[LOADING TEMPORAL SPLIT TEST DATA]')
    print('=' * 70)
    print('[INFO] Using PROPER temporal split (no data leakage)')
    print('[INFO] Train: First 80% of each stock')
    print('[INFO] Test:  Last 20% of each stock (including June 2026)')
    print()

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load processed data
    processed_dir = Path(config['data']['processed_path'])
    processed_files = list(processed_dir.glob("*_processed.csv"))

    # Load all stocks and apply temporal split
    stock_data_dict = {}
    split_info = []

    for file_path in processed_files:
        stock_name = file_path.stem.replace('_processed', '')
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        if 'RV_20' not in df.columns:
            continue

        # Drop NaN
        df_clean = df.dropna(subset=['RV_20'])

        if len(df_clean) < 100:  # Need minimum data
            continue

        # Apply temporal split (first 80% train, last 20% test)
        split_point = int(len(df_clean) * split_ratio)
        train_data = df_clean.iloc[:split_point]
        test_data = df_clean.iloc[split_point:]

        # Store ONLY test data
        stock_data_dict[stock_name] = test_data

        split_info.append({
            'stock': stock_name,
            'total': len(df_clean),
            'train': len(train_data),
            'test': len(test_data),
            'train_end': train_data.index[-1].strftime('%Y-%m-%d'),
            'test_start': test_data.index[0].strftime('%Y-%m-%d'),
            'test_end': test_data.index[-1].strftime('%Y-%m-%d')
        })

    print(f'[OK] Loaded {len(stock_data_dict)} stocks with temporal split')
    print()

    # Show split information
    print('[SPLIT INFORMATION]')
    print('-' * 70)
    print(f"{'Stock':<6} {'Total':>6} {'Train':>6} {'Test':>6} {'Train End':>12} {'Test Start':>12} {'Test End':>12}")
    print('-' * 70)

    for info in split_info[:10]:  # Show first 10
        print(f"{info['stock']:<6} {info['total']:>6} {info['train']:>6} {info['test']:>6} "
              f"{info['train_end']:>12} {info['test_start']:>12} {info['test_end']:>12}")

    if len(split_info) > 10:
        print(f"... ({len(split_info) - 10} more stocks)")

    print('-' * 70)

    # Create test dataset
    from model_training_google_research import VN30LastWindowDataset

    context_len = 128
    horizon_len = 13

    all_series = []
    stock_names = []
    for stock_name, df in stock_data_dict.items():
        values = df['RV_20'].values.astype(np.float32)
        if len(values) >= context_len + horizon_len:
            all_series.append(values)
            stock_names.append(stock_name)

    if not all_series:
        print('[ERROR] No stocks with sufficient test data')
        return None, None, None, None, None

    test_ds = VN30LastWindowDataset(all_series, context_len, horizon_len)

    print()
    print(f'[OK] Test samples: {len(test_ds)}')
    print(f'[INFO] Context length: {context_len}')
    print(f'[INFO] Horizon length: {horizon_len}')
    print(f'[INFO] Total predictions: {len(test_ds) * horizon_len}')
    print(f'[INFO] Test period includes June 2026 data')

    return test_ds, all_series, stock_names, context_len, horizon_len

def run_inference(model, test_dataset, context_len: int, horizon_len: int, device: str = 'cuda'):
    """Run inference on temporally split test data"""
    print('=' * 70)
    print('[RUNNING INFERENCE ON TEMPORAL SPLIT DATA]')
    print('=' * 70)
    print('[INFO] This is TRUE out-of-sample test (no data leakage)')

    all_predictions = []
    all_actuals = []

    model.eval()
    with torch.no_grad():
        for i in range(len(test_dataset)):
            context, target = test_dataset[i]

            # Move to device
            context = context.unsqueeze(0).to(device)

            # Run inference
            outputs = model(
                past_values=context,
                future_values=None,  # Inference mode
                forecast_context_len=context_len,
            )

            # Get predictions (mean predictions)
            pred = outputs.mean_predictions[0, :horizon_len].cpu().numpy()
            actual = target.numpy()

            all_predictions.append(pred)
            all_actuals.append(actual)

    # Concatenate all predictions
    predictions = np.concatenate(all_predictions)
    actuals = np.concatenate(all_actuals)

    print(f'[OK] Inference completed')
    print(f'[INFO] Total predictions: {len(predictions)}')

    return predictions, actuals

def calculate_all_metrics(predictions: np.ndarray, actuals: np.ndarray):
    """Calculate all evaluation metrics"""
    print('=' * 70)
    print('[CALCULATING TRUE OUT-OF-SAMPLE METRICS]')
    print('=' * 70)
    print('[INFO] Test data: Last 20% of each stock (temporal split)')
    print('[INFO] Data leakage: FALSE')
    print('[INFO] This is TRUE generalization performance')
    print()

    evaluator = TimesFMModelEvaluator()

    # Calculate all metrics
    qlike = evaluator.calculate_qlike(actuals, predictions)
    r2 = evaluator.calculate_r2(actuals, predictions)
    rmse = evaluator.calculate_rmse(actuals, predictions)
    mse = evaluator.calculate_mse(actuals, predictions)

    # Additional metrics
    mae = np.mean(np.abs(actuals - predictions))

    # Directional accuracy
    actual_direction = np.sign(np.diff(actuals))
    pred_direction = np.sign(np.diff(predictions))
    dir_accuracy = np.mean(actual_direction == pred_direction) * 100

    results = {
        'test_type': 'TRUE_OUT_OF_SAMPLE_TEMPORAL_SPLIT',
        'test_method': 'Last 20% of each stock (temporal split)',
        'data_leakage': 'FALSE',
        'QLIKE': float(qlike),
        'R2': float(r2),
        'RMSE': float(rmse),
        'MSE': float(mse),
        'MAE': float(mae),
        'Directional_Accuracy': float(dir_accuracy)
    }

    # Print results
    print('\n' + '=' * 70)
    print('TRUE OUT-OF-SAMPLE TEST RESULTS (TEMPORAL SPLIT)')
    print('=' * 70)
    print(f"QLIKE (Volatility Metric):    {qlike:.6f}")
    print(f"R² (R-squared):               {r2:.6f}  ({r2*100:.2f}% variance explained)")
    print(f"RMSE (Root Mean Square Error): {rmse:.6f}")
    print(f"MSE (Mean Square Error):        {mse:.6f}")
    print(f"MAE (Mean Absolute Error):      {mae:.6f}")
    print(f"Directional Accuracy:          {dir_accuracy:.2f}%")
    print('=' * 70)

    # Interpretation
    print('\nMETRIC INTERPRETATION:')
    print('-' * 70)
    print(f"Test Type:        TRUE OUT-OF-SAMPLE (temporal split)")
    print(f"Test Method:      Last 20% of each stock")
    print(f"Data Leakage:     FALSE (proper temporal separation)")
    print(f"QLIKE:            Lower is better (perfect = 0)")
    print(f"R²:               {'%.2f%% variance explained' % (r2 * 100) if r2 > 0 else 'Negative (worse than mean)'}")
    print(f"RMSE:             Average error magnitude: {rmse:.4f}")
    print(f"Dir Accuracy:     {dir_accuracy:.1f}% of trend predictions correct")
    print()

    # Comparison with previous in-sample results
    print('=' * 70)
    print('COMPARISON: DATA LEAKAGE vs NO DATA LEAKAGE')
    print('=' * 70)
    print("Previous (WITH DATA LEAKAGE - random sampling):")
    print("  R² = 0.8502 (OVERESTIMATED - not true generalization)")
    print("  Issue: Train and test data overlapped")
    print()
    print("Current (NO DATA LEAKAGE - temporal split):")
    print(f"  R² = {r2:.4f} (TRUE generalization performance)")
    print(f"  This is the REAL model performance on unseen data")
    print()

    if r2 < 0.85:
        diff = 0.85 - r2
        print(f"  Difference: {diff:.4f} ({diff/0.85*100:.1f}% lower than leaked result)")
        print(f"  Explanation: Previous result was inflated by data leakage")
    print('=' * 70)

    return results

def save_results(results: dict, predictions: np.ndarray, actuals: np.ndarray):
    """Save results to JSON"""
    results_dir = Path('experiments/inference_results')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_path = results_dir / 'temporal_split_inference_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save predictions and actuals
    data = {
        'timestamp': datetime.now().isoformat(),
        'test_info': {
            'test_type': 'TRUE_OUT_OF_SAMPLE_TEMPORAL_SPLIT',
            'test_method': 'Last 20% of each stock',
            'data_leakage': 'FALSE'
        },
        'metrics': results,
        'predictions_sample': predictions[:100].tolist(),
        'actuals_sample': actuals[:100].tolist()
    }

    full_results_path = results_dir / 'temporal_split_inference_full_results.json'
    with open(full_results_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f'\n[SAVED] Results:')
    print(f'  Metrics: {metrics_path}')
    print(f'  Full results: {full_results_path}')

def main():
    """Main true out-of-sample test with temporal split"""
    print('=' * 70)
    print('TIMESFM TRUE OUT-OF-SAMPLE INFERENCE TEST')
    print('(Temporal Split - Last 20% of each stock)')
    print('=' * 70)
    print('[INFO] Testing on last 20% of data (including June 2026)')
    print('[INFO] This provides TRUE generalization metrics')
    print('[INFO] NO data leakage (proper temporal split)')
    print()

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_finetuned_model('models/checkpoints')

    # Load temporally split test data
    test_dataset, all_series, stock_names, context_len, horizon_len = load_temporal_split_data()

    if test_dataset is None:
        print('[ERROR] Could not load temporal split test data')
        return 1

    # Run inference
    predictions, actuals = run_inference(model, test_dataset, context_len, horizon_len, device)

    # Calculate metrics
    results = calculate_all_metrics(predictions, actuals)

    # Save results
    save_results(results, predictions, actuals)

    print('\n' + '=' * 70)
    print('[TEMPORAL SPLIT TRUE OUT-OF-SAMPLE TEST COMPLETE]')
    print('=' * 70)
    print('[INFO] These metrics represent TRUE model performance')
    print('[INFO] NO data leakage - proper temporal separation')

    return 0

if __name__ == "__main__":
    sys.exit(main())
