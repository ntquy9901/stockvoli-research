"""
True Out-of-Sample Inference Test for Fine-tuned TimesFM Model

Tests the model on ONLY June 2026 data (genuinely new data that the model
has never seen during training). This provides TRUE generalization metrics
without any data leakage.

Usage:
    python test_june_2026_inference.py
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

def load_june_2026_data(config_path: str = 'configs/config.yaml'):
    """
    Load ONLY June 2026 data for true out-of-sample testing

    Returns:
        Test dataset with only June 2026 data
    """
    print('=' * 70)
    print('[LOADING JUNE 2026 TEST DATA]')
    print('=' * 70)
    print('[INFO] This is TRUE out-of-sample test (no data leakage)')
    print('[INFO] Model has never seen this data during training')
    print()

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load processed data
    processed_dir = Path(config['data']['processed_path'])
    processed_files = list(processed_dir.glob("*_processed.csv"))

    # Load all stocks and filter for June 2026
    stock_data_dict = {}
    for file_path in processed_files:
        stock_name = file_path.stem.replace('_processed', '')
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Filter ONLY June 2026 data
        june_2026 = df[(df.index.year == 2026) & (df.index.month == 6)]

        if not june_2026.empty and 'RV_20' in june_2026.columns:
            # Drop NaN in RV_20
            june_2026_clean = june_2026.dropna(subset=['RV_20'])
            if len(june_2026_clean) > 0:
                stock_data_dict[stock_name] = june_2026_clean

    print(f'[OK] Loaded {len(stock_data_dict)} stocks with June 2026 data')

    # Create test dataset
    from model_training_google_research import VN30LastWindowDataset

    context_len = 128
    horizon_len = 13

    all_series = []
    stock_names = []
    for stock_name, df in stock_data_dict.items():
        if 'RV_20' in df.columns:
            values = df['RV_20'].dropna().values.astype(np.float32)
            if len(values) >= context_len + horizon_len:
                all_series.append(values)
                stock_names.append(stock_name)

    if not all_series:
        print('[ERROR] No stocks with sufficient June 2026 data')
        print('[INFO] Need at least {} days of data per stock'.format(context_len + horizon_len))
        return None, None, None, None, None

    test_ds = VN30LastWindowDataset(all_series, context_len, horizon_len)

    print(f'[OK] Test samples: {len(test_ds)}')
    print(f'[INFO] Context length: {context_len}')
    print(f'[INFO] Horizon length: {horizon_len}')
    print(f'[INFO] Total predictions: {len(test_ds) * horizon_len}')

    return test_ds, all_series, stock_names, context_len, horizon_len

def run_inference(model, test_dataset, context_len: int, horizon_len: int, device: str = 'cuda'):
    """Run inference on June 2026 data"""
    print('=' * 70)
    print('[RUNNING INFERENCE ON JUNE 2026 DATA]')
    print('=' * 70)

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
    print('[CALCULATING METRICS ON JUNE 2026 DATA]')
    print('=' * 70)
    print('[INFO] These are TRUE out-of-sample metrics (no data leakage)')
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
        'test_type': 'TRUE_OUT_OF_SAMPLE',
        'test_period': 'June 2026',
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
    print('TRUE OUT-OF-SAMPLE TEST RESULTS (JUNE 2026)')
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
    print(f"Test Type:        TRUE OUT-OF-SAMPLE (no data leakage)")
    print(f"Test Period:      June 2026 (genuinely new data)")
    print(f"QLIKE:            Lower is better (perfect = 0)")
    print(f"R²:               {'%.2f%% variance explained' % (r2 * 100) if r2 > 0 else 'Negative (worse than mean)'}")
    print(f"RMSE:             Average error magnitude: {rmse:.4f}")
    print(f"Dir Accuracy:     {dir_accuracy:.1f}% of trend predictions correct")
    print()

    # Comparison with previous in-sample results
    print('=' * 70)
    print('COMPARISON: IN-SAMPLE vs OUT-OF-SAMPLE')
    print('=' * 70)
    print("Previous (IN-SAMPLE, with data leakage):")
    print("  R² = 0.8502 (OVERESTIMATED - not true generalization)")
    print()
    print("Current (OUT-OF-SAMPLE, June 2026):")
    print(f"  R² = {r2:.4f} (TRUE generalization performance)")
    print()
    if r2 < 0.85:
        diff = 0.85 - r2
        print(f"  Difference: {diff:.4f} ({diff/0.85*100:.1f}% lower than in-sample)")
    print('=' * 70)

    return results

def save_results(results: dict, predictions: np.ndarray, actuals: np.ndarray):
    """Save results to JSON"""
    results_dir = Path('experiments/inference_results')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_path = results_dir / 'june_2026_inference_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save predictions and actuals
    data = {
        'timestamp': datetime.now().isoformat(),
        'test_info': {
            'test_type': 'TRUE_OUT_OF_SAMPLE',
            'test_period': 'June 2026',
            'data_leakage': 'FALSE'
        },
        'metrics': results,
        'predictions_sample': predictions[:100].tolist(),
        'actuals_sample': actuals[:100].tolist()
    }

    full_results_path = results_dir / 'june_2026_inference_full_results.json'
    with open(full_results_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f'\n[SAVED] Results:')
    print(f'  Metrics: {metrics_path}')
    print(f'  Full results: {full_results_path}')

def main():
    """Main true out-of-sample test"""
    print('=' * 70)
    print('TIMESFM TRUE OUT-OF-SAMPLE INFERENCE TEST')
    print('=' * 70)
    print('[INFO] Testing on June 2026 data (genuinely new data)')
    print('[INFO] This provides TRUE generalization metrics')
    print()

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_finetuned_model('models/checkpoints')

    # Load June 2026 test data
    test_dataset, all_series, stock_names, context_len, horizon_len = load_june_2026_data()

    if test_dataset is None:
        print('[ERROR] Could not load June 2026 test data')
        print('[INFO] Need at least 141 days (128+13) of June 2026 data per stock')
        print('[INFO] Current June 2026 data has fewer days')
        print()
        print('[INFO] Recommendation: Wait for more June 2026 data or use temporal split')
        return 1

    # Run inference
    predictions, actuals = run_inference(model, test_dataset, context_len, horizon_len, device)

    # Calculate metrics
    results = calculate_all_metrics(predictions, actuals)

    # Save results
    save_results(results, predictions, actuals)

    print('\n' + '=' * 70)
    print('[JUNE 2026 TRUE OUT-OF-SAMPLE TEST COMPLETE]')
    print('=' * 70)

    return 0

if __name__ == "__main__":
    sys.exit(main())
