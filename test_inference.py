"""
Inference Test for Fine-tuned TimesFM Model

Tests the fine-tuned model on VN30 test data and calculates metrics:
- QLIKE (volatility-specific metric)
- R² (R-squared)
- RMSE (Root Mean Square Error)
- MSE (Mean Square Error)
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
    """
    Load fine-tuned TimesFM model

    Args:
        checkpoint_path: Path to LoRA adapter checkpoint

    Returns:
        Loaded model with LoRA adapters
    """
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

def load_test_data(config_path: str = 'configs/config.yaml'):
    """
    Load test data from VN30 stocks

    Returns:
        Test dataloader
    """
    print('=' * 70)
    print('[LOADING TEST DATA]')
    print('=' * 70)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load processed data
    processed_dir = Path(config['data']['processed_path'])
    processed_files = list(processed_dir.glob("*_processed.csv"))

    # Load all stocks
    stock_data_dict = {}
    for file_path in processed_files:
        stock_name = file_path.stem.replace('_processed', '')
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        stock_data_dict[stock_name] = df

    print(f'Loaded {len(stock_data_dict)} stocks')

    # Create test dataset (last 20% of data)
    from model_training_google_research import VN30LastWindowDataset

    context_len = 128  # From training
    horizon_len = 13   # From training

    all_series = []
    for stock_name, df in stock_data_dict.items():
        if 'RV_20' in df.columns:
            values = df['RV_20'].dropna().values.astype(np.float32)
            if len(values) >= context_len + horizon_len:
                all_series.append(values)

    test_ds = VN30LastWindowDataset(all_series, context_len, horizon_len)

    print(f'Test samples: {len(test_ds)} (last window from each stock)')
    print(f'Context length: {context_len}')
    print(f'Horizon length: {horizon_len}')

    return test_ds, all_series, context_len, horizon_len

def run_inference(model, test_dataset, context_len: int, horizon_len: int, device: str = 'cuda'):
    """
    Run inference on test data

    Returns:
        predictions and actuals arrays
    """
    print('=' * 70)
    print('[RUNNING INFERENCE]')
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
    print(f'Total predictions: {len(predictions)}')
    print(f'Prediction shape: {predictions.shape}')

    return predictions, actuals

def calculate_all_metrics(predictions: np.ndarray, actuals: np.ndarray):
    """
    Calculate all evaluation metrics

    Returns:
        Dictionary with all metrics
    """
    print('=' * 70)
    print('[CALCULATING METRICS]')
    print('=' * 70)

    evaluator = TimesFMModelEvaluator()

    # Calculate all metrics
    qlike = evaluator.calculate_qlike(actuals, predictions)
    r2 = evaluator.calculate_r2(actuals, predictions)
    rmse = evaluator.calculate_rmse(actuals, predictions)
    mse = evaluator.calculate_mse(actuals, predictions)

    # Additional metrics
    mae = np.mean(np.abs(actuals - predictions))

    # Directional accuracy (percentage of correct direction predictions)
    actual_direction = np.sign(np.diff(actuals))
    pred_direction = np.sign(np.diff(predictions))
    dir_accuracy = np.mean(actual_direction == pred_direction) * 100

    results = {
        'QLIKE': qlike,
        'R2': r2,
        'RMSE': rmse,
        'MSE': mse,
        'MAE': mae,
        'Directional_Accuracy': dir_accuracy
    }

    # Print results
    print('\n' + '=' * 70)
    print('EVALUATION METRICS')
    print('=' * 70)
    print(f"QLIKE (Volatility Metric):    {qlike:.6f}")
    print(f"R² (R-squared):               {r2:.6f}")
    print(f"RMSE (Root Mean Square Error): {rmse:.6f}")
    print(f"MSE (Mean Square Error):        {mse:.6f}")
    print(f"MAE (Mean Absolute Error):      {mae:.6f}")
    print(f"Directional Accuracy:          {dir_accuracy:.2f}%")
    print('=' * 70)

    # Interpretation
    print('\nMETRIC INTERPRETATION:')
    print('-' * 70)
    print(f"QLIKE: Lower is better (perfect = 0)")
    print(f"R²:    {'%.2f%% variance explained' % (r2 * 100) if r2 > 0 else 'Negative (worse than mean)'}")
    print(f"RMSE: Average error magnitude: {rmse:.4f}")
    print(f"Dir Accuracy: {dir_accuracy:.1f}% of trend predictions correct")

    return results

def save_results(results: dict, predictions: np.ndarray, actuals: np.ndarray):
    """Save results to JSON"""
    results_dir = Path('experiments/inference_results')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics (convert numpy types to native Python types)
    serializable_results = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                             for k, v in results.items()}

    metrics_path = results_dir / 'inference_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    # Save predictions and actuals (convert to lists)
    data = {
        'timestamp': datetime.now().isoformat(),
        'metrics': serializable_results,
        'predictions_sample': predictions[:100].tolist(),  # First 100 predictions
        'actuals_sample': actuals[:100].tolist()          # First 100 actuals
    }

    full_results_path = results_dir / 'inference_full_results.json'
    with open(full_results_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f'\n[SAVED] Results:')
    print(f'  Metrics: {metrics_path}')
    print(f'  Full results: {full_results_path}')

def main():
    """Main inference test"""
    print('=' * 70)
    print('TIMESFM FINE-TUNED MODEL INFERENCE TEST')
    print('=' * 70)
    print()

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_finetuned_model('models/checkpoints')

    # Load test data
    test_dataset, all_series, context_len, horizon_len = load_test_data()

    # Run inference
    predictions, actuals = run_inference(model, test_dataset, context_len, horizon_len, device)

    # Calculate metrics
    results = calculate_all_metrics(predictions, actuals)

    # Save results
    save_results(results, predictions, actuals)

    print('\n' + '=' * 70)
    print('[INFERENCE TEST COMPLETE]')
    print('=' * 70)

    return 0

if __name__ == "__main__":
    sys.exit(main())
