"""
TimesFM Incremental Learning Implementation
Based on TimesFM research paper (arXiv:2505.11163)

This module implements incremental fine-tuning for TimesFM model to adapt
to new Vietnamese stock market data over time.

Key Methodology from Paper:
"incremental fine-tuning, which allows the model to adapt to new financial
return data over time, is essential for learning volatility patterns effectively"
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import mlflow
import yaml
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimesFMDataset(Dataset):
    """
    Dataset for TimesFM incremental learning following paper methodology.

    Creates samples with 512-day context window for volatility prediction
    """

    def __init__(self, window_data: Dict[str, pd.DataFrame],
                 feature_cols: List[str],
                 target_col: str = 'RV_20',
                 context_length: int = 512):
        """
        Initialize TimesFM dataset

        Args:
            window_data: Dictionary of stock DataFrames for current window
            feature_cols: List of feature columns to use
            target_col: Target volatility column
            context_length: Context window length (512 from TimesFM paper)
        """
        self.window_data = window_data
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.context_length = context_length

        # Create samples
        self.samples = self._create_samples()
        logger.info(f"Created dataset with {len(self.samples)} samples")

    def _create_samples(self) -> List[Dict]:
        """Create training samples from window data"""
        samples = []

        for symbol, data in self.window_data.items():
            # Ensure we have all required columns
            if self.target_col not in data.columns:
                logger.warning(f"Target column {self.target_col} not found for {symbol}")
                continue

            if not all(col in data.columns for col in self.feature_cols):
                logger.warning(f"Missing features for {symbol}")
                continue

            # Create sliding window samples
            dates = sorted(data.index)

            for i in range(self.context_length, len(dates) - 1):
                context_dates = dates[i-self.context_length:i]
                target_date = dates[i]

                # Get context features
                context_features = data.loc[context_dates, self.feature_cols].values

                # Get target value
                target_value = data.loc[target_date, self.target_col]

                # Skip if missing values
                if np.isnan(context_features).any() or np.isnan(target_value):
                    continue

                samples.append({
                    'context': context_features.astype(np.float32),
                    'target': float(target_value),
                    'symbol': symbol,
                    'target_date': target_date
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        return {
            'context': torch.tensor(sample['context'], dtype=torch.float32),
            'target': torch.tensor(sample['target'], dtype=torch.float32),
            'symbol': sample['symbol'],
            'target_date': sample['target_date']
        }


class TimesFMIncrementalLearner:
    """
    TimesFM Incremental Learning implementation based on paper methodology.

    From TimesFM paper (arXiv:2505.11163):
    "incremental fine-tuning, which allows the model to adapt to new financial
    return data over time, is essential for learning volatility patterns effectively"
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize TimesFM incremental learner"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Get configuration
        self.device = self.config['system']['device']
        self.context_length = self.config['features']['context_length']
        self.window_size = self.config['incremental_learning']['window_size']

        # Training configuration
        training_config = self.config['incremental_learning']['training']
        self.batch_size = training_config['batch_size']
        self.learning_rate = training_config['learning_rate']
        self.gradient_clip = training_config['gradient_clip']

        # Model state
        self.incremental_step = 0
        self.best_loss = float('inf')
        self.model = None
        self.optimizer = None

        # MLflow tracking
        self._setup_mlflow()

        logger.info(f"Initialized TimesFM Incremental Learner on {self.device}")

    def _setup_mlflow(self):
        """Setup MLflow experiment tracking"""
        mlflow_config = self.config.get('experiment_tracking', {})

        if mlflow_config.get('backend') == 'mlflow':
            experiment_name = mlflow_config.get('experiment_name', 'vietnamese_volatility_forecasting')
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment: {experiment_name}")

    def load_base_model(self, model_path: Optional[str] = None):
        """
        Load base TimesFM model

        Args:
            model_path: Path to pre-trained model, or None to download from HuggingFace
        """
        logger.info("Loading TimesFM base model...")

        try:
            from transformers import AutoModelForTimeSeriesForecasting

            if model_path is None:
                model_name = self.config['models']['timesfm']['model_name']
                logger.info(f"Loading model from HuggingFace: {model_name}")
                self.model = AutoModelForTimeSeriesForecasting.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
                )
            else:
                logger.info(f"Loading model from local path: {model_path}")
                self.model = AutoModelForTimeSeriesForecasting.from_pretrained(model_path)

            self.model.to(self.device)
            logger.info("✓ TimesFM model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading TimesFM model: {e}")
            logger.info("Creating placeholder model for testing")

            # Create placeholder model for testing
            self.model = self._create_placeholder_model()

    def _create_placeholder_model(self) -> nn.Module:
        """Create placeholder model for testing when TimesFM is not available"""
        class PlaceholderTimesFM(nn.Module):
            def __init__(self, input_dim=512, hidden_dim=128, output_dim=1):
                super().__init__()
                self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                self.decoder = nn.Linear(hidden_dim, output_dim)

            def forward(self, context):
                # context shape: [batch_size, sequence_length, features]
                lstm_out, _ = self.encoder(context)

                # Take last timestep output
                last_output = lstm_out[:, -1, :]

                # Predict volatility
                predictions = self.decoder(last_output)

                return predictions

        input_dim = len(self.config['features']['volatility_windows']) + 10  # Approximate
        model = PlaceholderTimesFM(input_dim=input_dim)
        model.to(self.device)

        logger.info("Created placeholder TimesFM model")
        return model

    def prepare_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """
        Prepare feature columns for training

        Args:
            data: Sample DataFrame

        Returns:
            List of feature column names
        """
        feature_cols = []

        # Volatility features
        for window in self.config['features']['volatility_windows']:
            feature_cols.extend([f'RV_{window}', f'RV_{window}_MA'])

        # Technical indicators
        tech_indicators = ['MA_10', 'MA_20', 'MA_50', 'EMA_12', 'EMA_26',
                          'MACD', 'RSI', 'ATR', 'BB_Width']

        for indicator in tech_indicators:
            if indicator in data.columns:
                feature_cols.append(indicator)

        # Vietnamese market features
        viet_features = ['Day_Of_Week', 'Month_Start', 'Month_End',
                        'Is_Tet_Period', 'Trend']

        for feature in viet_features:
            if feature in data.columns:
                feature_cols.append(feature)

        logger.info(f"Using {len(feature_cols)} features")
        return feature_cols

    def incremental_update(self, window_data: Dict[str, pd.DataFrame],
                         window_id: Optional[int] = None) -> Dict:
        """
        Perform incremental update on new data window (TimesFM methodology)

        From TimesFM paper: "incremental fine-tuning, which allows the model
        to adapt to new financial return data over time"

        Args:
            window_data: Dictionary of stock DataFrames for current window
            window_id: Window identifier

        Returns:
            Training metrics
        """
        self.incremental_step += 1

        if window_id is None:
            window_id = self.incremental_step

        logger.info(f"\n{'='*50}")
        logger.info(f"INCREMENTAL UPDATE {window_id}")
        logger.info(f"{'='*50}")

        # Prepare feature columns
        first_stock = list(window_data.keys())[0]
        feature_cols = self.prepare_feature_columns(window_data[first_stock])

        # Create dataset
        try:
            dataset = TimesFMDataset(
                window_data,
                feature_cols,
                context_length=self.context_length
            )

            if len(dataset) == 0:
                logger.warning(f"Window {window_id}: No valid samples")
                return {'error': 'No valid samples'}

            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0  # Windows compatibility
            )

            logger.info(f"Window {window_id}: {len(dataset)} samples, {len(dataloader)} batches")

        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            return {'error': str(e)}

        # Train single epoch (TimesFM incremental approach)
        try:
            metrics = self._train_single_epoch(dataloader, window_id)

            # Log to MLflow
            self._log_metrics(metrics, window_id)

            # Save model if improvement
            if metrics['loss'] < self.best_loss:
                self.best_loss = metrics['loss']
                self._save_model(f"best_model_step_{self.incremental_step}")
                logger.info(f"✓ New best model: {metrics['loss']:.6f}")

            return metrics

        except Exception as e:
            logger.error(f"Error during training: {e}")
            return {'error': str(e)}

    def _train_single_epoch(self, dataloader: DataLoader, window_id: int) -> Dict:
        """
        Train single epoch on current window (TimesFM methodology)

        From paper: Single epoch per window for incremental adaptation
        """
        if self.model is None:
            self.load_base_model()

        # Initialize optimizer if needed
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate
            )

        self.model.train()

        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_targets = []

        for batch_idx, batch in enumerate(dataloader):
            try:
                context = batch['context'].to(self.device)
                target = batch['target'].to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                if hasattr(self.model, 'forward'):
                    outputs = self.model(context)
                    predictions = outputs.squeeze() if outputs.dim() > 1 else outputs
                else:
                    predictions = self.model(context)

                # Calculate loss
                loss = nn.MSELoss()(predictions, target)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                all_predictions.extend(predictions.cpu().detach().numpy())
                all_targets.extend(target.cpu().detach().numpy())

                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.6f}")

            except Exception as e:
                logger.warning(f"Error in batch {batch_idx}: {e}")
                continue

        if num_batches == 0:
            logger.error("No valid batches processed")
            return {'error': 'No valid batches'}

        avg_loss = total_loss / num_batches

        # Calculate additional metrics
        predictions_np = np.array(all_predictions)
        targets_np = np.array(all_targets)

        mae = np.mean(np.abs(predictions_np - targets_np))
        rmse = np.sqrt(np.mean((predictions_np - targets_np) ** 2))

        metrics = {
            'window_id': window_id,
            'step': self.incremental_step,
            'loss': avg_loss,
            'mae': mae,
            'rmse': rmse,
            'samples': len(all_predictions),
            'batches': num_batches
        }

        logger.info(f"Window {window_id} Results:")
        logger.info(f"  Loss: {avg_loss:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")

        return metrics

    def _log_metrics(self, metrics: Dict, window_id: int):
        """Log metrics to MLflow"""
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{key}_window_{window_id}", value, step=self.incremental_step)
        except Exception as e:
            logger.warning(f"MLflow logging error: {e}")

    def _save_model(self, model_name: str):
        """Save model checkpoint"""
        try:
            models_dir = Path("models/timesfm")
            models_dir.mkdir(parents=True, exist_ok=True)

            save_path = models_dir / model_name

            if hasattr(self.model, 'save_pretrained'):
                self.model.save_pretrained(str(save_path))
            else:
                torch.save(self.model.state_dict(), save_path / "model.pt")

            logger.info(f"Model saved to {save_path}")

        except Exception as e:
            logger.warning(f"Error saving model: {e}")

    def predict(self, context_data: np.ndarray) -> np.ndarray:
        """
        Make predictions on context data

        Args:
            context_data: Context features [sequence_length, features]

        Returns:
            Volatility predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_base_model() first.")

        self.model.eval()

        with torch.no_grad():
            context_tensor = torch.tensor(context_data, dtype=torch.float32).unsqueeze(0).to(self.device)

            if hasattr(self.model, 'forward'):
                output = self.model(context_tensor)
                prediction = output.squeeze().cpu().numpy()
            else:
                prediction = self.model(context_tensor).squeeze().cpu().numpy()

        return prediction


def run_incremental_learning(windows: List[Dict], config_path: str = "configs/config.yaml") -> List[Dict]:
    """
    Run complete incremental learning process on all windows

    Args:
        windows: List of incremental windows
        config_path: Path to configuration file

    Returns:
        List of training metrics for each window
    """
    logger.info("="*50)
    logger.info("TIMESFM INCREMENTAL LEARNING")
    logger.info("="*50)

    learner = TimesFMIncrementalLearner(config_path)

    # Load base model
    learner.load_base_model()

    # Start MLflow run
    mlflow.start_run()

    results = []

    for window in windows:
        window_id = window['window_id']
        window_data = window['data']

        logger.info(f"\nProcessing Window {window_id}:")
        logger.info(f"  Date range: {window['start_date']} to {window['end_date']}")
        logger.info(f"  Stocks: {len(window_data)}")

        # Perform incremental update
        metrics = learner.incremental_update(window_data, window_id)

        if 'error' not in metrics:
            results.append(metrics)
            logger.info(f"✓ Window {window_id} completed successfully")
        else:
            logger.warning(f"✗ Window {window_id} failed: {metrics['error']}")

    # End MLflow run
    mlflow.end_run()

    # Summary
    logger.info("\n" + "="*50)
    logger.info("INCREMENTAL LEARNING SUMMARY")
    logger.info("="*50)
    logger.info(f"Total windows processed: {len(results)}/{len(windows)}")
    logger.info(f"Best loss achieved: {learner.best_loss:.6f}")

    return results


def main():
    """Main execution function for testing"""
    logger.info("Starting TimesFM incremental learning...")

    # This would normally load windows from data preprocessing
    # For now, create a simple test

    learner = TimesFMIncrementalLearner()

    # Load base model
    learner.load_base_model()

    logger.info("TimesFM incremental learner initialized successfully!")
    logger.info(f"Device: {learner.device}")
    logger.info(f"Context length: {learner.context_length}")
    logger.info(f"Window size: {learner.window_size}")


if __name__ == "__main__":
    main()
