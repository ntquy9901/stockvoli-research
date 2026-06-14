"""
TimesFM VN30 Fine-tuning - Official Google Research Approach
Rewritten to follow google-research/timesfm official finetuning methodology
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import yaml
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import random

# Workaround for PyTorch compatibility with Transformers 5.10.2
if not hasattr(torch, 'float8_e8m0fnu'):
    torch.float8_e8m0fnu = torch.float8_e4m3fn

# HuggingFace Transformers
from transformers import TimesFm2_5ModelForPrediction

# Monkey-patch importlib to completely disable bitsandbytes detection
import sys
import types
import importlib.util

# Store original find_spec
original_find_spec = importlib.util.find_spec

# Patch find_spec to return None for bitsandbytes
def patched_find_spec(name, package=None):
    if name and 'bitsandbytes' in name:
        return None  # Report as not found
    return original_find_spec(name, package)

importlib.util.find_spec = patched_find_spec

# Create minimal fake module for safety
fake_bnb = types.ModuleType('bitsandbytes')
fake_bnb.__path__ = []
fake_bnb.__file__ = '<disabled>'
sys.modules['bitsandbytes'] = fake_bnb

# Patch PEFT utilities if they exist
try:
    import peft.import_utils as peft_utils
    peft_utils.is_bnb_available = lambda: False
    if hasattr(peft_utils, 'is_bnb_4bit_available'):
        peft_utils.is_bnb_4bit_available = lambda: False
    if hasattr(peft_utils, 'is_bnb_8bit_available'):
        peft_utils.is_bnb_8bit_available = lambda: False
except:
    pass

# Now import PEFT components
from peft import LoraConfig, get_peft_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments/model_training.log'),
        logging.StreamHandler()
    ]
)


def set_random_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset - Following Google Research structure
# ---------------------------------------------------------------------------

class VN30TimeSeriesDataset(Dataset):
    """
    Simple time series dataset for TimesFM fine-tuning

    Following Google Research approach:
    - Returns (context, target) tuples
    - No complex batch formatting
    - Compatible with TimesFM training API
    """

    def __init__(self, stock_data_dict: Dict[str, pd.DataFrame],
                 context_len: int = 64,
                 horizon_len: int = 1,
                 num_samples: int = 200,
                 mode: str = 'train',
                 seed: int = 42):
        """
        Args:
            stock_data_dict: Dictionary mapping stock names to processed DataFrames
            context_len: Context window length (must be multiple of 32)
            horizon_len: Prediction horizon
            num_samples: Number of samples per stock
            mode: 'train' or 'test'
            seed: Random seed for reproducibility
        """
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.mode = mode

        # Extract RV_20 series from each stock
        self.series_list = []
        for stock_name, df in stock_data_dict.items():
            if 'RV_20' in df.columns:
                series_data = df['RV_20'].dropna().values.astype(np.float32)

                # Ensure we have enough data
                min_len = context_len + horizon_len
                if len(series_data) >= min_len:
                    self.series_list.append({
                        'name': stock_name,
                        'data': series_data
                    })

        logging.info(f"Loaded {len(self.series_list)} stocks for {mode} mode")

        # Generate random samples
        rng = np.random.default_rng(seed)
        self.samples = []

        for series_info in self.series_list:
            series_data = series_info['data']
            data_len = len(series_data)
            min_len = context_len + horizon_len

            # Determine split point
            if mode == 'train':
                max_start = int(data_len * 0.8) - min_len
                if max_start < 10:
                    continue
            else:  # test mode
                min_start = int(data_len * 0.8) + context_len
                max_start = data_len - min_len
                if min_start >= max_start:
                    continue

            # Generate samples
            for _ in range(num_samples):
                start = rng.integers(0, max_start + 1)
                self.samples.append((series_info['name'], start))

        logging.info(f"Generated {len(self.samples)} samples for {mode} mode")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stock_name, start = self.samples[idx]
        series_data = next(s['data'] for s in self.series_list if s['name'] == stock_name)

        end = start + self.context_len + self.horizon_len
        context = series_data[start:start + self.context_len]
        target = series_data[start + self.context_len:end]

        return (
            torch.tensor(context, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )


def create_vn30_dataloaders(config_path: str = 'configs/config.yaml') -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for VN30 stocks

    Args:
        config_path: Path to configuration file

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load processed data
    processed_dir = Path(config['data']['processed_path'])
    processed_files = list(processed_dir.glob("*_processed.csv"))

    logging.info(f"Loading {len(processed_files)} processed stock files")

    # Load all processed data
    stock_data_dict = {}
    for file_path in processed_files:
        stock_name = file_path.stem.replace('_processed', '')
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        stock_data_dict[stock_name] = df

    logging.info(f"Loaded data for {len(stock_data_dict)} stocks")

    # Create datasets
    context_len = config['dataset']['context_length']
    horizon_len = config['dataset'].get('horizon_length', 1)
    samples_per_stock = config['dataset'].get('samples_per_stock', 200)

    train_dataset = VN30TimeSeriesDataset(
        stock_data_dict,
        context_len=context_len,
        horizon_len=horizon_len,
        num_samples=samples_per_stock,
        mode='train',
        seed=42
    )

    test_dataset = VN30TimeSeriesDataset(
        stock_data_dict,
        context_len=context_len,
        horizon_len=horizon_len,
        num_samples=max(50, samples_per_stock // 4),
        mode='test',
        seed=42
    )

    # Create dataloaders
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0  # Windows compatibility
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    logging.info(f"Created dataloaders:")
    logging.info(f"  Training: {len(train_dataset)} samples, {len(train_loader)} batches")
    logging.info(f"  Testing: {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training - Following Google Research approach
# ---------------------------------------------------------------------------

class TimesFMVN30Finetuner:
    """
    TimesFM VN30 Fine-tuner using official Google Research methodology

    Based on: google-research/timesfm/timesfm-forecasting/examples/finetuning/finetune_lora.py
    """

    def __init__(self, config_path: str = 'configs/config.yaml'):
        """
        Initialize TimesFM fine-tuner

        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set random seed
        set_random_seed(self.config['system']['random_seed'])

        # Device
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

    def load_timesfm_model(self) -> None:
        """
        Load TimesFM 2.5 foundation model using HuggingFace Transformers

        Following Google Research approach:
        - Use TimesFm2_5ModelForPrediction.from_pretrained()
        - Apply LoRA adapters for parameter-efficient fine-tuning
        """
        self.logger.info("=" * 70)
        self.logger.info("[LOADING TIMESFM 2.5 FOUNDATION MODEL]")
        self.logger.info("=" * 70)

        try:
            model_id = self.config['model']['model_name']

            # Load model using HuggingFace Transformers
            self.logger.info(f"Loading model: {model_id}")
            self.base_model = TimesFm2_5ModelForPrediction.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map=self.device
            )

            self.logger.info(f"[OK] TimesFM 2.5 loaded successfully")
            self.logger.info(f"Model parameters: {self.config['model']['parameters']}")

            # Get actual context length from model
            model_context_len = self.base_model.config.context_length
            self.logger.info(f"Model context length: {model_context_len}")

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to load TimesFM: {e}")
            raise

    def setup_lora_adapters(self) -> None:
        """
        Configure LoRA adapters for parameter-efficient fine-tuning

        Following Google Research LoRA configuration
        """
        self.logger.info("=" * 70)
        self.logger.info("[CONFIGURING LORA ADAPTERS]")
        self.logger.info("=" * 70)

        try:
            # Get LoRA configuration
            lora_config_dict = self.config['model']['lora']

            # Create LoRA configuration - specific modules to avoid bnb
            lora_config = LoraConfig(
                r=lora_config_dict['r'],                    # Rank: 4
                lora_alpha=lora_config_dict['lora_alpha'],  # Alpha: 8
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention only
                lora_dropout=lora_config_dict.get('lora_dropout', 0.05),
                bias="none",
                task_type="FEATURE_EXTRACTION"  # Avoid quantization paths
            )

            # Apply LoRA adapters
            self.model = get_peft_model(self.base_model, lora_config)
            self.model.print_trainable_parameters()

            self.logger.info(f"[OK] LoRA adapters configured successfully")

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to setup LoRA: {e}")
            raise

    def setup_optimizer(self) -> None:
        """
        Setup optimizer following Google Research approach

        Using AdamW optimizer (as in official example)
        """
        self.logger.info("=" * 70)
        self.logger.info("[SETTING UP OPTIMIZER]")
        self.logger.info("=" * 70)

        try:
            training_config = self.config['training']

            # Use AdamW optimizer (Google Research approach)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=0.01  # As in official example
            )

            # Cosine annealing scheduler
            num_epochs = training_config['num_epochs']
            # Note: will update after creating dataloaders
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,  # Will be updated after we know batch count
                eta_min=training_config.get('min_learning_rate', 1e-6)
            )

            self.logger.info(f"Optimizer: AdamW (lr={training_config['learning_rate']:.6f})")
            self.logger.info(f"Weight decay: 0.01")
            self.logger.info(f"Scheduler: Cosine Annealing")

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to setup optimizer: {e}")
            raise

    def train_one_epoch(self, train_loader: DataLoader,
                       epoch: int, num_epochs: int) -> Dict[str, float]:
        """
        Train one epoch following Google Research methodology

        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            num_epochs: Total number of epochs

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        context_len = self.config['dataset']['context_length']

        for context, target_vals in train_loader:
            context = context.to(self.device)
            target_vals = target_vals.to(self.device)

            try:
                # Forward pass using Google Research API
                outputs = self.model(
                    past_values=context,
                    future_values=target_vals,
                    forecast_context_len=context_len,
                )

                # Access loss from outputs
                loss = outputs.loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()

                # Metrics
                total_loss += loss.item()
                n_batches += 1

                # Log progress every 10 batches
                if (n_batches + 1) % 10 == 0:
                    avg_loss = total_loss / n_batches
                    self.logger.info(
                        f"Epoch {epoch+1} | Batch {n_batches+1}/{len(train_loader)} | "
                        f"Loss: {avg_loss:.4f}"
                    )

            except Exception as e:
                self.logger.error(f"Training error at batch {n_batches}: {e}")
                continue

        avg_train_loss = total_loss / max(n_batches, 1)

        return {
            'loss': avg_train_loss,
            'num_batches': n_batches
        }

    def validate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model on test set

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        val_batches = 0

        context_len = self.config['dataset']['context_length']

        with torch.no_grad():
            for context, target_vals in test_loader:
                context = context.to(self.device)
                target_vals = target_vals.to(self.device)

                try:
                    # Forward pass
                    outputs = self.model(
                        past_values=context,
                        future_values=target_vals,
                        forecast_context_len=context_len,
                    )

                    val_loss += outputs.loss.item()
                    val_batches += 1

                except Exception as e:
                    self.logger.error(f"Validation error: {e}")
                    continue

        avg_val_loss = val_loss / max(val_batches, 1)

        return {
            'val_loss': avg_val_loss,
            'num_batches': val_batches
        }

    def train_model(self, train_loader: DataLoader,
                   test_loader: DataLoader) -> None:
        """
        Complete training loop following Google Research methodology

        Args:
            train_loader: Training data loader
            test_loader: Test data loader
        """
        self.logger.info("=" * 70)
        self.logger.info("[STARTING TIMESFM FINE-TUNING]")
        self.logger.info("=" * 70)

        # Update scheduler T_max with actual batch count
        num_epochs = self.config['training']['num_epochs']
        self.scheduler.T_max = num_epochs * len(train_loader)

        self.logger.info(f"Training configuration:")
        self.logger.info(f"  Epochs: {num_epochs}")
        self.logger.info(f"  Train batches: {len(train_loader)}")
        self.logger.info(f"  Test batches: {len(test_loader)}")
        self.logger.info(f"  Batch size: {self.config['training']['batch_size']}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_metrics = self.train_one_epoch(train_loader, epoch, num_epochs)

            # Validate
            val_metrics = self.validate_model(test_loader)

            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log epoch results
            self.logger.info("=" * 70)
            self.logger.info(f"EPOCH {epoch+1}/{num_epochs} COMPLETE")
            self.logger.info("=" * 70)
            self.logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            self.logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
            self.logger.info(f"Learning Rate: {current_lr:.8f}")

            # Store training history
            epoch_log = {
                'epoch': epoch + 1,
                'timestamp': datetime.now().isoformat(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'learning_rate': current_lr
            }
            self.training_history.append(epoch_log)

            # Save best model
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint('best_model')
                self.logger.info(f"[NEW BEST] Val loss = {val_metrics['val_loss']:.4f}")

            # Periodic checkpoint
            if (epoch + 1) % self.config['training']['save_every_n_epochs'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}')
                self.logger.info(f"[CHECKPOINT] Saved epoch {epoch+1}")

            # Save training history
            self.save_training_history()

    def save_checkpoint(self, model_name: str) -> None:
        """
        Save LoRA adapter checkpoint

        Args:
            model_name: Name for the checkpoint
        """
        output_dir = Path(self.config['output']['checkpoints_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = output_dir / model_name

        # Save LoRA adapter (not full model)
        self.model.save_pretrained(str(checkpoint_path))

        self.logger.info(f"[CHECKPOINT] Saved: {checkpoint_path}")

    def save_training_history(self) -> None:
        """Save training history to JSON"""
        experiments_dir = Path(self.config['experiment_tracking']['experiments_dir'])
        experiments_dir.mkdir(parents=True, exist_ok=True)

        history_path = experiments_dir / 'training_history.json'

        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)


def main():
    """Main execution function"""

    # Initialize finetuner
    finetuner = TimesFMVN30Finetuner()

    # Load TimesFM model
    finetuner.load_timesfm_model()

    # Setup LoRA adapters
    finetuner.setup_lora_adapters()

    # Setup optimizer
    finetuner.setup_optimizer()

    # Load data
    logging.info("Loading VN30 dataloaders...")
    train_loader, test_loader = create_vn30_dataloaders()

    # Train model
    finetuner.train_model(train_loader, test_loader)

    # Final summary
    logging.info("=" * 70)
    logging.info("[TRAINING COMPLETE]")
    logging.info("=" * 70)
    logging.info(f"Best val loss: {finetuner.best_val_loss:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())