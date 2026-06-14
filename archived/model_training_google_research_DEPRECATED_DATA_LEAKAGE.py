"""
TimesFM VN30 Fine-tuning - Official Google Research Methodology
Following EXACTLY: google-research/timesfm/timesfm-forecasting/examples/finetuning/finetune_lora.py
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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environment
import matplotlib.pyplot as plt

# Workaround for PyTorch compatibility with Transformers 5.10.2
if not hasattr(torch, 'float8_e8m0fnu'):
    torch.float8_e8m0fnu = torch.float8_e4m3fn

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

# Google Research imports - EXACT MATCH
from transformers import TimesFm2_5ModelForPrediction  # TimesFm2_5, not TimesFm
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

logger = logging.getLogger(__name__)


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
# Dataset - EXACT Google Research Methodology
# ---------------------------------------------------------------------------

class VN30TimeSeriesDataset(Dataset):
    """
    Random-window dataset for time series fine-tuning.

    Following EXACT Google Research methodology from finetune_lora.py:
    - Pre-samples random (series, split-point) windows
    - Each window has full context_len context (no zero-padding)
    - Returns TUPLE (context, target) NOT dictionary
    """

    def __init__(
        self,
        series_list: list[np.ndarray],
        context_len: int,
        horizon_len: int,
        num_samples: int = 5000,
        seed: int = 42,
    ):
        """
        Args:
            series_list: List of numpy arrays, one per time series
            context_len: Context length for training windows
            horizon_len: Forecast horizon
            num_samples: Number of random windows to pre-sample
            seed: Random seed
        """
        self.series_list = series_list
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.samples: list[tuple[int, int]] = []

        rng = np.random.default_rng(seed)
        min_len = context_len + horizon_len
        valid = [i for i, s in enumerate(series_list) if len(s) >= min_len]

        if not valid:
            raise ValueError(
                f"No series long enough for context_len={context_len} + "
                f"horizon_len={horizon_len}. Shortest series: "
                f"{min(len(s) for s in series_list)}"
            )

        logger.info(f"Valid series: {len(valid)} (need >= {min_len} data points)")

        # Pre-sample random windows
        for _ in range(num_samples):
            idx = rng.choice(valid)
            series = series_list[idx]
            max_start = len(series) - min_len
            start = rng.integers(0, max_start + 1)
            self.samples.append((idx, start))

        logger.info(f"Pre-sampled {len(self.samples)} random windows")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a tuple (context, target) following Google Research format

        CRITICAL: Returns TUPLE not dictionary
        """
        idx, start = self.samples[i]
        series = self.series_list[idx]
        end = start + self.context_len + self.horizon_len

        context = torch.tensor(
            series[start : start + self.context_len], dtype=torch.float32
        )
        target = torch.tensor(
            series[start + self.context_len : end], dtype=torch.float32
        )

        return context, target  # TUPLE format


class VN30LastWindowDataset(Dataset):
    """
    Validation dataset using the last window of each series.

    Following Google Research methodology for validation
    """

    def __init__(
        self,
        series_list: list[np.ndarray],
        context_len: int,
        horizon_len: int,
    ):
        self.items: list[tuple[torch.Tensor, torch.Tensor]] = []
        min_len = context_len + horizon_len

        for s in series_list:
            if len(s) >= min_len:
                ctx = torch.tensor(s[-min_len:-horizon_len], dtype=torch.float32)
                tgt = torch.tensor(s[-horizon_len:], dtype=torch.float32)
                self.items.append((ctx, tgt))

        logger.info(f"Validation dataset: {len(self.items)} series")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.items[i]


def load_vn30_data(config_path: str) -> tuple[list[np.ndarray], int, int]:
    """
    Load VN30 stock data and prepare series list

    Returns:
        series_list: List of RV_20 numpy arrays
        context_len: Context length (limited by model config)
        horizon_len: Forecast horizon
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load processed data
    processed_dir = Path(config['data']['processed_path'])
    processed_files = list(processed_dir.glob("*_processed.csv"))

    logger.info(f"Loading {len(processed_files)} processed stock files")

    # Load all processed data
    stock_data_dict = {}
    for file_path in processed_files:
        stock_name = file_path.stem.replace('_processed', '')
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        stock_data_dict[stock_name] = df

    logger.info(f"Loaded data for {len(stock_data_dict)} stocks")

    # Extract RV_20 series from each stock
    all_series: list[np.ndarray] = []
    for stock_name, df in stock_data_dict.items():
        if 'RV_20' in df.columns:
            values = df['RV_20'].dropna().values.astype(np.float32)
            if len(values) > 0:  # Only add if has data
                all_series.append(values)

    logger.info(f"Valid stocks with RV_20 data: {len(all_series)}")

    # Get context and horizon lengths from config
    context_len = config['dataset']['context_length']
    horizon_len = config['dataset'].get('horizon_length', 1)

    return all_series, context_len, horizon_len


def create_vn30_dataloaders(
    config_path: str = 'configs/config.yaml',
    context_len: int = None,
    horizon_len: int = None
) -> tuple[DataLoader, DataLoader, int, int]:
    """
    Create train and test dataloaders for VN30 stocks

    Args:
        config_path: Path to configuration file
        context_len: Override context length
        horizon_len: Override horizon length

    Returns:
        (train_loader, test_loader, context_len, horizon_len)
    """
    # Load data
    all_series, config_context_len, config_horizon_len = load_vn30_data(config_path)

    # Use provided values or config defaults
    if context_len is None:
        context_len = config_context_len
    if horizon_len is None:
        horizon_len = config_horizon_len

    # Get num_samples from config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    num_samples = config['dataset'].get('samples_per_stock', 200) * len(all_series)
    batch_size = config['training']['batch_size']

    # Create datasets following Google Research methodology
    train_ds = VN30TimeSeriesDataset(
        all_series,
        context_len=context_len,
        horizon_len=horizon_len,
        num_samples=num_samples,
        seed=42
    )

    val_ds = VN30LastWindowDataset(
        all_series,
        context_len=context_len,
        horizon_len=horizon_len
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True  # Google Research uses drop_last
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False
    )

    logger.info(f"Train samples: {len(train_ds)} ({len(train_loader)} batches)")
    logger.info(f"Val samples: {len(val_ds)} ({len(val_loader)} batches)")

    return train_loader, val_loader, context_len, horizon_len


# ---------------------------------------------------------------------------
# Training - EXACT Google Research Methodology
# ---------------------------------------------------------------------------

class TimesFMVN30Finetuner:
    """
    TimesFM VN30 Fine-tuner following EXACT Google Research methodology

    Based on: google-research/timesfm/timesfm-forecasting/examples/finetuning/finetune_lora.py
    """

    def __init__(self, config_path: str = 'configs/config.yaml'):
        """
        Initialize TimesFM fine-tuner

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set random seed
        set_random_seed(self.config['system']['random_seed'])

        # Device
        self.device = torch.device(self.config['system']['device'])
        logger.info(f"Using device: {self.device}")

        # Model storage
        self.model = None
        self.context_len = None
        self.horizon_len = None
        self.best_val_loss = float('inf')
        self.patience_counter = 0  # Early stopping counter

        # Training metrics
        self.training_history = []

    def load_timesfm_model(self) -> None:
        """
        Load TimesFM 2.5 model using EXACT Google Research approach

        From finetune_lora.py:
        model = TimesFm2_5ModelForPrediction.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        """
        logger.info("=" * 70)
        logger.info("[LOADING TIMESFM 2.5 MODEL]")
        logger.info("=" * 70)

        try:
            model_id = self.config['model']['model_name']

            logger.info(f"Loading model: {model_id}")
            self.model = TimesFm2_5ModelForPrediction.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,  # Google Research uses bfloat16
                device_map=self.device
            )

            logger.info(f"[OK] TimesFM 2.5 loaded successfully")

            # IMPORTANT: Handle context_len - Google Research ensures it doesn't exceed model limit
            config_context_len = self.config['dataset']['context_length']
            self.horizon_len = self.config['dataset'].get('horizon_length', 1)

            # Google Research: context_len = min(args.context_len, model.config.context_length)
            self.context_len = min(config_context_len, self.model.config.context_length)

            logger.info(f"Config context length: {config_context_len}")
            logger.info(f"Model context length: {self.model.config.context_length}")
            logger.info(f"Actual context length: {self.context_len}")
            logger.info(f"Horizon length: {self.horizon_len}")

        except Exception as e:
            logger.error(f"[FAIL] Failed to load TimesFM: {e}")
            raise

    def setup_lora_adapters(self) -> None:
        """
        Configure LoRA adapters using EXACT Google Research configuration

        From finetune_lora.py:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules="all-linear",
            lora_dropout=args.lora_dropout,
            bias="none",
        )
        """
        logger.info("=" * 70)
        logger.info("[CONFIGURING LORA ADAPTERS]")
        logger.info("=" * 70)

        try:
            # Get LoRA configuration from config
            lora_config_dict = self.config['model']['lora']

            # Google Research configuration
            lora_config = LoraConfig(
                r=lora_config_dict['r'],
                lora_alpha=lora_config_dict['lora_alpha'],
                target_modules="all-linear",  # Google Research uses "all-linear"
                lora_dropout=lora_config_dict.get('lora_dropout', 0.05),
                bias="none"
            )

            # Apply LoRA adapters
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

            logger.info(f"[OK] LoRA adapters configured successfully")

        except Exception as e:
            logger.error(f"[FAIL] Failed to setup LoRA: {e}")
            raise

    def setup_optimizer(self, num_training_steps: int) -> None:
        """
        Setup optimizer and scheduler using Google Research approach

        From finetune_lora.py:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs * len(train_loader)
        )
        """
        logger.info("=" * 70)
        logger.info("[SETTING UP OPTIMIZER & SCHEDULER]")
        logger.info("=" * 70)

        try:
            training_config = self.config['training']

            # Google Research: AdamW with weight_decay=0.01
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=0.01  # Google Research uses 0.01
            )

            # Cosine annealing scheduler (Google Research approach)
            num_epochs = training_config['num_epochs']
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs * num_training_steps  # T_max = epochs * steps_per_epoch
            )

            logger.info(f"Optimizer: AdamW (lr={training_config['learning_rate']}, weight_decay=0.01)")
            logger.info(f"Scheduler: CosineAnnealingLR (T_max={num_epochs * num_training_steps})")

        except Exception as e:
            logger.error(f"[FAIL] Failed to setup optimizer: {e}")
            raise

    def train_one_epoch(self, train_loader: DataLoader, epoch: int, num_epochs: int) -> Dict[str, float]:
        """
        Train one epoch following EXACT Google Research methodology

        From finetune_lora.py:
        for context, target_vals in train_loader:
            context = context.to(device)
            target_vals = target_vals.to(device)

            outputs = model(
                past_values=context,
                future_values=target_vals,
                forecast_context_len=context_len,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        """
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0

        for context, target_vals in train_loader:
            # Google Research: Unpack tuple directly
            context = context.to(self.device)
            target_vals = target_vals.to(self.device)

            try:
                # Google Research forward pass - EXACT 3 parameters
                outputs = self.model(
                    past_values=context,
                    future_values=target_vals,
                    forecast_context_len=self.context_len,  # CRITICAL PARAMETER
                )

                loss = outputs.loss

                # Google Research backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

                # Metrics
                epoch_loss += loss.item()
                n_batches += 1

                # Log progress every 10 batches
                if n_batches % 10 == 0:
                    avg_loss = epoch_loss / n_batches
                    logger.info(
                        f"Epoch {epoch+1} | Batch {n_batches}/{len(train_loader)} | Loss: {avg_loss:.4f}"
                    )

            except Exception as e:
                logger.error(f"Training error at batch {n_batches}: {e}")
                continue

        avg_train_loss = epoch_loss / max(n_batches, 1)

        return {
            'loss': avg_train_loss,
            'num_batches': n_batches
        }

    def validate_model(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model on test set following Google Research methodology

        From finetune_lora.py:
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for context, target_vals in val_loader:
                context = context.to(device)
                target_vals = target_vals.to(device)
                outputs = model(
                    past_values=context,
                    future_values=target_vals,
                    forecast_context_len=context_len,
                )
                val_loss += outputs.loss.item()
                val_batches += 1
        """
        self.model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for context, target_vals in val_loader:
                context = context.to(self.device)
                target_vals = target_vals.to(self.device)

                try:
                    # Google Research validation forward pass
                    outputs = self.model(
                        past_values=context,
                        future_values=target_vals,
                        forecast_context_len=self.context_len,
                    )

                    val_loss += outputs.loss.item()
                    val_batches += 1

                except Exception as e:
                    logger.error(f"Validation error: {e}")
                    continue

        avg_val_loss = val_loss / max(val_batches, 1)

        return {
            'val_loss': avg_val_loss,
            'num_batches': val_batches
        }

    def train_model(self, train_loader: DataLoader = None,
                   val_loader: DataLoader = None) -> None:
        """
        Complete training loop following Google Research methodology

        Args:
            train_loader: Training data loader (optional)
            val_loader: Validation data loader (optional)
        """
        logger.info("=" * 70)
        logger.info("[STARTING TIMESFM FINE-TUNING]")
        logger.info("=" * 70)

        # Create dataloaders if not provided
        if train_loader is None or val_loader is None:
            train_loader, val_loader, _, _ = create_vn30_dataloaders(
                self.config_path,
                context_len=self.context_len,
                horizon_len=self.horizon_len
            )

        num_epochs = self.config['training']['num_epochs']

        # Setup optimizer and scheduler (need num_training_steps)
        self.setup_optimizer(num_epochs * len(train_loader))

        logger.info(f"Training configuration:")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")
        logger.info(f"  Batch size: {self.config['training']['batch_size']}")
        logger.info(f"  Context length: {self.context_len}")
        logger.info(f"  Horizon length: {self.horizon_len}")

        # Training loop following Google Research pattern
        for epoch in range(num_epochs):
            # Train one epoch
            train_metrics = self.train_one_epoch(train_loader, epoch, num_epochs)

            # Validate
            val_metrics = self.validate_model(val_loader)

            # Log epoch results (Google Research format)
            logger.info("=" * 70)
            logger.info(f"Epoch {epoch+1}/{num_epochs} ({train_metrics['num_batches']} steps)")
            logger.info(f"Train loss: {train_metrics['loss']:.4f}, Val loss: {val_metrics['val_loss']:.4f}")
            logger.info("=" * 70)

            # Store training history
            epoch_log = {
                'epoch': epoch + 1,
                'timestamp': datetime.now().isoformat(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'learning_rate': self.optimizer.param_groups[0]['lr']  # Track learning rate
            }
            self.training_history.append(epoch_log)

            # Save best model (Google Research approach)
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint()
                logger.info(f"  [BEST] Saved adapter → {self.config['output']['checkpoints_dir']}")

            # Periodic checkpoint
            if (epoch + 1) % self.config['training']['save_every_n_epochs'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}')
                logger.info(f"  [CHECKPOINT] Saved epoch {epoch+1}")

            # Save training history
            self.save_training_history()

            # Update learning curves (REAL-TIME VISUALIZATION)
            self.plot_learning_curves()

            # Early stopping check
            patience = self.config['training'].get('early_stopping_patience', 5)
            if val_metrics['val_loss'] < self.best_val_loss:
                self.patience_counter = 0  # Reset counter on improvement
            else:
                self.patience_counter += 1
                logger.info(f"  No improvement for {self.patience_counter} epoch(s) (patience={patience})")

                if self.patience_counter >= patience:
                    logger.info(f"  [EARLY STOPPING] No improvement for {patience} consecutive epochs")
                    logger.info(f"  Stopping at epoch {epoch+1}/{num_epochs}")
                    break

        logger.info("=" * 70)
        logger.info("[TRAINING COMPLETE]")
        logger.info("=" * 70)
        logger.info(f"Best val loss: {self.best_val_loss:.4f}")
        logger.info(f"Total epochs trained: {len(self.training_history)}")

        # Create final training summary plot
        self.create_training_summary()

    def save_checkpoint(self, model_name: str = None) -> None:
        """
        Save LoRA adapter checkpoint

        Google Research: model.save_pretrained(args.output_dir)
        """
        output_dir = Path(self.config['output']['checkpoints_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        if model_name is None:
            checkpoint_path = output_dir
        else:
            checkpoint_path = output_dir / model_name

        # Save LoRA adapter (Google Research approach)
        self.model.save_pretrained(str(checkpoint_path))

        logger.info(f"[CHECKPOINT] Saved: {checkpoint_path}")

    def create_training_summary(self) -> None:
        """
        Create comprehensive training summary with plots and statistics

        Generates:
        1. Final learning curves
        2. Training statistics summary
        3. Best model information
        """
        experiments_dir = Path(self.config['experiment_tracking']['experiments_dir'])
        experiments_dir.mkdir(parents=True, exist_ok=True)

        if not self.training_history:
            logger.warning("No training history to summarize")
            return

        # Extract data
        epochs = [entry['epoch'] for entry in self.training_history]
        train_losses = [entry['train_metrics']['loss'] for entry in self.training_history]
        val_losses = [entry['val_metrics']['val_loss'] for entry in self.training_history]
        learning_rates = [entry.get('learning_rate', 0) for entry in self.training_history]

        # Find best epoch
        best_val_idx = val_losses.index(min(val_losses))
        best_epoch = epochs[best_val_idx]
        best_train_loss = train_losses[best_val_idx]
        best_val_loss = val_losses[best_val_idx]

        # Create comprehensive summary plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('TimesFM Fine-tuning Training Summary', fontsize=16, fontweight='bold')

        # Plot 1: Loss curves
        ax = axes[0, 0]
        ax.plot(epochs, train_losses, label='Train Loss', marker='o', linewidth=2)
        ax.plot(epochs, val_losses, label='Val Loss', marker='s', linewidth=2, color='orange')
        ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch: {best_epoch}')
        ax.axhline(y=best_val_loss, color='red', linestyle='--', alpha=0.3, label=f'Best Val: {best_val_loss:.4f}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training & Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Loss ratio (Val/Train)
        ax = axes[0, 1]
        ratios = [v/t if t > 0 else 0 for v, t in zip(val_losses, train_losses)]
        ax.plot(epochs, ratios, marker='o', linewidth=2, color='purple')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Ratio=1.0')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val Loss / Train Loss')
        ax.set_title('Overfitting Indicator (Ratio > 1 = Overfitting)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.fill_between(epochs, 1, ratios, where=[r > 1 for r in ratios], alpha=0.3, color='red', label='Overfitting Zone')

        # Plot 3: Learning rate schedule
        ax = axes[1, 0]
        ax.plot(epochs, learning_rates, marker='^', linewidth=2, color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Plot 4: Training statistics table
        ax = axes[1, 1]
        ax.axis('off')

        # Calculate statistics
        total_epochs = len(epochs)
        initial_train_loss = train_losses[0]
        final_train_loss = train_losses[-1]
        train_improvement = ((initial_train_loss - final_train_loss) / initial_train_loss) * 100

        initial_val_loss = val_losses[0]
        final_val_loss = val_losses[-1]
        val_improvement = ((initial_val_loss - final_val_loss) / initial_val_loss) * 100

        # Create summary table
        summary_text = f"""
TRAINING SUMMARY
{'=' * 50}
Total Epochs:        {total_epochs}
Best Epoch:          {best_epoch}
Best Val Loss:       {best_val_loss:.4f}
Train Loss @ Best:   {best_train_loss:.4f}

LOSS IMPROVEMENT:
Train Loss:          {initial_train_loss:.4f} → {final_train_loss:.4f} ({train_improvement:.1f}%)
Val Loss:            {initial_val_loss:.4f} → {final_val_loss:.4f} ({val_improvement:.1f}%)

CONFIGURATION:
Model:               {self.config['model']['model_name']}
Context Length:       {self.context_len}
Horizon Length:       {self.horizon_len}
Batch Size:          {self.config['training']['batch_size']}
Learning Rate:        {self.config['training']['learning_rate']}

LORA CONFIGURATION:
Rank (r):             {self.config['model']['lora']['r']}
Alpha:                {self.config['model']['lora']['lora_alpha']}
Dropout:              {self.config['model']['lora'].get('lora_dropout', 0.05)}
"""

        ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax.transAxes)

        plt.tight_layout()

        # Save summary
        summary_path = experiments_dir / 'training_summary.png'
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  [SUMMARY] Training summary saved: {summary_path}")

        # Also save text summary
        text_summary_path = experiments_dir / 'training_summary.txt'
        with open(text_summary_path, 'w') as f:
            f.write(summary_text)
            f.write(f"\nGenerated: {datetime.now().isoformat()}\n")

        logger.info(f"  [SUMMARY] Text summary saved: {text_summary_path}")

    def save_training_history(self) -> None:
        """Save training history to JSON"""
        experiments_dir = Path(self.config['experiment_tracking']['experiments_dir'])
        experiments_dir.mkdir(parents=True, exist_ok=True)

        history_path = experiments_dir / 'training_history.json'

        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

    def plot_learning_curves(self) -> None:
        """
        Plot and save learning curves after each epoch

        Creates:
        1. Loss curves (train & val)
        2. Learning rate curve
        3. Saves to experiments directory
        """
        if not self.training_history:
            return

        experiments_dir = Path(self.config['experiment_tracking']['experiments_dir'])
        experiments_dir.mkdir(parents=True, exist_ok=True)

        # Extract data
        epochs = [entry['epoch'] for entry in self.training_history]
        train_losses = [entry['train_metrics']['loss'] for entry in self.training_history]
        val_losses = [entry['val_metrics']['val_loss'] for entry in self.training_history]

        # Get learning rates if available
        learning_rates = []
        for entry in self.training_history:
            if 'learning_rate' in entry:
                learning_rates.append(entry['learning_rate'])
            else:
                learning_rates.append(None)

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot 1: Loss curves
        ax1.plot(epochs, train_losses, label='Train Loss', marker='o', linewidth=2, markersize=4)
        ax1.plot(epochs, val_losses, label='Val Loss', marker='s', linewidth=2, markersize=4, color='orange')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)

        # Add best val loss marker
        if val_losses:
            best_val_idx = val_losses.index(min(val_losses))
            best_val_epoch = epochs[best_val_idx]
            best_val_loss = val_losses[best_val_idx]
            ax1.axhline(y=best_val_loss, color='green', linestyle='--', alpha=0.5, label=f'Best: {best_val_loss:.4f}')
            ax1.plot(best_val_epoch, best_val_loss, 'g*', markersize=15, label=f'Epoch {best_val_epoch}')
            ax1.legend(fontsize=9)

        # Plot 2: Learning Rate (if available)
        if learning_rates and all(lr is not None for lr in learning_rates):
            ax2.plot(epochs, learning_rates, label='Learning Rate', marker='^', linewidth=2, markersize=4, color='red')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Learning Rate', fontsize=12)
            ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(left=0)
            ax2.set_yscale('log')
        else:
            # Hide second plot if no LR data
            ax2.axis('off')
            fig.delaxes(ax2)
            ax1 = fig.axes[0]

        # Adjust layout
        plt.tight_layout()

        # Save plot
        curve_path = experiments_dir / 'learning_curves.png'
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  [PLOT] Learning curves updated: {curve_path}")


def main():
    """Main execution function"""

    # Initialize finetuner
    finetuner = TimesFMVN30Finetuner()

    # Load TimesFM model
    finetuner.load_timesfm_model()

    # Setup LoRA adapters
    finetuner.setup_lora_adapters()

    # Train model (dataloaders created internally)
    finetuner.train_model()

    return 0


if __name__ == "__main__":
    sys.exit(main())
