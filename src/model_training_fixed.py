"""
TimesFM VN30 Fine-tuning - Official Tutorial Approach
Following Niels Rogge's official Transformers tutorial methodology
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
import os  # For CPU count optimization
import time  # For performance timing

# For learning curve visualization
import matplotlib
# Use appropriate backend for Colab (interactive) vs server (non-interactive)
try:
    import google.colab
    # In Colab - use inline for real-time visualization
    matplotlib.use('Agg')  # Still use Agg, but we'll display manually
    IN_COLAB = True
except:
    # Not in Colab - use non-interactive backend
    matplotlib.use('Agg')
    IN_COLAB = False
import matplotlib.pyplot as plt

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

# CRITICAL: Set CUDA memory management BEFORE any PyTorch operations
# This prevents memory fragmentation on 8GB GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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

# Set up logging - Create experiments directory first
Path('experiments').mkdir(exist_ok=True)
Path('models').mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments/model_training.log', encoding='utf-8'),
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
# Dataset - Following Tutorial Methodology
# ---------------------------------------------------------------------------

class VN30TimeSeriesDataset(Dataset):
    """
    Improved Time Series Dataset for TimesFM Fine-tuning with Proper Temporal Split

    Key Features:
    - PROPER temporal split: Train (first 80%), Test (last 20%)
    - NO data leakage: Training and test samples are time-separated
    - Increased test samples: 100 per stock (3000 total) for robust validation
    - Deterministic sampling with seed for reproducibility

    Temporal Split Logic:
    - Train mode: Samples from [0, 0.8*N - context - horizon] only
    - Test mode: Samples from [0.8*N, N] only
    - Guarantees: max(train_start) < min(test_start) - NO overlap
    """

    def __init__(self, stock_data_dict: Dict[str, pd.DataFrame],
                 context_len: int = 64,
                 horizon_len: int = 1,
                 num_samples: int = 200,
                 mode: str = 'train',
                 seed: int = 42,
                 test_samples_per_stock: int = 30):
        """
        Args:
            stock_data_dict: Dictionary mapping stock names to processed DataFrames
            context_len: Context window length (default: 128 for TimesFM 2.5)
            horizon_len: Prediction horizon (default: 1 day ahead)
            num_samples: Number of samples per stock for training
            mode: 'train' or 'test' - determines temporal split
            seed: Random seed for reproducibility
            test_samples_per_stock: Number of test samples per stock (default: 30, balanced for ~6.7:1 train:test ratio)
        """
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.mode = mode
        self.test_samples_per_stock = test_samples_per_stock

        # Extract RV_20 series from each stock with metadata
        self.series_list = []
        for stock_name, df in stock_data_dict.items():
            if 'RV_20' in df.columns:
                series_data = df['RV_20'].dropna().values.astype(np.float32)
                dates = df.index[~df['RV_20'].isna()].values  # Keep dates for tracking

                # Ensure we have enough data
                min_len = context_len + horizon_len
                if len(series_data) >= min_len:
                    self.series_list.append({
                        'name': stock_name,
                        'data': series_data,
                        'dates': dates,
                        'data_len': len(series_data)
                    })

        logging.info(f"Loaded {len(self.series_list)} stocks for {mode} mode")

        # Track temporal split info for validation
        split_info = []
        all_start_positions = []

        # Generate random samples with proper temporal split
        rng = np.random.default_rng(seed)
        self.samples = []

        for series_info in self.series_list:
            series_data = series_info['data']
            data_len = len(series_data)
            min_len = context_len + horizon_len
            stock_name = series_info['name']

            # Determine split point based on mode
            if mode == 'train':
                max_start = int(data_len * 0.8) - min_len
                if max_start < 10:
                    logging.warning(f"{stock_name}: Insufficient train data ({max_start} < 10)")
                    continue

                # Generate training samples from FIRST 80% only
                for _ in range(num_samples):
                    start = rng.integers(0, max_start + 1)
                    self.samples.append((stock_name, start))
                    all_start_positions.append(start)

                split_info.append({
                    'stock': stock_name,
                    'train_max_start': max_start,
                    'split_point': int(data_len * 0.8)
                })

            else:  # test mode
                min_start = int(data_len * 0.8) + context_len
                max_start = data_len - min_len
                if min_start >= max_start:
                    logging.warning(f"{stock_name}: Insufficient test data")
                    continue

                # Generate test samples from LAST 20% only
                # Increased samples for robust validation
                for _ in range(test_samples_per_stock):
                    start = rng.integers(min_start, max_start + 1)
                    self.samples.append((stock_name, start))
                    all_start_positions.append(start)

                split_info.append({
                    'stock': stock_name,
                    'test_min_start': min_start,
                    'split_point': int(data_len * 0.8)
                })

        logging.info(f"Generated {len(self.samples)} samples for {mode} mode")

        # Log temporal split validation
        if mode == 'train' and all_start_positions:
            logging.info(f"Train samples range: [{min(all_start_positions)}, {max(all_start_positions)}]")
        elif mode == 'test' and all_start_positions:
            logging.info(f"Test samples range: [{min(all_start_positions)}, {max(all_start_positions)}]")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stock_name, start = self.samples[idx]
        series_data = next(s['data'] for s in self.series_list if s['name'] == stock_name)

        end = start + self.context_len + self.horizon_len
        context = series_data[start:start + self.context_len]
        ground_truth = series_data[start + self.context_len:end]

        # Return dictionary format as per tutorial
        return {
            'context': torch.tensor(context, dtype=torch.float32),
            'ground_truth': torch.tensor(ground_truth, dtype=torch.float32)
        }


def create_vn30_dataloaders(config_path: str = 'configs/config.yaml',
                             context_len: int = None,
                             horizon_len: int = None,
                             test_samples_per_stock: int = 30) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for VN30 stocks with proper temporal split

    Args:
        config_path: Path to configuration file
        context_len: Override context length (use model's actual length)
        horizon_len: Override horizon length (use model's actual length)
        test_samples_per_stock: Number of test samples per stock (default: 30 for balanced train:test ratio)

    Returns:
        Tuple of (train_loader, test_loader)

    Note:
        - Train samples: from first 80% of data only
        - Test samples: from last 20% of data only
        - NO temporal overlap between train and test
        - Increased test samples for more reliable validation
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

    # Use model's actual dimensions if provided, otherwise use config
    if context_len is None:
        context_len = config['dataset']['context_length']
    if horizon_len is None:
        horizon_len = config['dataset'].get('horizon_length', 1)

    samples_per_stock = config['dataset'].get('samples_per_stock', 200)

    logging.info(f"Creating datasets with context_len={context_len}, horizon_len={horizon_len}")
    logging.info(f"Train samples: {samples_per_stock} per stock")
    logging.info(f"Test samples: {test_samples_per_stock} per stock (increased for robust validation)")

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
        num_samples=0,  # Not used for test mode
        mode='test',
        seed=42,
        test_samples_per_stock=test_samples_per_stock  # Use dedicated test samples parameter
    )

    # Create dataloaders with OPTIMIZED settings for faster training
    batch_size = config['training']['batch_size']

    # OPTIMIZATION: Use num_workers from config or default to 4 for parallel data loading
    num_workers = config['training'].get('num_workers', min(4, os.cpu_count() or 1))
    persistent_workers = num_workers > 0  # Keep workers alive between epochs

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True if torch.cuda.is_available() else False,  # Faster CPU->GPU transfer
        prefetch_factor=2 if num_workers > 0 else None  # Prefetch 2 batches per worker
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    logging.info(f"Created dataloaders:")
    logging.info(f"  Training: {len(train_dataset)} samples, {len(train_loader)} batches")
    logging.info(f"  Testing: {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training - Following Tutorial Methodology
# ---------------------------------------------------------------------------

class TimesFMVN30Finetuner:
    """
    TimesFM VN30 Fine-tuner following official tutorial methodology

    Based on: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TimesFM/Fine_tune_TimesFM_on_a_custom_dataset.ipynb
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

        # Early stopping
        self.epochs_since_improvement = 0
        self.early_stopping_patience = self.config['training'].get('early_stopping_patience', 5)

        # Verify GPU setup immediately after initialization
        self.verify_gpu_setup()

    def verify_gpu_setup(self) -> None:
        """
        Comprehensive GPU verification and diagnostics

        Checks:
        - CUDA availability
        - GPU device name and capabilities
        - Memory (total, free, allocated)
        - bfloat16 support
        - Current device assignment

        Raises:
            RuntimeError: If GPU verification fails or configuration is invalid
        """
        self.logger.info("=" * 70)
        self.logger.info("[GPU VERIFICATION & DIAGNOSTICS]")
        self.logger.info("=" * 70)

        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        self.logger.info(f"CUDA Available: {cuda_available}")

        if not cuda_available:
            if self.device.type == 'cuda':
                raise RuntimeError(
                    f"Configuration requests CUDA device but CUDA is not available!\n"
                    f"  Requested device: {self.device}\n"
                    f"  CUDA available: {cuda_available}\n"
                    f"\n  SOLUTION: Change 'device: cuda' to 'device: cpu' in config.yaml"
                )
            else:
                self.logger.warning("[WARNING] Training on CPU - this will be VERY slow!")
                return

        # GPU is available - gather detailed info
        device_count = torch.cuda.device_count()
        self.logger.info(f"CUDA Device Count: {device_count}")

        if device_count == 0:
            raise RuntimeError("CUDA reports available but no devices found!")

        # Current device info
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        self.logger.info(f"Current CUDA Device: {current_device} ({device_name})")

        # Verify configured device matches available device
        if self.device.type == 'cuda':
            device_index = self.device.index or 0
            if device_index >= device_count:
                raise RuntimeError(
                    f"Configured device index {device_index} out of range! "
                    f"Available devices: 0-{device_count-1}"
                )

        # GPU Capabilities
        major, minor = torch.cuda.get_device_capability(current_device)
        self.logger.info(f"Compute Capability: {major}.{minor}")

        # Memory information
        total_memory = torch.cuda.get_device_properties(current_device).total_memory
        total_memory_gb = total_memory / 1e9
        allocated_memory = torch.cuda.memory_allocated(current_device) / 1e9
        reserved_memory = torch.cuda.memory_reserved(current_device) / 1e9
        free_memory = total_memory_gb - reserved_memory

        self.logger.info(f"GPU Memory:")
        self.logger.info(f"  Total: {total_memory_gb:.2f} GB")
        self.logger.info(f"  Free: {free_memory:.2f} GB")
        self.logger.info(f"  Allocated: {allocated_memory:.3f} GB")
        self.logger.info(f"  Reserved: {reserved_memory:.3f} GB")

        # Minimum memory check
        min_required_gb = 6  # Minimum 6GB for TimesFM fine-tuning
        if free_memory < min_required_gb:
            self.logger.warning(
                f"[WARNING] Low GPU memory: {free_memory:.2f} GB free < {min_required_gb} GB required"
            )
            self.logger.warning("         Consider reducing batch_size in config.yaml")

        # bfloat16 support (critical for TimesFM)
        bf16_support = torch.cuda.is_bf16_supported()
        self.logger.info(f"bfloat16 Supported: {bf16_support}")

        if not bf16_support:
            self.logger.warning(
                "[WARNING] GPU does not support bfloat16!\n"
                "         Model will use fallback dtype (slower, more memory)"
            )

        # PyTorch version info
        self.logger.info(f"PyTorch Version: {torch.__version__}")
        self.logger.info(f"CUDA Version: {torch.version.cuda}")

        # Summary verdict
        self.logger.info("=" * 70)
        if total_memory_gb >= 8 and bf16_support:
            self.logger.info("[OK] GPU setup looks good for training!")
        elif total_memory_gb >= 6:
            self.logger.info("[OK] GPU may work (low memory, consider reducing batch_size)")
        else:
            self.logger.warning("[WARNING] GPU memory may be insufficient")

    def run_benchmark(self, num_iterations: int = 100) -> Dict[str, float]:
        """
        Quick GPU benchmark to verify actual performance

        Args:
            num_iterations: Number of benchmark iterations

        Returns:
            Dictionary with benchmark metrics (avg_time_ms, throughput_ops_per_sec)
        """
        self.logger.info("=" * 70)
        self.logger.info("[GPU BENCHMARK]")
        self.logger.info("=" * 70)

        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available - skipping benchmark")
            return {'avg_time_ms': 0, 'throughput_ops_per_sec': 0}

        # Create test tensors (mimicking TimesFM batch size)
        batch_size = self.config['training']['batch_size']
        context_len = self.config['dataset']['context_length']
        horizon_len = self.config['dataset'].get('horizon_length', 1)

        context = torch.randn(batch_size, context_len, device=self.device, dtype=torch.bfloat16)
        weights = torch.randn(context_len, 512, device=self.device, dtype=torch.bfloat16)

        # Warmup
        for _ in range(10):
            _ = torch.matmul(context, weights)

        # Benchmark
        torch.cuda.synchronize()
        import time

        # Warmup completed, now benchmark
        start = time.time()

        for _ in range(num_iterations):
            result = torch.matmul(context, weights)

        torch.cuda.synchronize()
        end = time.time()

        # Calculate metrics with safety checks
        elapsed_time = end - start

        if elapsed_time <= 0:
            self.logger.warning("[WARNING] Benchmark time too short, using fallback values")
            avg_time_ms = 0.001  # Fallback: 1 microsecond
            throughput_ops_per_sec = 0
        else:
            avg_time_ms = elapsed_time / num_iterations * 1000
            ops_per_iteration = 2 * batch_size * context_len * 512
            throughput_ops_per_sec = ops_per_iteration / (elapsed_time / num_iterations)

        self.logger.info(f"Benchmark Results ({num_iterations} iterations):")
        self.logger.info(f"  Total time: {elapsed_time:.4f} seconds")
        self.logger.info(f"  Avg time per operation: {avg_time_ms:.3f} ms")
        if throughput_ops_per_sec > 0:
            self.logger.info(f"  Throughput: {throughput_ops_per_sec/1e9:.2f} GFLOPS")
        else:
            self.logger.info(f"  Throughput: N/A (operations too fast)")

        return {
            'avg_time_ms': avg_time_ms,
            'throughput_ops_per_sec': throughput_ops_per_sec
        }

    def load_timesfm_model(self) -> None:
        """
        Load TimesFM 2.5 model for financial fine-tuning

        Using TimesFm2_5ModelForPrediction with bfloat16 precision
        """
        self.logger.info("=" * 70)
        self.logger.info("[LOADING TIMESFM 2.5 FOUNDATION MODEL]")
        self.logger.info("=" * 70)

        try:
            model_id = self.config['model']['model_name']

            # Load TimesFM 2.5 model with MODERN API (no deprecated torch_dtype)
            self.logger.info(f"Loading model: {model_id}")
            self.base_model = TimesFm2_5ModelForPrediction.from_pretrained(
                model_id,
                dtype=torch.bfloat16,  # Use 'dtype' instead of deprecated 'torch_dtype'
                device_map=self.device,
                attn_implementation="sdpa"  # CRITICAL: Use scaled_dot_product_attention for 2-3x speedup
            )

            self.logger.info(f"[OK] TimesFM loaded successfully")
            self.logger.info(f"Model parameters: {self.config['model']['parameters']}")

            # IMPORTANT: Handle context_len - ensure it doesn't exceed model limit
            config_context_len = self.config['dataset']['context_length']
            config_horizon_len = self.config['dataset'].get('horizon_length', 1)

            # Use min of config and model limits (Google Research approach)
            self.context_len = min(config_context_len, self.base_model.config.context_length)
            self.horizon_len = min(config_horizon_len, self.base_model.config.horizon_length)

            # Store for reference
            self.model_context_len = self.base_model.config.context_length
            self.model_horizon_len = self.base_model.config.horizon_length

            self.logger.info(f"Config context length: {config_context_len}")
            self.logger.info(f"Model context length: {self.model_context_len}")
            self.logger.info(f"Actual context length: {self.context_len}")
            self.logger.info(f"Config horizon length: {config_horizon_len}")
            self.logger.info(f"Model horizon length: {self.model_horizon_len}")
            self.logger.info(f"Actual horizon length: {self.horizon_len}")

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to load TimesFM: {e}")
            raise

    def setup_lora_adapters(self) -> None:
        """
        Configure LoRA adapters using Google Research configuration

        Based on finetune_lora.py from TimesFM Google Research:
        - target_modules="all-linear"
        - r=4, lora_alpha=8, lora_dropout=0.05
        """
        self.logger.info("=" * 70)
        self.logger.info("[CONFIGURING LORA ADAPTERS]")
        self.logger.info("=" * 70)

        try:
            # Get LoRA configuration from config.yaml (Google Research approach)
            lora_config_dict = self.config['model']['lora']

            lora_config = LoraConfig(
                r=lora_config_dict['r'],                    # Rank: 4 (from config)
                lora_alpha=lora_config_dict['lora_alpha'],  # Alpha: 8 (from config)
                target_modules=lora_config_dict['target_modules'],  # "all-linear" (Google Research)
                lora_dropout=lora_config_dict.get('lora_dropout', 0.05),
                bias=lora_config_dict.get('bias', 'none')
            )

            self.logger.info(f"LoRA Configuration:")
            self.logger.info(f"  r (rank): {lora_config_dict['r']}")
            self.logger.info(f"  lora_alpha: {lora_config_dict['lora_alpha']}")
            self.logger.info(f"  target_modules: {lora_config_dict['target_modules']}")
            self.logger.info(f"  lora_dropout: {lora_config_dict.get('lora_dropout', 0.05)}")

            # Apply LoRA adapters
            self.model = get_peft_model(self.base_model, lora_config)
            self.model.print_trainable_parameters()

            self.logger.info(f"[OK] LoRA adapters configured successfully (Google Research config)")

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to setup LoRA: {e}")
            raise

    def setup_optimizer(self) -> None:
        """
        Setup optimizer using Google Research configuration

        Based on finetune_lora.py from TimesFM Google Research:
        - AdamW with lr=1e-4, weight_decay=0.01
        - CosineAnnealingLR scheduler
        """
        self.logger.info("=" * 70)
        self.logger.info("[SETTING UP OPTIMIZER]")
        self.logger.info("=" * 70)

        try:
            # Get training configuration from config.yaml
            training_config = self.config['training']

            # Google Research: AdamW with weight_decay=0.01
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=training_config['learning_rate'],  # 1e-4 from config
                weight_decay=0.01  # Google Research uses 0.01
            )

            self.logger.info(f"Optimizer Configuration:")
            self.logger.info(f"  Type: AdamW")
            self.logger.info(f"  Learning rate: {training_config['learning_rate']}")
            self.logger.info(f"  Weight decay: 0.01 (Google Research standard)")

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to setup optimizer: {e}")
            raise

    def setup_scheduler(self, num_training_steps: int) -> None:
        """
        Setup learning rate scheduler using Google Research configuration

        Based on finetune_lora.py:
        - CosineAnnealingLR with T_max = epochs * len(train_loader)

        Args:
            num_training_steps: Total number of training steps (epochs * batches)
        """
        self.logger.info("=" * 70)
        self.logger.info("[SETTING UP LEARNING RATE SCHEDULER]")
        self.logger.info("=" * 70)

        try:
            # Get min_learning_rate from training config (directly, not nested under scheduler)
            min_lr = self.config['training'].get('min_learning_rate', 1e-6)

            # Google Research: CosineAnnealingLR
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps,
                eta_min=float(min_lr)
            )

            self.logger.info(f"Scheduler Configuration:")
            self.logger.info(f"  Type: CosineAnnealingLR")
            self.logger.info(f"  T_max (steps): {num_training_steps}")
            self.logger.info(f"  eta_min (min lr): {min_lr}")

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to setup scheduler: {e}")
            raise

    def train_one_epoch(self, train_loader: DataLoader,
                       epoch: int, num_epochs: int) -> Dict[str, float]:
        """
        Train one epoch following tutorial methodology

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
        consecutive_errors = 0
        max_consecutive_errors = 10  # Stop if too many errors in a row
        epoch_start_time = time.time()
        last_log_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time()

            try:
                # DEBUG: Check devices and timing
                import time as time_module
                batch_start = time_module.time()

                # OPTIMIZED: Convert to bfloat16 on CPU FIRST (cheaper), then transfer to GPU
                # This is faster than: float32->GPU->bfloat16 conversion
                model_dtype = getattr(self.model.config, 'dtype', getattr(self.model.config, 'torch_dtype', None))
                needs_bfloat16 = model_dtype == torch.bfloat16 or str(model_dtype) == 'torch.bfloat16'

                if needs_bfloat16:
                    # Convert on CPU (cheaper) then transfer smaller data to GPU
                    context_batch = batch['context'].to(torch.bfloat16).to(self.device, non_blocking=True)
                    ground_truth_batch = batch['ground_truth'].to(torch.bfloat16).to(self.device, non_blocking=True)
                else:
                    # Transfer as-is
                    context_batch = batch['context'].to(self.device, non_blocking=True)
                    ground_truth_batch = batch['ground_truth'].to(self.device, non_blocking=True)

                data_transfer_time = time_module.time() - batch_start

                # Frequency ID for Vietnamese stocks (assuming daily = 0)
                frequency_id_batch = [[0]] * context_batch.shape[0]

                # CRITICAL: Check if model is actually on GPU
                if n_batches == 1:  # Only log once
                    self.logger.info(f"[DEBUG] Model device: {next(self.model.parameters()).device}")
                    self.logger.info(f"[DEBUG] Context device: {context_batch.device}")
                    self.logger.info(f"[DEBUG] Data transfer time: {data_transfer_time*1000:.2f}ms")

                # Forward pass using tutorial API
                forward_start = time_module.time()
                outputs = self.model(
                    past_values=context_batch,
                    freq=frequency_id_batch,        # CRITICAL: freq parameter
                    future_values=ground_truth_batch  # CRITICAL: for loss calculation
                )
                forward_time = time_module.time() - forward_start

                # Access loss from outputs
                loss = outputs.loss

                # Save loss value BEFORE cleanup
                loss_value = loss.item()

                # Backward pass
                backward_start = time_module.time()
                self.optimizer.zero_grad()
                loss.backward()

                # OPTIMIZATION: Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Optimizer step
                self.optimizer.step()

                # Scheduler step (Google Research: step per batch for CosineAnnealingLR)
                if self.scheduler is not None:
                    self.scheduler.step()

                backward_time = time_module.time() - backward_start

                # SIMPLIFIED: Only cleanup loss (PyTorch handles the rest automatically)
                del loss

                # Metrics and timing debug
                if n_batches == 1:  # Only log once
                    self.logger.info(f"[DEBUG] Forward time: {forward_time*1000:.2f}ms")
                    self.logger.info(f"[DEBUG] Backward time: {backward_time*1000:.2f}ms")
                    self.logger.info(f"[DEBUG] Total batch time: {(time_module.time()-batch_start)*1000:.2f}ms")

                total_loss += loss_value
                n_batches += 1
                consecutive_errors = 0  # Reset error counter

                # Log progress every 10 batches with timing
                if n_batches % 10 == 0:
                    avg_loss = total_loss / n_batches
                    now = time.time()
                    batch_time = now - batch_start_time
                    time_since_last_log = now - last_log_time
                    last_log_time = now

                    # Calculate speed metrics
                    batches_per_sec = 10 / time_since_last_log
                    estimated_epoch_time = (len(train_loader) - n_batches) / batches_per_sec / 60  # minutes

                    self.logger.info(
                        f"Epoch {epoch+1} | Batch {n_batches}/{len(train_loader)} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Speed: {batch_time*1000:.1f}ms/batch, {batches_per_sec:.1f} batches/sec | "
                        f"ETA: {estimated_epoch_time:.1f} min"
                    )

            except Exception as e:
                consecutive_errors += 1
                self.logger.error(f"Training error at batch {n_batches}: {e}")
                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(f"Too many consecutive errors ({max_consecutive_errors}). Stopping training.") from e
                continue  # Skip to next batch

        # Calculate epoch timing
        epoch_time = time.time() - epoch_start_time
        self.logger.info(f"Epoch {epoch+1} completed in {epoch_time/60:.2f} minutes")

        # SIMPLIFIED: Only cleanup at end of epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        with torch.no_grad():
            for batch in test_loader:
                # OPTIMIZED: Convert to bfloat16 on CPU FIRST, then transfer to GPU
                model_dtype = getattr(self.model.config, 'dtype', getattr(self.model.config, 'torch_dtype', None))
                needs_bfloat16 = model_dtype == torch.bfloat16 or str(model_dtype) == 'torch.bfloat16'

                if needs_bfloat16:
                    context_batch = batch['context'].to(torch.bfloat16).to(self.device, non_blocking=True)
                    ground_truth_batch = batch['ground_truth'].to(torch.bfloat16).to(self.device, non_blocking=True)
                else:
                    context_batch = batch['context'].to(self.device, non_blocking=True)
                    ground_truth_batch = batch['ground_truth'].to(self.device, non_blocking=True)

                frequency_id_batch = [[0]] * context_batch.shape[0]

                try:
                    # Forward pass
                    outputs = self.model(
                        past_values=context_batch,
                        freq=frequency_id_batch,
                        future_values=ground_truth_batch
                    )

                    val_loss += outputs.loss.item()
                    val_batches += 1

                    # SIMPLIFIED: Only cleanup outputs
                    del outputs

                except Exception as e:
                    self.logger.error(f"Validation error: {e}")
                    continue

        avg_val_loss = val_loss / max(val_batches, 1)

        return {
            'val_loss': avg_val_loss,
            'num_batches': val_batches
        }

    def train_model(self, train_loader: DataLoader = None,
                   test_loader: DataLoader = None) -> None:
        """
        Complete training loop following tutorial methodology

        Args:
            train_loader: Training data loader (optional, will create if None)
            test_loader: Test data loader (optional, will create if None)
        """
        self.logger.info("=" * 70)
        self.logger.info("[STARTING TIMESFM FINE-TUNING]")
        self.logger.info("=" * 70)

        # Create dataloaders with model's actual dimensions if not provided
        if train_loader is None or test_loader is None:
            logging.info("Creating dataloaders with adjusted dimensions...")
            logging.info(f"  Using context_len={self.context_len}, horizon_len={self.horizon_len}")
            train_loader, test_loader = create_vn30_dataloaders(
                self.config['config_path'] if hasattr(self, 'config_path') else 'configs/config.yaml',
                context_len=self.context_len,      # Use ADJUSTED context length (128)
                horizon_len=self.horizon_len       # Use ADJUSTED horizon length (1)
            )

        num_epochs = self.config['training']['num_epochs']

        self.logger.info(f"Training configuration:")
        self.logger.info(f"  Epochs: {num_epochs}")
        self.logger.info(f"  Train batches: {len(train_loader)}")
        self.logger.info(f"  Test batches: {len(test_loader)}")
        self.logger.info(f"  Batch size: {self.config['training']['batch_size']}")
        self.logger.info(f"  Context length: {self.context_len} (config)")
        self.logger.info(f"  Horizon length: {self.horizon_len} (config)")
        self.logger.info(f"  Model max context: {self.model_context_len} (model limit)")
        self.logger.info(f"  Model max horizon: {self.model_horizon_len} (model limit)")
        self.logger.info(f"  Actual input shape: [batch_size, {self.context_len}] -> [batch_size, {self.horizon_len}]")

        # NOTE: Following Google Research approach - NOT using torch.compile()
        # torch.compile() requires Triton which is not available on Windows
        # Google Research uses eager mode directly (still fast on GPU)
        self.logger.info("=" * 70)
        self.logger.info("[TRAINING MODE]")
        self.logger.info("=" * 70)
        self.logger.info("Using eager mode (Google Research approach)")
        # torch.compile() DISABLED - causes AttributeError with TimesFM 2.5
        # Model is already optimized with bfloat16 + SDPA attention
        self.logger.info("Using eager mode (torch.compile disabled for compatibility)")

        # Run quick GPU benchmark to verify performance
        self.run_benchmark(num_iterations=200)  # More iterations for reliable timing

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            try:
                # Train one epoch
                train_metrics = self.train_one_epoch(train_loader, epoch, num_epochs)

                # Validate
                val_metrics = self.validate_model(test_loader)

                # Log epoch results
                self.logger.info("=" * 70)
                self.logger.info(f"EPOCH {epoch+1}/{num_epochs} COMPLETE")
                self.logger.info("=" * 70)
                self.logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
                self.logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")

                # Store training history
                epoch_log = {
                    'epoch': epoch + 1,
                    'timestamp': datetime.now().isoformat(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }
                self.training_history.append(epoch_log)

                # Save best model
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.epochs_since_improvement = 0  # Reset counter
                    self.save_checkpoint('best_model')
                    self.logger.info(f"[NEW BEST] Val loss = {val_metrics['val_loss']:.4f}")
                else:
                    self.epochs_since_improvement += 1
                    self.logger.info(f"[NO IMPROVEMENT] {self.epochs_since_improvement}/{self.early_stopping_patience} epochs since best")

                # Early stopping check
                if self.epochs_since_improvement >= self.early_stopping_patience:
                    self.logger.info("=" * 70)
                    self.logger.info(f"[EARLY STOPPING] No improvement for {self.early_stopping_patience} epochs")
                    self.logger.info(f"Stopping at epoch {epoch+1}/{num_epochs}")
                    self.logger.info(f"Best val loss: {self.best_val_loss:.4f}")
                    self.logger.info("=" * 70)
                    break

                # Periodic checkpoint
                if (epoch + 1) % self.config['experiment_tracking']['save_every_n_epochs'] == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch+1}')
                    self.logger.info(f"[CHECKPOINT] Saved epoch {epoch+1}")

                # Save training history (with error handling)
                self.save_training_history()

                # Update learning curves (REAL-TIME VISUALIZATION)
                self.plot_learning_curves()

            except Exception as e:
                self.logger.error("=" * 70)
                self.logger.error(f"[ERROR] Exception during epoch {epoch+1}: {e}")
                self.logger.error("=" * 70)
                self.logger.error(f"[ERROR] Training stopped due to error")
                self.logger.error(f"[ERROR] Best val loss so far: {self.best_val_loss:.4f}")
                self.logger.error(f"[ERROR] Epochs completed: {epoch+1}/{num_epochs}")

                # Save what we have before exiting
                try:
                    self.save_training_history()
                    self.plot_learning_curves()
                except:
                    pass

                # Re-raise to exit
                raise

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
        try:
            experiments_dir = Path(self.config['experiment_tracking']['experiments_dir'])
            experiments_dir.mkdir(parents=True, exist_ok=True)

            history_path = experiments_dir / 'training_history.json'

            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)

            self.logger.info(f"[HISTORY] Saved: {history_path}")

        except Exception as e:
            self.logger.warning(f"[WARNING] Failed to save training history: {e}")
            self.logger.warning(f"[WARNING] Training will continue, but history may not be saved")

    def plot_learning_curves(self) -> None:
        """
        Plot and save learning curves (train/val loss) during training.

        Creates real-time visualization of training progress showing:
        - Training loss per epoch
        - Validation loss per epoch
        - Best model checkpoint indicator

        Saves to: experiments/learning_curves.png
        """
        if not self.training_history:
            return  # No data to plot yet

        try:
            # Extract metrics from training history
            epochs = [log['epoch'] for log in self.training_history]
            train_losses = [log['train_metrics']['loss'] for log in self.training_history]
            val_losses = [log['val_metrics']['val_loss'] for log in self.training_history]

            # Create figure
            plt.figure(figsize=(10, 6))

            # Plot training and validation loss
            plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=4)
            plt.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=4)

            # Mark best model (lowest validation loss)
            if len(self.training_history) > 0:
                best_epoch = min(range(len(val_losses)), key=lambda i: val_losses[i])
                best_val_loss = val_losses[best_epoch]
                plt.axvline(x=epochs[best_epoch], color='g', linestyle='--', alpha=0.5,
                           label=f'Best Model (Epoch {epochs[best_epoch]}, Val Loss={best_val_loss:.4f})')

            # Formatting
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('TimesFM Fine-tuning Learning Curves', fontsize=14, fontweight='bold')
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, alpha=0.3)

            # Add current epoch annotation
            current_epoch = epochs[-1]
            current_train = train_losses[-1]
            current_val = val_losses[-1]
            plt.annotate(f'Epoch {current_epoch}: Train={current_train:.4f}, Val={current_val:.4f}',
                        xy=(current_epoch, current_val),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
                        fontsize=9)

            # Save figure
            experiments_dir = Path(self.config['experiment_tracking']['experiments_dir'])
            experiments_dir.mkdir(parents=True, exist_ok=True)
            curves_path = experiments_dir / 'learning_curves.png'
            plt.savefig(curves_path, dpi=100, bbox_inches='tight')

            # Display inline in Colab
            if IN_COLAB:
                from IPython.display import Image, display
                display(Image(str(curves_path)))
            else:
                plt.close()

            # Log update (only every epoch to avoid spam)
            self.logger.info(f"[LEARNING CURVES] Updated: {curves_path}")

        except Exception as e:
            self.logger.warning(f"[WARNING] Failed to plot learning curves: {e}")


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

    # Create dataloaders first to calculate training steps for scheduler
    print("\n[INFO] Creating dataloaders to calculate training steps for scheduler...")
    train_loader, _ = create_vn30_dataloaders('configs/config.yaml')

    # Setup scheduler
    num_epochs = finetuner.config['training']['num_epochs']
    num_training_steps = num_epochs * len(train_loader)
    finetuner.setup_scheduler(num_training_steps)

    # Train model (dataloader will be created internally with correct dimensions)
    finetuner.train_model()

    # Final summary
    logging.info("=" * 70)
    logging.info("[TRAINING COMPLETE]")
    logging.info("=" * 70)
    logging.info(f"Best val loss: {finetuner.best_val_loss:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
