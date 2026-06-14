"""
Multi-Stock Dataset for TimesFM VN30 Fine-tuning
Implements channel-independent architecture for 30 Vietnamese stocks
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from sklearn.model_selection import train_test_split
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments/dataset_creation.log'),
        logging.StreamHandler()
    ]
)


class VN30MultiStockDataset(Dataset):
    """
    Multi-stock dataset treating each stock as separate univariate time series

    Architecture follows Issue #230 resolution for multiple time series:
    - Each stock (VCB, VIC, VNM, etc.) is independent series
    - Random window sampling across all stocks
    - Proper time-based train/test split (no leakage)
    - Channel-independent processing

    Args:
        stock_data_dict: Dictionary mapping stock names to processed DataFrames
        context_len: Historical context window (default: 128 trading days)
        horizon_len: Prediction horizon (default: 1 trading day ahead)
        mode: 'train' or 'test' for time-based split
        samples_per_stock: Number of random samples to generate per stock
    """

    def __init__(self, stock_data_dict: Dict[str, pd.DataFrame],
                 context_len: int = 128, horizon_len: int = 1,
                 mode: str = 'train', samples_per_stock: int = 200):
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.mode = mode
        self.samples_per_stock = samples_per_stock

        self.logger = logging.getLogger(__name__)

        # Process each stock as separate time series
        self.series_list = []
        for stock_name, df in stock_data_dict.items():
            # Use RV_20 as our primary target variable
            if 'RV_20' in df.columns:
                series_data = df['RV_20'].dropna().values

                # Ensure we have enough data
                if len(series_data) >= context_len + horizon_len:
                    self.series_list.append({
                        'name': stock_name,
                        'data': series_data,
                        'dates': df.index[~df['RV_20'].isna()]  # Keep dates for reference
                    })
                    self.logger.info(f"Added {stock_name}: {len(series_data)} observations")
                else:
                    self.logger.warning(f"Skipped {stock_name}: insufficient data ({len(series_data)} < {context_len + horizon_len})")

        # Determine train/test split point (temporal split)
        if mode == 'train':
            self.split_point = 0.8  # Use first 80% for training
        else:
            self.split_point = 0.8  # Use last 20% for testing

        # Generate samples
        self.samples = self._create_random_windows()

        self.logger.info(f"Created {len(self.samples)} samples for {mode} mode")

    def _create_random_windows(self) -> List[Dict]:
        """Create random window samples from all stocks"""
        samples = []

        for series_info in self.series_list:
            series_data = series_info['data']
            stock_name = series_info['name']
            data_len = len(series_data)

            # Determine split point for this stock
            if self.mode == 'train':
                max_start = int(data_len * self.split_point) - self.context_len - self.horizon_len
                if max_start < 10:  # Need minimum data for split
                    continue
                start_indices = np.random.randint(0, max_start, size=self.samples_per_stock)
            else:  # test mode
                min_start = int(data_len * self.split_point) + self.context_len
                if min_start >= data_len - self.context_len - self.horizon_len:
                    continue
                start_indices = np.random.randint(min_start, data_len - self.context_len - self.horizon_len, size=self.samples_per_stock)

            for start_idx in start_indices:
                # Extract context window
                context = series_data[start_idx:start_idx + self.context_len].astype(np.float32)

                # Extract target (next day's RV_20)
                target = series_data[start_idx + self.context_len + self.horizon_len - 1]

                # Store sample
                samples.append({
                    'context': context,
                    'target': target,
                    'stock': stock_name
                })

        self.logger.info(f"Generated {len(samples)} samples from {len(self.series_list)} stocks")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'context': torch.from_numpy(sample['context']).float(),  # More efficient than FloatTensor
            'target': torch.tensor([sample['target']], dtype=torch.float32),
            'stock': sample['stock']
        }


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

    train_dataset = VN30MultiStockDataset(
        stock_data_dict,
        context_len=context_len,
        horizon_len=horizon_len,
        mode='train',
        samples_per_stock=samples_per_stock
    )

    test_dataset = VN30MultiStockDataset(
        stock_data_dict,
        context_len=context_len,
        horizon_len=horizon_len,
        mode='test',
        samples_per_stock=max(50, samples_per_stock // 4)  # Fewer samples for test
    )

    # Create dataloaders
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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


def test_dataset_creation():
    """Test dataset creation and validate output"""
    logging.info("=" * 70)
    logging.info("[TESTING MULTI-STOCK DATASET]")
    logging.info("=" * 70)

    try:
        train_loader, test_loader = create_vn30_dataloaders()

        # Test a batch from training loader
        batch = next(iter(train_loader))
        logging.info(f"Training batch:")
        logging.info(f"  Context shape: {batch['context'].shape}")
        logging.info(f"  Target shape: {batch['target'].shape}")
        logging.info(f"  Batch size: {len(batch['context'])}")

        # Test a batch from test loader
        test_batch = next(iter(test_loader))
        logging.info(f"Test batch:")
        logging.info(f"  Context shape: {test_batch['context'].shape}")
        logging.info(f"  Target shape: {test_batch['target'].shape}")

        logging.info("[OK] Dataset creation successful!")
        return True

    except Exception as e:
        logging.error(f"[FAIL] Dataset creation failed: {e}")
        return False


def main():
    """Main execution function"""
    import json

    # Test dataset creation
    success = test_dataset_creation()

    if success:
        # Create final datasets for training
        train_loader, test_loader = create_vn30_dataloaders()

        # Generate dataset report
        dataset_info = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'train_samples': len(train_loader.dataset),
            'test_samples': len(test_loader.dataset),
            'train_batches': len(train_loader),
            'test_batches': len(test_loader),
            'batch_size': train_loader.batch_size
        }

        # Save dataset info
        experiments_dir = Path('experiments')
        with open(experiments_dir / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)

        logging.info("=" * 70)
        logging.info("[DATASET CREATION SUMMARY]")
        logging.info("=" * 70)
        logging.info(f"Training samples: {dataset_info['train_samples']}")
        logging.info(f"Test samples: {dataset_info['test_samples']}")
        logging.info(f"Batch size: {dataset_info['batch_size']}")
        logging.info("[OK] Dataset creation completed successfully!")

    return 0 if success else 1


if __name__ == "__main__":
    import sys
    import pandas as pd  # Need pandas for timestamp in report
    sys.exit(main())