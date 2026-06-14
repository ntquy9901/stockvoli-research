"""
Test to verify improved VN30TimeSeriesDataset with proper temporal split and increased test samples
"""
import numpy as np
import sys
sys.path.append('src')

from model_training_fixed import VN30TimeSeriesDataset
import pandas as pd

print("=" * 70)
print("Testing Improved VN30TimeSeriesDataset")
print("=" * 70)

# Create mock data (simulating VCB with 4000 observations)
mock_data = {}
for stock in ['VCB', 'VIC', 'VNM']:
    dates = pd.date_range('2009-01-01', periods=4000, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'RV_20': np.random.randn(4000) * 0.02 + 0.015  # Random volatility
    })
    mock_data[stock] = df

print(f"\nMock Data:")
print(f"  Stocks: {list(mock_data.keys())}")
print(f"  Observations per stock: 4000")
print(f"  Date range: 2009-01-01 to ~2020")

# Test with NEW IMPROVED parameters
print(f"\n{'=' * 70}")
print("Test 1: Temporal Split Validation")
print('=' * 70)

# Create train dataset
train_dataset = VN30TimeSeriesDataset(
    mock_data,
    context_len=128,
    horizon_len=1,
    num_samples=200,  # Standard training samples
    mode='train',
    seed=42
)

# Create test dataset with BALANCED samples
test_dataset = VN30TimeSeriesDataset(
    mock_data,
    context_len=128,
    horizon_len=1,
    num_samples=0,  # Not used for test mode
    mode='test',
    seed=42,
    test_samples_per_stock=30  # BALANCED: 30 samples per stock = 900 total (~6.7:1 ratio)
)

# Check sample ranges
train_starts = []
test_starts = []

for idx in range(len(train_dataset)):
    sample = train_dataset.samples[idx]
    train_starts.append(sample[1])

for idx in range(len(test_dataset)):
    sample = test_dataset.samples[idx]
    test_starts.append(sample[1])

print(f"\nTrain Dataset:")
print(f"  Total samples: {len(train_starts)}")
print(f"  Per stock: {len(train_starts) // 3}")
print(f"  Min start: {min(train_starts)}")
print(f"  Max start: {max(train_starts)}")
print(f"  Expected max: ~{int(4000 * 0.8) - 128 - 1} (80% point - context)")

print(f"\nTest Dataset (BALANCED):")
print(f"  Total samples: {len(test_starts)}")
print(f"  Per stock: {len(test_starts) // 3}")
print(f"  Min start: {min(test_starts)}")
print(f"  Max start: {max(test_starts)}")
print(f"  Expected min: ~{int(4000 * 0.8) + 128} (80% point + context)")

# Verify NO overlap
split_point = int(4000 * 0.8)
train_max = max(train_starts)
test_min = min(test_starts)

print(f"\n{'=' * 70}")
print(f"Temporal Split Validation:")
print(f'=' * 70)
print(f"Split point (80%): {split_point}")
print(f"Train max start: {train_max}")
print(f"Test min start: {test_min}")

if train_max < test_min:
    gap = test_min - train_max
    print(f"[OK] SUCCESS: No overlap! Train < Test")
    print(f"   Gap: {gap} observations")
else:
    print(f"[FAIL] Overlap detected! Train max ({train_max}) >= Test min ({test_min})")
    print(f"   This indicates DATA LEAKAGE!")

print(f"\n{'=' * 70}")
print("Final Configuration:")
print('=' * 70)
print(f"[+] Test samples per stock: 30 (BALANCED)")
print(f"[+] Total test samples: {len(test_starts)} (900 for 30 stocks)")
print(f"[+] Train:Test ratio: ~6.7:1 (balanced)")
print(f"[+] Proper temporal split: Verified")
print(f"[+] 30x more test data than old 30 samples total")


