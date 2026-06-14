# Batch Size Optimization Guide

## Problem Summary

Current training is very slow: **~2.4 seconds per batch** with batch_size=12

### Performance Breakdown
```
Data transfer time:     0.30ms    (negligible)
Forward time:       1407.49ms    (56% - MAIN BOTTLENECK)
Backward time:      1026.22ms    (41%)
Total:              2434.93ms    (2.4 seconds/batch)
```

### Root Cause
- **TimesFM 2.5 is overkill for this task:**
  - Model designed for: Up to 16,384 time steps context
  - Currently using: 64 time steps (0.4% of capability!)
  - 200M parameters computing through transformer layers

- **GPU massively underutilized:**
  - Current batch size: 12 samples
  - GPU capacity: 64-128 samples easily
  - GPU VRAM: 22.5GB but only using ~5GB

## Test Configurations Created

### Test 1: Batch Size 48 (4x Improvement)
- **Config:** `configs/config_batch48_test.yaml`
- **Expected speedup:** 3-4x faster
- **Expected batch time:** 0.6-0.8 seconds
- **Effective batch size:** 48 (no accumulation needed)

### Test 2: Batch Size 96 (8x Improvement)
- **Config:** `configs/config_batch96_test.yaml`
- **Expected speedup:** 4-6x faster
- **Expected batch time:** 0.4-0.6 seconds
- **Expected GPU memory:** 10-12GB VRAM
- **Effective batch size:** 96 (no accumulation needed)

## How to Run Tests

### Option 1: Automated Performance Test (Recommended)
```bash
python src/test_batch_performance.py
```

This will:
- Test both batch sizes (48 and 96)
- Measure actual batch processing times
- Compare performance results
- Recommend optimal configuration
- Save results to `experiments/batch_performance_tests/results.json`

### Option 2: Manual Test with Training Script
```bash
# Test batch_size=48
python src/model_training.py --config configs/config_batch48_test.yaml

# Test batch_size=96
python src/model_training.py --config configs/config_batch96_test.yaml
```

## Expected Performance Improvements

| Configuration | Batch Size | Current Time | Expected Time | Speedup |
|--------------|-----------|--------------|---------------|---------|
| Baseline | 12 | 2.4s | - | 1x |
| Test 1 | 48 | 2.4s | 0.6-0.8s | 3-4x |
| Test 2 | 96 | 2.4s | 0.4-0.6s | 4-6x |

## Impact on Epoch Time

**Current (batch_size=12):**
- 375 batches/epoch
- 2.4s/batch × 375 = **~15 minutes/epoch**

**Test 1 (batch_size=48):**
- 94 batches/epoch
- 0.7s/batch × 94 = **~1.1 minutes/epoch** (13x faster!)

**Test 2 (batch_size=96):**
- 47 batches/epoch
- 0.5s/batch × 47 = **~0.4 minutes/epoch** (37x faster!)

## Memory Considerations

### GPU VRAM Usage (Estimated)
| Batch Size | Expected VRAM | Safe? |
|-----------|--------------|-------|
| 12 | ~5GB | ✅ Very safe |
| 48 | ~8-10GB | ✅ Safe |
| 96 | ~10-12GB | ✅ Safe (22.5GB available) |

### System RAM Usage
- 6 workers × prefetch_factor=3 = 18 batches preloaded
- Batch size 48: ~18×48 = 864 samples in memory
- Batch size 96: ~18×96 = 1,728 samples in memory
- Both easily fit in 53GB system RAM

## Next Steps

### Immediate
1. Run automated test: `python src/test_batch_performance.py`
2. Review results in `experiments/batch_performance_tests/results.json`
3. Choose optimal batch size based on actual measurements

### After Choosing Optimal
1. Update main config (`configs/config.yaml`) with optimal batch size
2. Archive old config if needed: `mv configs/config.yaml archived/config_old_2025-06-14.yaml`
3. Start training with new configuration

### Monitor First Epoch
When training starts with new batch size:
- Watch first 10 batches carefully
- Check GPU memory usage
- Monitor for any OOM errors
- Verify batch times match expectations

## Troubleshooting

### If Out of Memory (OOM)
```yaml
# Reduce batch size
training:
  batch_size: 36  # Or 64, depending on what works
```

### If GPU Still Underutilized
```yaml
# Further increase batch size
training:
  batch_size: 128  # If you have enough VRAM
```

### If DataLoader Becomes Bottleneck
```yaml
# Reduce workers to save CPU
training:
  num_workers: 4  # From 6
  prefetch_factor: 2  # From 3
```

## Key Changes from Baseline

### Removed (No Longer Needed)
- `gradient_accumulation_steps: 6` → `1` (larger batch = no accumulation needed)

### Kept (Still Important)
- `use_mixed_precision: true` - bfloat16 critical
- `gradient_checkpointing: true` - saves memory
- `num_workers: 6` - efficient data loading
- `pin_memory: true` - faster CPU→GPU transfer

## Files Created

1. `configs/config_batch48_test.yaml` - Test batch size 48
2. `configs/config_batch96_test.yaml` - Test batch size 96
3. `src/test_batch_performance.py` - Automated testing script
4. `docs/BATCH_SIZE_OPTIMIZATION.md` - This guide

## Questions to Answer After Testing

1. **Which batch size gives best throughput?**
2. **Is GPU memory still underutilized?**
3. **Are there any new bottlenecks appearing?**
4. **Does model quality change with larger batches?**

---

*Last Updated: 2025-06-14*
*Purpose: Optimize TimesFM training performance*
