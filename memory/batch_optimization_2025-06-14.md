---
name: batch_optimization_colab
description: Colab batch size optimization from 12 to 48
type: project
---

# Batch Size Optimization - Colab G4 Configuration

## Configuration Changes (2025-06-14)

**Original Configuration (Slow):**
- batch_size: 12
- gradient_accumulation_steps: 6
- Effective batch: 72
- Batches per epoch: 375
- Performance: ~2.4s/batch, ~15 min/epoch

**Optimized Configuration (Fast):**
- batch_size: 48 (4x improvement)
- gradient_accumulation_steps: 1 (no accumulation needed)
- Batches per epoch: 94 (4x fewer)
- Expected performance: 0.6-0.8s/batch, ~3.75 min/epoch
- Expected speedup: 3-4x faster per epoch

## Test Configurations Created

1. **configs/config_batch48_test.yaml** - Batch size 48 (4x improvement)
2. **configs/config_batch96_test.yaml** - Batch size 96 (8x improvement)
3. **src/test_batch_performance.py** - Automated testing script

## Performance Bottleneck Analysis

**Root cause:** GPU massively underutilized
- Current batch size: 12 samples
- GPU capacity: 64-128 samples easily
- GPU VRAM: 22.5GB but only using ~5GB

**Performance breakdown:**
- Forward time: 1407.49ms (56% - MAIN BOTTLENECK)
- Backward time: 1026.22ms (41%)
- Total: 2434.93ms (2.4 seconds/batch)

## Fixes Applied to Test Script

**File:** src/test_batch_performance.py

1. **Import fixes:**
   - Changed from `ml_ds_common_rules.training` to `src.model_training_fixed`
   - Used `create_vn30_dataloaders()` instead of manual data loading
   - Fixed method name: `load_timesfm_model()` instead of `load_model()`

2. **Logging fixes:**
   - Added `sys.stdout` to handlers
   - Added `force=True` to override existing loggers
   - Fixed f-string formatting in warning messages

3. **Compatibility fixes:**
   - Removed Unicode box-drawing characters (╔═╗) - replaced with ASCII (=)
   - Removed emoji characters (🧪✅❌) - replaced with [TEST][OK][ERROR]

## HuggingFace Cache Setup

**Problem:** C: drive insufficient space (925MB model, only 261MB free)

**Solution:** Moved cache to D: drive
```bash
export HF_HOME=/d/huggingface_cache
export TRANSFORMERS_CACHE=/d/huggingface_cache/models
export HF_HUB_CACHE=/d/huggingface_cache/hub
```

**Windows Issue:** Encountered "paging file is too small" error - not resolved yet.

## Git Commits

1. `30cf9af` - feat: Add batch size optimization configurations and testing
2. `95d1047` - fix: Correct batch performance test script imports and logging
3. `245fa13` - feat: Increase Colab batch size from 12 to 48 for better GPU utilization

## Why Batch Size 48 (Not 96)?

**User decision:** Chose batch_size=48 for Colab (not 96)

**Rationale:**
- 48 provides 3-4x speedup (good enough)
- 96 would require 10-12GB VRAM (more risk of OOM)
- Better stability vs marginal extra gain
- Colab G4 has 22.5GB VRAM, batch 48 uses ~8-10GB (safe margin)

## Next Steps (When Continuing)

1. **Fix Windows paging file issue** - Increase virtual memory
2. **Run batch performance test** - `python src/test_batch_performance.py`
3. **Monitor first epoch** - Watch GPU memory, batch times, convergence
4. **Compare with baseline** - Verify speedup matches expectations

## Key Files Reference

- **Main config:** `configs/config.yaml` (batch_size=48, gradient_accumulation_steps=1)
- **Test configs:** `configs/config_batch48_test.yaml`, `configs/config_batch96_test.yaml`
- **Test script:** `src/test_batch_performance.py`
- **Documentation:** `docs/BATCH_SIZE_OPTIMIZATION.md`

## Expected Performance Metrics

With batch_size=48 on Colab G4:
- Batch time: 0.6-0.8 seconds (vs 2.4s baseline)
- Throughput: 60-80 samples/second
- Epoch time: ~3.75 minutes (vs 15 min baseline)
- GPU memory: ~8-10GB (safe within 22.5GB)
- Speedup: 3-4x faster overall

**Why:** TimesFM 2.5 is designed for up to 16,384 time steps but we only use 64 steps (0.4% of capability). Larger batches utilize GPU better without sacrificing convergence quality.

---

*Last Updated: 2025-06-14*
*Purpose: Optimize Colab G4 training performance*
