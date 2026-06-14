# Data Leakage & Temporal Split Fix Plan

**Project:** stockvoli-research
**Date:** 2026-06-13
**Status:** DRAFT - FOR REVIEW
**Author:** Winston (System Architect)

---

## Executive Summary

Critical data leakage and temporal split issues discovered in TimesFM training pipeline that compromise model evaluation integrity. This plan provides comprehensive fix strategy.

---

## Issues Identified

### Issue 1: Temporal Split BUG in `model_training_fixed.py`

**Location:** `src/model_training_fixed.py:145-158`

**Problem:**
```python
# Test mode - THE BUG!
if mode == 'test':
    min_start = int(data_len * 0.8) + context_len  # ← Calculated
    max_start = data_len - min_len                  # ← Calculated
    # BUG: min_start NOT USED!
    start = rng.integers(0, max_start + 1)  # ← Should be (min_start, max_start + 1)
```

**Impact:** Test samples drawn from [0, max_start] instead of [min_start, max_start]
- Test data includes samples from first 80% (should be training only)
- **NO proper temporal separation**

---

### Issue 2: No Temporal Split in `model_training_google_research.py`

**Location:** `src/model_training_google_research.py:142-143`

**Problem:**
```python
max_start = len(series) - min_len
start = rng.integers(0, max_start + 1)  # ← From ENTIRE series!
```

**Impact:**
- Training samples can include data from 2025-2026 (recent!)
- Validation only 30 samples (last windows)
- **Data leakage: model sees future during training**

---

### Issue 3: Insufficient Validation Data

**Location:** `src/model_training_google_research.py:289-293`

**Problem:**
```python
val_ds = VN30LastWindowDataset(
    all_series,
    context_len=context_len,
    horizon_len=horizon_len
)  # ← Only 30 samples (1 per stock)
```

**Impact:**
- Val loss based on only 30 samples (too few!)
- Not representative of full test period
- High variance in validation metrics

---

## Current Data Flow (BROKEN)

```
┌─────────────────────────────────────────────────────────────────┐
│                    FULL DATA (2009 → 2026)                      │
│                         VCB: 4000 obs                           │
└─────────────────────────────────────────────────────────────────┘

TRAIN (VN30TimeSeriesDataset):
- 6000 random windows from ANYWHERE (2009-2026)
- Sample 1: VCB[2015-06 → 2015-06+1]
- Sample 2: VCB[2025-01 → 2025-01+1]  ← DATA LEAKAGE!
- Sample 3: VIC[2020-03 → 2020-03+1]

VAL (VN30LastWindowDataset):
- 30 samples = last window ONLY
- Very limited validation data
```

---

## Target Data Flow (FIXED)

```
┌─────────────────────────────────────────────────────────────────┐
│                    FULL DATA (2009 → 2026)                      │
│                         VCB: 4000 obs                           │
└─────────────────────────────────────────────────────────────────┘
                      split_point = 3200 (80%)
┌────────────────────────────┬──────────────────────────────────┐
│     TRAIN REGION           │       TEST REGION                 │
│   [0 → 3200)               │   [3200 → 4000)                   │
│   2009 → ~2023             │   ~2023 → 2026                    │
│                            │                                  │
│ Context windows from       │ Context windows from             │
│ FIRST 80% only             │ LAST 20% only                     │
│ 200 samples/stock          │ 50 samples/stock                  │
└────────────────────────────┴──────────────────────────────────┘

✅ NO OVERLAP - Proper temporal split!
✅ Train on past, Test on future
```

---

## Fix Strategy

### Recommended Approach: Use `vn30_dataset.py` (VN30MultiStockDataset)

**Why:**
- ✅ Already implements proper 80/20 temporal split
- ✅ Train samples ONLY from first 80%
- ✅ Test samples ONLY from last 20%
- ✅ NO data leakage
- ✅ Sufficient test data (50 samples × 30 stocks = 1500)

### Implementation Options

| Option | Description | Effort | Risk |
|--------|-------------|--------|------|
| **A. Fix bugs + retrain** | Fix `model_training_fixed.py` bug, deprecate `model_training_google_research.py`, retrain | Medium | Medium - Need full retrain |
| **B. Switch to vn30_dataset.py** | Use existing `VN30MultiStockDataset`, update training script | Low | Low - Dataset already correct |
| **C. Create new training script** | Clean slate with proper split from start | High | Low - More code to maintain |

**Recommended: Option B** - Minimal code changes, uses proven correct implementation.

---

## Implementation Plan (Option B)

### Phase 1: Immediate Code Fixes

**1.1 Fix `model_training_fixed.py` BUG**
```python
# src/model_training_fixed.py:157
# BEFORE (BUGGY):
start = rng.integers(0, max_start + 1)

# AFTER (FIXED):
if mode == 'train':
    start = rng.integers(0, max_start + 1)
else:  # test mode
    start = rng.integers(min_start, max_start + 1)  # ← Use min_start!
```

**1.2 Deprecate `model_training_google_research.py`**
- Add warning header: "DEPRECATED: Data leakage issues - use vn30_dataset.py instead"
- Do NOT delete (keep for reference)

**1.3 Update `vn30_dataset.py` imports**
- Ensure `VN30MultiStockDataset` is easily accessible
- Add docstring explaining proper temporal split

---

### Phase 2: Training Script Update

**2.1 Create `src/train_timesfm_proper.py`**

```python
"""
TimesFM Training with PROPER Temporal Split
Uses VN30MultiStockDataset for 80/20 train/test split
NO data leakage - train on past, test on future
"""

from vn30_dataset import VN30MultiStockDataset, create_vn30_dataloaders
from model_training_google_research import TimesFMVN30Finetuner

def main():
    # Use proper temporal split dataset
    train_loader, test_loader = create_vn30_dataloaders()

    # Train with proper split
    finetuner = TimesFMVN30Finetuner()
    finetuner.train_model(train_loader, test_loader)

if __name__ == "__main__":
    main()
```

---

### Phase 3: Model Retraining

**3.1 Backup current models**
```bash
cp -r models/checkpoints models/checkpoints_backup_BEFORE_FIX
```

**3.2 Retrain with proper split**
```bash
python src/train_timesfm_proper.py
```

**3.3 Compare results**
- Training history: `experiments/training_history.json`
- Compare R², QLIKE, RMSE before/after fix

---

### Phase 4: Validation & Testing

**4.1 Verify temporal split**
```python
# Test script to verify no overlap
def verify_temporal_split():
    train_dataset = VN30MultiStockDataset(data, mode='train')
    test_dataset = VN30MultiStockDataset(data, mode='test')

    # Get all sample dates
    train_dates = get_all_sample_dates(train_dataset)
    test_dates = get_all_sample_dates(test_dataset)

    # Verify no overlap
    assert max(train_dates) < min(test_dates), "DATA LEAKAGE DETECTED!"
    print("✅ Temporal split verified - no overlap")
```

**4.2 Evaluate on true out-of-sample data**
- Test set = last 20% (~2023 → 2026)
- This is TRUE future prediction
- Metrics comparable to production

---

### Phase 5: HAR-RV Baseline Integration

**5.1 Use same temporal split for HAR-RV**
```python
# HAR-RV training should use same 80/20 split
split_point = int(len(data) * 0.8)
train_data = data.iloc[:split_point]
test_data = data.iloc[split_point:]
```

**5.2 Fair comparison**
- Both models tested on same last 20% period
- No data leakage in either model
- Comparable evaluation

---

## Success Metrics

✅ **Code Fixes:**
- `model_training_fixed.py` bug fixed
- Training script uses `VN30MultiStockDataset`

✅ **Data Verification:**
- Train samples: only from first 80%
- Test samples: only from last 20%
- No overlap verified

✅ **Model Performance:**
- New model trained with proper split
- R², QLIKE, RMSE, MSE on true out-of-sample data
- Performance comparable or better (more realistic)

✅ **HAR-RV Baseline:**
- Uses same temporal split
- Fair comparison with TimesFM
- Diebold-Mariano test on same test period

---

## Files to Modify

### New Files:
- `src/train_timesfm_proper.py` - Training script with proper split
- `tests/test_temporal_split.py` - Verification tests

### Modified Files:
- `src/model_training_fixed.py` - Fix BUG (line 157)
- `src/model_training_google_research.py` - Add deprecation notice

### Unchanged (Already Correct):
- `src/vn30_dataset.py` - Proper temporal split implementation

---

## Timeline

| Phase | Task | Estimated Time |
|-------|------|----------------|
| 1 | Code fixes | 30 min |
| 2 | Training script update | 45 min |
| 3 | Model retraining | 2-3 hours (GPU) |
| 4 | Validation & testing | 30 min |
| 5 | HAR-RV integration | 1 hour |
| **Total** | | **~5 hours** |

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Model performance worse after fix | Medium | Medium | Expected - more realistic evaluation |
| Retraining takes too long | Low | Low | Can use checkpoint from epoch 50 |
| HAR-RV baseline still has issues | Low | Low | Use same vn30_dataset.py approach |

---

## Next Steps

**For Review:**
1. Does this fix approach address all identified issues?
2. Is Option B (switch to vn30_dataset.py) the right choice?
3. Should we deprecate model_training_google_research.py or fix it too?

**After Approval:**
1. Implement Phase 1 (Code fixes)
2. Create verification tests
3. Retrain model
4. Update HAR-RV baseline plan

---

**Please review and provide feedback on this fix plan.**
