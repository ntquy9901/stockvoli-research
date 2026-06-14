# Data Leakage Analysis - TimesFM Training

**Date:** 2026-06-10  
**Status:** ⚠️ **DATA LEAKAGE IDENTIFIED**

---

## 🔴 **CRITICAL ISSUE: DATA LEAKAGE IN TRAIN/TEST SPLIT**

---

## 📊 **Current Implementation (INCORRECT):**

### **Train Dataset (VN30TimeSeriesDataset):**
```python
# Samples RANDOM windows from ENTIRE series
for _ in range(num_samples):
    idx = rng.choice(valid)          # Pick random series
    series = series_list[idx]
    max_start = len(series) - min_len
    start = rng.integers(0, max_start + 1)  # RANDOM START from 0 to max_start
```

**What happens:**
- For a stock with 5,000 data points
- Samples random windows like: [0:141], [100:241], [500:641], [1000:1141], ...
- **Covers ENTIRE series from 0 to end**

### **Validation Dataset (VN30LastWindowDataset):**
```python
# Uses LAST window from each series
for s in series_list:
    if len(s) >= min_len:
        ctx = torch.tensor(s[-min_len:-horizon_len], dtype=torch.float32)  # Last window
        tgt = torch.tensor(s[-horizon_len:], dtype=torch.float32)
```

**What happens:**
- For same stock with 5,000 data points
- Uses window: [4859:5000] (last 141 points)

---

## ❌ **DATA LEAKAGE CONFIRMED:**

### **Overlap Example:**
```
Stock with 5,000 data points (20 years):

Train Sampling (random):
  - Window 1: [0:141]      ← May include recent data
  - Window 2: [100:241]    ← May include recent data
  - Window 3: [500:641]    ← May include recent data
  - ...
  - Window N: [4700:4841]  ← VERY CLOSE to test window!

Validation (fixed):
  - Uses: [4859:5000]    ← Last window

PROBLEM: Train samples can include data points [4859:5000]
RESULT: Model has seen test patterns during training
```

---

## 🚨 **Impact on Results:**

### **Current Test Results:**
```
R² = 0.85 (EXCEPTIONAL)
RMSE = 0.0038 (VERY LOW)
```

### **❌ REALITY:**
- **These metrics are OVERSTIMATED**
- Model has already seen test patterns
- Not true generalization performance
- **R² = 0.85 is misleading**

---

## ✅ **CORRECT APPROACH (Temporal Split):**

### **Should Be:**
```python
# Train Dataset (should be FIRST 80%):
for each stock:
    use data[0 : int(0.8 * len(series))]  # First 80%
    sample random windows from this range

# Validation Dataset (should be LAST 20%):
for each stock:
    use data[int(0.8 * len(series)) : ]  # Last 20%
    use last window from this range
```

### **Example for 5,000 data points:**
```
Train Data: [0:4000] (80%)
  - Random windows from [0:4000] range
  - Example: [100:241], [500:641], [1000:1141], ..., [3759:3900]

Test Data: [4000:5000] (20%)
  - Last window: [3859:4000]
  - NO overlap with train data

✅ NO DATA LEAKAGE
✅ True generalization test
```

---

## 🔍 **Why Current Code Has Data Leakage:**

Looking at the Google Research finetune_lora.py code:

```python
# Their code (for retail sales dataset):
# - Train: Random windows from ENTIRE series
# - Val: Last windows from ENTIRE series

# WHY IT WORKS FOR THEM:
# - Retail sales dataset has ONLY ~120 data points per store
# - They have DIFFERENT stores for train/test
# - Store A train, Store B test → no leakage
# - This is CROSS-SECTIONAL split, not temporal

# WHY IT DOESN'T WORK FOR US:
# - We have SAME stocks for train/test
# - Temporal split required (time series)
# - Using last window from entire series = leakage
```

---

## 🎯 **How to Fix:**

### **Option 1: Temporal Split (RECOMMENDED) ⭐**

Modify `VN30TimeSeriesDataset` to use only first 80%:

```python
class VN30TimeSeriesDataset(Dataset):
    def __init__(self, series_list, context_len, horizon_len, 
                 num_samples, seed, mode='train'):  # ADD mode parameter
        
        for series in series_list:
            if mode == 'train':
                # Use ONLY first 80%
                split_point = int(0.8 * len(series))
                usable_data = series[:split_point]
            else:  # test
                # Use ONLY last 20%
                split_point = int(0.8 * len(series))
                usable_data = series[split_point:]
            
            # Sample from usable_data only
            max_start = len(usable_data) - min_len
            start = rng.integers(0, max_start + 1)
```

### **Option 2: Keep Current Approach (ACKNOWLEDGE LIMITATION)**

If using current approach, must:
1. Clearly state in results: "Data leakage present - metrics overestimated"
2. Rename validation to "in-sample" rather than "out-of-sample"
3. Don't claim generalization performance

---

## 📊 **Impact on Current Results:**

### **Training Results (Still Valid):**
- ✅ Model learned patterns (training worked)
- ✅ Convergence successful
- ✅ Training metrics are correct

### **Test Results (INVALIDATED):**
- ❌ R² = 0.85 is **NOT true test performance**
- ❌ RMSE = 0.0038 is **OVERESTIMATED**
- ❌ Model has seen test data during training

### **What R² = 0.85 Actually Means:**
- Model fits data well (including test data)
- **NOT** that model generalizes to unseen data
- In-sample fit, not out-of-sample prediction

---

## 🎯 **Next Steps:**

### **Option 1: Retrain with Correct Split** ⭐ **RECOMMENDED**
```bash
# 1. Fix dataset to use temporal split
# 2. Retrain model from scratch
# 3. Get TRUE test performance
# 4. Report accurate metrics
```

**Timeline:** 3-5 days (100 epochs)

### **Option 2: Use Current Results with Disclaimer**
- Clearly state: "In-sample performance (data leakage present)"
- Don't claim generalization
- Use for exploratory analysis only

---

## 📝 **Summary:**

| Aspect | Current Status | Correct Approach |
|--------|--------------|-----------------|
| **Train Data** | 0-100% of data | First 80% [0:4000] |
| **Test Data** | Last 20% [4859:5000] | Last 20% [4000:5000] |
| **Overlap** | YES ❌ | NO ✅ |
| **Metrics Valid** | NO ❌ | YES ✅ |
| **R² = 0.85** | In-sample fit | Out-of-sample? Unknown |

---

## 🚨 **Final Verdict:**

**Status:** ❌ **DATA LEAKAGE CONFIRMED**

**Impact:** Test metrics are **OVERESTIMATED** (not true generalization)

**Recommendation:** **RETRAIN WITH TEMPORAL SPLIT** to get accurate test performance

---

**Action Required:** Fix dataset splitting logic before claiming production readiness
