# Common Pitfalls & Solutions - Lessons Learned

**Last Updated:** 2026-06-11
**Purpose:** Document real-world issues and how to avoid them
**Focus:** Financial ML and Time Series

---

## 🎯 **Overview**

This document captures real-world issues encountered during TimesFM fine-tuning for Vietnamese stocks. Each pitfall includes:
- Description of the problem
- How we detected it
- Impact on results
- Solution implemented
- Prevention strategies

---

## 🚨 **Critical Issue #1: Data Leakage (The Big One)**

### **Problem Description:**

```python
# WRONG CODE (from model_training_google_research.py)
class VN30TimeSeriesDataset(Dataset):
    def __init__(self, series_list, context_len, horizon_len, num_samples):
        for _ in range(num_samples):
            idx = rng.choice(valid)
            series = series_list[idx]
            max_start = len(series) - min_len
            start = rng.integers(0, max_start + 1)  # ❌ RANDOM from 0 to end
            # Samples from ENTIRE series, including test period
```

**What happened:**
- Training sampled randomly from [0:5000] of stock data
- Testing used last window [4872:5000] from same data
- Training could include data points [4872:5000]
- **Result:** Model saw test patterns during training

### **How We Detected It:**

1. **Red flag:** R² = 0.85 seemed too good
2. **Investigation:** Examined dataset code
3. **Confirmation:** Verified train/test overlap
4. **Impact:** 38.9% overestimation of performance

### **Impact:**

| Metric | With Leakage | True Value | Inflation |
|--------|--------------|------------|-----------|
| R² | 0.8502 | 0.5193 | +38.9% |
| RMSE | 0.0038 | 0.0062 | -38.7% |
| MSE | 0.000014 | 0.000038 | -63.2% |

### **Solution:**

```python
# CORRECT CODE (temporal split)
class VN30TimeSeriesDataset(Dataset):
    def __init__(self, series_list, context_len, horizon_len,
                 num_samples, mode='train'):
        split_ratio = 0.8

        for series in series_list:
            split_point = int(len(series) * split_ratio)

            if mode == 'train':
                usable_data = series[:split_point]  # ✅ First 80%
            else:  # test
                usable_data = series[split_point:]  # ✅ Last 20%

            # Sample ONLY from usable range
            max_start = len(usable_data) - (context_len + horizon_len)
            start = rng.integers(0, max_start + 1)
```

### **Prevention:**

```python
# Add to training code
def verify_no_leakage(train_data, test_data):
    """Always verify before training"""
    for train, test in zip(train_data, test_data):
        assert train.index[-1] < test.index[0], \
            "DATA LEAKAGE DETECTED!"

verify_no_leakage(train_series, test_series)
```

### **Lesson:**

> **ALWAYS use temporal split for time series. NEVER random sample.**

---

## ⚠️ **Issue #2: Bitsandbytes Compatibility (Windows)**

### **Problem Description:**

```python
# ERROR on Windows
AttributeError: module 'bitsandbytes' has no attribute 'nn'

# Context: Loading TimesFM with PEFT on Windows
```

**What happened:**
- PEFT library tried to use bitsandbytes for quantization
- bitsandbytes has compatibility issues on Windows
- Model loading failed completely

### **How We Detected It:**

1. **Error:** Model loading crashed with AttributeError
2. **Context:** Windows environment, PEFT library
3. **Investigation:** Traced to bitsandbytes import

### **Impact:**

- **Severity:** HIGH (complete failure)
- **Scope:** All model loading operations
- **Workaround:** Required immediately

### **Solution:**

```python
# Monkey-patch to disable bitsandbytes
import sys
import types
import importlib.util

# Patch import system
original_find_spec = importlib.util.find_spec

def patched_find_spec(name, package=None):
    if name and 'bitsandbytes' in name:
        return None
    return original_find_spec(name, package)

importlib.util.find_spec = patched_find_spec

# Create fake module
fake_bnb = types.ModuleType('bitsandbytes')
fake_bnb.__path__ = []
fake_bnb.__file__ = '<disabled>'
sys.modules['bitsandbytes'] = fake_bnb

# Patch PEFT
try:
    import peft.import_utils as peft_utils
    peft_utils.is_bnb_available = lambda: False
    if hasattr(peft_utils, 'is_bnb_4bit_available'):
        peft_utils.is_bnb_4bit_available = lambda: False
    if hasattr(peft_utils, 'is_bnb_8bit_available'):
        peft_utils.is_bnb_8bit_available = lambda: False
except:
    pass
```

### **Prevention:**

```python
# Add to all model-loading scripts
# Check Windows compatibility
if sys.platform == 'win32':
    # Apply bitsandbytes workaround
    apply_bitsandbytes_patch()
```

### **Lesson:**

> **ALWAYS test cross-platform compatibility. Windows ≠ Linux.**

---

## 📊 **Issue #3: Data Format Inconsistency (Yahoo Finance)**

### **Problem Description:**

```python
# CRAWLED DATA (June 2026)
2026-06-01,62300.0,62800.0,61700.0,62200.0,2924300  # Raw VND prices

# EXISTING DATA (May 2026)
2026-05-29,63.0,63.0,61.8,62.0,5820000  # Normalized prices (thousands)
```

**What happened:**
- Yahoo Finance returns raw VND prices (62,300)
- Existing data uses normalized prices (62.0 = 62,000 VND)
- Mixing formats created analysis issues

### **How We Detected It:**

1. **Symptom:** Sudden price jumps in June data
2. **Investigation:** Checked raw crawled data
3. **Discovery:** Format mismatch (×1000 difference)

### **Impact:**

- **Severity:** MEDIUM (data quality issue)
- **Scope:** June 2026 data only
- **Fix:** Simple normalization

### **Solution:**

```python
# In crawler
def normalize_prices(data):
    """Convert raw VND to match existing format"""
    for col in ['open', 'high', 'low', 'close']:
        data[col] = (data[col] / 1000).round(2)
    return data
```

### **Prevention:**

```python
# Add validation to crawler
def validate_price_format(df, expected_range=(10, 100)):
    """Verify prices in expected range"""
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if df[col].max() > expected_range[1]:
            raise ValueError(
                f"Price format mismatch: {col} max = {df[col].max()}, "
                f"expected < {expected_range[1]}"
            )
```

### **Lesson:**

> **ALWAYS validate data formats match existing data.**

---

## 🔢 **Issue #4: JSON Serialization (NumPy Types)**

### **Problem Description:**

```python
# ERROR when saving results
TypeError: Object of type float32 is not JSON serializable

# Context: Saving metrics to JSON
json.dump({'R2': r2_score}, f)  # r2_score is numpy.float32
```

**What happened:**
- Metrics calculated as NumPy types (float32, float64)
- JSON can't serialize NumPy types
- Results saving failed

### **How We Detected It:**

1. **Error:** TypeError during JSON save
2. **Context:** Inference results saving
3. **Investigation:** NumPy types in dictionary

### **Impact:**

- **Severity:** LOW (save operation only)
- **Scope:** Results saving
- **Fix:** Simple type conversion

### **Solution:**

```python
# Convert NumPy types to native Python
serializable_results = {
    k: float(v) if isinstance(v, (np.float32, np.float64)) else v
    for k, v in results.items()
}

# Now save to JSON
with open('results.json', 'w') as f:
    json.dump(serializable_results, f)
```

### **Prevention:**

```python
# Helper function
def to_json_serializable(obj):
    """Convert NumPy types to JSON-serializable"""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Use always
json_dict = {k: to_json_serializable(v) for k, v in results.items()}
```

### **Lesson:**

> **ALWAYS convert NumPy types before JSON serialization.**

---

## 💻 **Issue #5: Windows Encoding Issues**

### **Problem Description:**

```python
# WARNING in logs
--- Logging error ---
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705'

# Context: Using emojis in log messages on Windows
logger.info(f"✅ Downloaded {len(data)} rows")
```

**What happened:**
- Used emojis (✅, ❌) in log messages
- Windows console default encoding can't handle Unicode
- Logging errors (though not fatal)

### **How We Detected It:**

1. **Symptom:** Logging errors in console
2. **Context:** Windows terminal, emoji usage
3. **Impact:** Not critical but annoying

### **Impact:**

- **Severity:** VERY LOW (cosmetic only)
- **Scope:** Log output
- **Fix:** Encoding workaround

### **Solution:**

```python
# Fix Windows encoding
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding='utf-8',
        errors='replace'
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer,
        encoding='utf-8',
        errors='replace'
    )
```

### **Prevention:**

```python
# Add to all scripts
import sys
if sys.platform == 'win32':
    fix_windows_encoding()
```

### **Lesson:**

> **ALWAYS handle encoding on Windows. Unicode ≠ Default.**

---

## 🔍 **Issue #6: Insufficient Test Data (June 2026)**

### **Problem Description:**

```bash
# ERROR when testing on June 2026 only
[ERROR] No stocks with sufficient June 2026 data
[INFO] Need at least 141 days of data per stock
[INFO] Current June 2026 data has 7 days
```

**What happened:**
- Wanted to test ONLY on June 2026 data (genuinely new)
- Only 7 trading days in June 2026 so far
- Need 141 days (128 context + 13 horizon) for predictions

### **How We Detected It:**

1. **Plan:** Test on June 2026 only
2. **Error:** Insufficient data
3. **Solution:** Use temporal split instead

### **Impact:**

- **Severity:** LOW (testing approach issue)
- **Scope:** June 2026-only testing
- **Alternative:** Temporal split (last 20%)

### **Solution:**

```python
# Use temporal split instead
# First 80% train, last 20% test (includes June 2026)
def temporal_split_test(series_list, train_ratio=0.8):
    train_data = [s[:int(len(s)*train_ratio)] for s in series_list]
    test_data = [s[int(len(s)*train_ratio):] for s in series_list]
    return train_data, test_data
```

### **Prevention:**

```python
# Check data sufficiency before testing
def check_test_data_sufficiency(test_data, min_required=141):
    """Verify test data has enough observations"""
    for series in test_data:
        if len(series) < min_required:
            raise ValueError(
                f"Insufficient test data: {len(series)} < {min_required}"
            )
```

### **Lesson:**

> **ALWAYS check data sufficiency before testing.**

---

## 📊 **Issue #7: Train/Val Gap Misinterpretation**

### **Problem Description:**

```
During Training:
  Train Loss: 0.11 → Val Loss: 0.55
  Gap: 5x (Val loss much higher than train)

Initial Interpretation: Severe overfitting

Reality: Different data distributions
  - Train: Random windows (more volatile)
  - Val: Last windows (smoother trends)
```

**What happened:**
- Large train/val gap (5x) during training
- Initially thought to be overfitting
- Actually was different data distributions

### **How We Detected It:**

1. **Observation:** Train loss 0.11, Val loss 0.55
2. **Investigation:** Checked train/val data distributions
3. **Discovery:** Different sampling strategies
4. **Confirmation:** Test performance was good (R² = 0.52)

### **Impact:**

- **Severity:** LOW (interpretation issue only)
- **Scope:** Training monitoring
- **Reality:** Not overfitting, just different distributions

### **Solution:**

```python
# Monitor train/val gap but interpret carefully
def interpret_train_val_gap(train_loss, val_loss):
    """Interpret train/val gap correctly"""
    gap = val_loss / train_loss

    if gap > 5.0:
        # Could be:
        # 1. Overfitting (check test performance)
        # 2. Different distributions (likely)
        # 3. Data leakage (verify)

        print(f"Gap = {gap:.2f}x")
        print("Possible causes:")
        print("  1. Overfitting (check test performance)")
        print("  2. Different data distributions (likely)")
        print("  3. Data leakage (verify)")

    return gap
```

### **Prevention:**

```python
# Always check test performance
# Good test performance + large train/val gap = OK
# Poor test performance + large train/val gap = Overfitting
```

### **Lesson:**

> **Train/val gap ≠ Overfitting. Check test performance.**

---

## 🎯 **Summary of Pitfalls**

### **Critical (High Impact):**

| # | Issue | Impact | Detection | Prevention |
|---|-------|--------|-----------|------------|
| 1 | Data Leakage | 38.9% overestimation | Verify temporal split | Always use temporal split |
| 2 | Bitsandbytes | Complete failure | Test on Windows | Cross-platform testing |

### **Important (Medium Impact):**

| # | Issue | Impact | Detection | Prevention |
|---|-------|--------|-----------|------------|
| 3 | Data Format | Data quality issues | Validate ranges | Format validation |
| 4 | NumPy Types | Save failures | Unit tests | Type conversion |

### **Minor (Low Impact):**

| # | Issue | Impact | Detection | Prevention |
|---|-------|--------|-----------|------------|
| 5 | Encoding | Cosmetic | Windows testing | Encoding handling |
| 6 | Insufficient Data | Can't test | Check length | Data sufficiency checks |
| 7 | Gap Interpretation | Misunderstanding | Test performance | Verify with test data |

---

## ✅ **Prevention Checklist**

### **Before Training:**

- [ ] **Verify temporal split** (no data leakage)
- [ ] **Test cross-platform** (Windows/Linux)
- [ ] **Validate data formats** (consistency check)
- [ ] **Check data sufficiency** (minimum 1000 obs)
- [ ] **Set random seeds** (reproducibility)

### **During Training:**

- [ ] **Monitor train/val gap** (interpret correctly)
- [ ] **Track GPU memory** (avoid OOM)
- [ ] **Save checkpoints** (every N epochs)
- [ ] **Log comprehensive metrics** (not just loss)

### **After Training:**

- [ ] **Test on future data** (true out-of-sample)
- [ ] **Verify no leakage** (train_end < test_start)
- [ ] **Convert NumPy types** (before JSON save)
- [ ] **Document everything** (issues, solutions)

---

## 📚 **Key Takeaways**

### **Critical Lessons:**

1. **Data leakage is devastating** (38.9% impact)
   - Always use temporal split
   - Verify no overlap
   - Test on future data

2. **Cross-platform compatibility matters**
   - Windows ≠ Linux
   - Test on target platform
   - Handle encoding issues

3. **Data validation is essential**
   - Check formats match
   - Verify ranges
   - Ensure sufficiency

4. **Interpret metrics correctly**
   - Train/val gap ≠ overfitting
   - Check test performance
   - Understand distributions

### **Remember:**

> **Prevention > Detection > Resolution**
> **Test early, test often, test on target platform**

---

## 🔗 **Related Documentation**

- [Time-Series-ML-Fundamentals.md](./01-Time-Series-ML-Fundamentals.md) - Data leakage prevention
- [Data-Processing-Best-Practices.md](./02-Data-Processing-Best-Practices.md) - Data validation
- [Model-Training-Guidelines.md](./03-Model-Training-Guidelines.md) - Training best practices
- [Testing-Validation-Strategy.md](./04-Testing-Validation-Strategy.md) - Validation methodology

---

**Status:** ✅ Lessons Documented
**Purpose:** Help others avoid our mistakes
**Last Updated:** 2026-06-11

---

*These pitfalls cost us significant debugging time. Learn from our mistakes and avoid these issues in your projects!*

**Most Critical Lesson:**
> **ALWAYS use temporal split for time series. Data leakage can inflate metrics by 38.9%!**
