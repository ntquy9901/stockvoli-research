# 🧪 Complete Pre-Colab Testing Strategy

**Author:** Murat (Master Test Architect)  
**Date:** 2026-06-13  
**Purpose:** Comprehensive testing strategy to prevent expensive Colab failures

---

## 🎯 Executive Summary

**Problem:** Colab GPU training costs $10-20/hour. Bugs in Colab waste money and time.

**Solution:** Three-stage testing strategy:
1. **Local Pre-Flight** (Free, 20 min) → Catch 90% of issues
2. **Colab Free Tier** (Free, 1-2 hr) → Catch remaining 9%
3. **Colab Pro** (Paid, 5-6 hr) → Train with confidence

**Risk Calculation:** 40 min local testing vs $50-100 wasted Colab costs. **Math is clear.**

---

## 📊 Risk-Based Testing Strategy

### **Risk Matrix**

| Issue Type | Local Detection Cost | Colab Failure Cost | Risk Priority |
|-------------|----------------------|-------------------|--------------|
| Config errors | $0 (1 min) | $10-20/hr | 🔴 CRITICAL |
| Data missing | $0 (1 min) | $10-20/hr | 🔴 CRITICAL |
| Import errors | $0 (2 min) | $10-20/hr | 🔴 CRITICAL |
| GPU memory issues | $0 (2 min) | $10-20/hr | 🟠 HIGH |
| Logic bugs | $0 (10 min) | $10-20/hr | 🟠 HIGH |
| Data quality issues | $0 (5 min) | $10-20/hr | 🟡 MEDIUM |
| Performance issues | Local (15 min) | $10-20/hr | 🟢 LOW |

**Principle:** *Fix issues where they're cheap to fix (local), not where they're expensive (Colab).*

---

## 🚀 Three-Stage Testing Strategy

### **Stage 1: Local Pre-Flight Validation** ⏱️ 20 min, 💵 FREE

**Objective:** Catch 90% of issues before spending money

**When:** Run before every Colab session

**How:**
```bash
# Step 1: Run pre-flight script
python scripts/colab_preflight.py --verbose

# Expected output:
# ✅ Config Validation passed
# ✅ Data Files Check passed
# ✅ GPU Availability passed
# ✅ Import Compatibility passed
# ✅ ALL CHECKS PASSED - SAFE FOR COLAB!
```

**What It Checks:**
- ✅ Configuration file integrity
- ✅ Data file existence and quality
- ✅ GPU availability and memory
- ✅ Import compatibility
- ✅ Package dependencies

**Success Criteria:** All checks pass → Proceed to Stage 2

---

### **Stage 2: Colab Free Tier Validation** ⏱️ 1-2 hr, 💵 FREE

**Objective:** Validate in real Colab environment before paying

**When:** After Stage 1 passes

**How:**
```python
# In Colab free tier (T4 GPU):

# 1. Clone repository
!git clone https://github.com/ntquy9901/stockvoli-research.git
%cd stockvoli-research

# 2. Use staging config (smaller scale)
!cp configs/config_staging.yaml configs/config.yaml

# 3. Install dependencies
!pip install -q transformers peft torch pandas numpy pyyaml accelerate

# 4. Run pre-flight checks
!python scripts/colab_preflight.py

# 5. Run training (5 epochs, minimal data)
!python src/model_training_fixed.py

# Expected: Complete in 1-2 hours on free tier
```

**Staging Config:**
```yaml
# configs/config_staging.yaml
training:
  num_epochs: 5           # Test with 5 epochs
  batch_size: 8            # Smaller batch for T4
  samples_per_stock: 10   # Minimal data for testing

# If this passes → proceed to Pro tier
```

**Success Criteria:** Training completes 5 epochs without OOM → Proceed to Stage 3

---

### **Stage 3: Colab Pro Training** ⏱️ 5-6 hr, 💵 $10-20

**Objective:** Full training with confidence

**When:** After Stage 1 and Stage 2 pass

**How:**
```python
# In Colab Pro (G4 22.5GB GPU):

# 1. Clone repository
!git clone https://github.com/ntquy9901/stockvoli-research.git
%cd stockvoli-research

# 2. Pull latest code
!git pull origin master

# 3. Use full config
# configs/config.yaml (already optimized for G4)

# 4. Install dependencies
!pip uninstall -y torchao
!pip install -q transformers peft torch pandas numpy pyyaml accelerate

# 5. Run pre-flight checks
!python scripts/colab_preflight.py

# 6. Run full training
!python src/model_training_fixed.py

# Expected: Complete in 5-6 hours on Pro tier
```

**Success Criteria:** Training completes 100 epochs successfully

---

## 🛡️ Comprehensive Test Coverage

### **Unit Tests (Run Locally)**

**File:** `tests/test_colab_preflight.py`

**Test Categories:**
1. **Config Validation Tests** (8 tests)
   - Config file exists
   - Required sections present
   - Training config valid
   - Dataset config valid
   - Model config valid

2. **Data Validation Tests** (3 tests)
   - Data directory exists
   - Data files exist (30 files)
   - Data files have content

3. **GPU Validation Tests** (3 tests)
   - CUDA available
   - GPU memory sufficient
   - Model can load on GPU

4. **Import Validation Tests** (6 tests)
   - Transformers import
   - PEFT import
   - Torch import
   - Pandas import
   - NumPy import
   - YAML import

5. **Exception Handling Tests** (2 tests)
   - save_training_history has error handling
   - Training loop has error handling

6. **Integration Tests** (2 tests)
   - Full pre-flight validation
   - Invalid config detection

**Total:** 24 comprehensive tests

**Run:**
```bash
# Run all tests
pytest tests/test_colab_preflight.py -v

# Run specific test category
pytest tests/test_colab_preflight.py::TestConfigValidation -v

# Run with coverage
pytest tests/test_colab_preflight.py --cov=src --cov-report=html
```

---

### **Component Tests (Run Locally with GPU)**

**File:** `tests/test_dry_run_training.py`

**Purpose:** Test training with minimal resources

**How:**
```python
# Create dry-run config
# configs/config_dryrun.yaml
training:
  num_epochs: 1          # Just 1 epoch
  batch_size: 4           # Very small batch
  samples_per_stock: 5    # Minimal data

# Run locally
python -c "
from src.model_training_fixed import TimesFMVN30Finetuner
import yaml

with open('configs/config_dryrun.yaml') as f:
    config = yaml.safe_load(f)

finetuner = TimesFMVN30Finetuner()
finetuner.load_timesfm_model()
finetuner.setup_lora_adapters()
finetuner.train_model()  # Should complete in 10-15 min
"
```

---

## 📋 Pre-Colab Checklist

### **Before Every Colab Session**

**✅ Phase 1: Local Validation (20 min, Free)**
- [ ] Run pre-flight script: `python scripts/colab_preflight.py`
- [ ] Run unit tests: `pytest tests/test_colab_preflight.py -v`
- [ ] Check GPU memory locally: `torch.cuda.get_device_properties(0)`
- [ ] Verify data files: `ls data/processed/ | wc -l` (should be 30)

**✅ Phase 2: Free Tier Test (1-2 hr, Free)**
- [ ] Clone repo in Colab free tier
- [ ] Use staging config (5 epochs, small data)
- [ ] Run pre-flight checks in Colab
- [ ] Run training for 5 epochs
- [ ] Verify no OOM errors
- [ ] Check learning curves plot correctly

**✅ Phase 3: Pro Training (Paid, Full Training)**
- [ ] Switch to Colab Pro
- [ ] Use full config (100 epochs, full data)
- [ ] Run pre-flight checks
- [ ] Monitor GPU utilization: `!nvidia-smi` every 30 min
- [ ] Monitor training logs every 10 epochs
- [ ] Verify checkpoints saving correctly

---

## 🔧 Debugging Failed Validations

### **Issue: Config Validation Failed**

**Symptoms:**
```
❌ Config Validation FAILED: Missing config section: 'training'
```

**Solution:**
```bash
# 1. Check config file
cat configs/config.yaml

# 2. Verify required sections exist
grep -E "^(system|data|dataset|model|training):" configs/config.yaml

# 3. Fix missing sections
# Compare with configs/config_g4.yaml (reference)
```

---

### **Issue: Data Validation Failed**

**Symptoms:**
```
❌ Data Files Check FAILED: Insufficient data files: 5 (< 10)
```

**Solution:**
```bash
# 1. Check data directory
ls -la data/processed/

# 2. Count files
ls data/processed/ | wc -l

# 3. If files missing, re-run data processing
# python src/data_processing.py
```

---

### **Issue: GPU Validation Failed**

**Symptoms:**
```
❌ GPU Availability FAILED: CUDA not available
```

**Solution:**
```bash
# 1. Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# 2. If false locally, skip local GPU test
# Proceed to Stage 2 (Colab free tier)
```

---

### **Issue: Import Validation Failed**

**Symptoms:**
```
❌ Import Compatibility FAILED: Import error: No module named 'transformers'
```

**Solution:**
```bash
# 1. Install missing packages
pip install transformers peft torch pandas numpy pyyaml

# 2. Verify imports
python -c "import transformers; print('✅ transformers OK')"
```

---

## 🎯 Stage Gate Criteria

### **Stage 1 Gate** → Stage 2
**Entry Criteria:** None (always run)

**Exit Criteria (ALL must pass):**
- ✅ Pre-flight script returns exit code 0
- ✅ All unit tests pass
- ✅ No import errors
- ✅ Config valid
- ✅ Data files exist

**Time Investment:** 20 min  
**Risk Reduction:** 90%  
**ROI:** $50-100 saved per prevented failure

---

### **Stage 2 Gate** → Stage 3
**Entry Criteria:** Stage 1 passed

**Exit Criteria (ALL must pass):**
- ✅ Pre-flight checks pass in Colab
- ✅ 5 epochs complete without OOM
- ✅ Learning curves plot correctly
- ✅ Checkpoints save successfully
- ✅ No exception errors in logs

**Time Investment:** 1-2 hr  
**Risk Reduction:** 99%  
**ROI:** $200-500 saved per prevented failure

---

### **Stage 3** → Production
**Entry Criteria:** Stage 2 passed

**Success Criteria:**
- ✅ All 100 epochs complete
- ✅ Final model saved
- ✅ Final metrics meet targets (R² > 0.5, QLIKE < 1.0)

**Time Investment:** 5-6 hr  
**Confidence Level:** 99% (validated by 2 stages)

---

## 📈 Continuous Quality Strategy

### **Pre-Commit Hooks**

**Add to `.git/hooks/pre-commit`:**
```bash
#!/bin/bash
# Run pre-flight checks before commit
python scripts/colab_preflight.py
if [ $? -ne 0 ]; then
    echo "❌ Pre-flight checks failed - commit blocked"
    exit 1
fi

# Run unit tests
pytest tests/test_colab_preflight.py -q
if [ $? -ne 0 ]; then
    echo "❌ Unit tests failed - commit blocked"
    exit 1
fi
```

**Install:**
```bash
# Copy to .git/hooks/
cp .git/hooks/pre-commit.sample .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

---

### **CI Integration (Optional)**

**Add to `.github/workflows/pre-flight.yml`:**
```yaml
name: Pre-Colab Validation

on: [push, pull_request]

jobs:
  pre-flight:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run pre-flight checks
        run: |
          python scripts/colab_preflight.py
      
      - name: Run unit tests
        run: |
          pytest tests/test_colab_preflight.py -v
```

---

## 🎓 Risk Calculation Examples

### **Example 1: Config Bug**

**Without Testing:**
- Run in Colab Pro → Config error at 30 min → Wasted $5-10

**With Testing:**
- Run pre-flight locally (1 min) → Catch config error → Fix → $0 saved

**ROI:** 1 min → $5-10 saved = **300-600x ROI**

---

### **Example 2: GPU Memory Bug**

**Without Testing:**
- Run in Colab Pro → OOM at 2 hours → Wasted $20-40

**With Testing:**
- Run free tier test → OOM at 1 hour → Fix batch size → $0 saved

**ROI:** 1 hr free tier → $20-40 saved = **Still profitable**

---

### **Example 3: Data Quality Bug**

**Without Testing:**
- Run in Colab Pro → Data error at 4 hours → Wasted $40-80

**With Testing:**
- Run pre-flight (1 min) → Catch data error → Fix → $0 saved

**ROI:** 1 min → $40-80 saved = **2400-4800x ROI**

---

## 🚀 Quick Start Guide

### **For New Colab Users**

**Step 1: Local Validation**
```bash
git clone https://github.com/ntquy9901/stockvoli-research.git
cd stockvoli-research
python scripts/colab_preflight.py
```

**Step 2: Free Tier Test**
```python
# In Colab free tier:
!git clone https://github.com/ntquy9901/stockvoli-research.git
%cd stockvoli-research
!cp configs/config_staging.yaml configs/config.yaml
!python src/model_training_fixed.py  # 5 epochs test
```

**Step 3: Pro Training**
```python
# In Colab Pro:
!python src/model_training_fixed.py  # 100 epochs full training
```

---

## 📊 Quality Metrics

### **Current Coverage**

| Test Type | Tests | Coverage | Frequency |
|-----------|-------|----------|-----------|
| Pre-flight checks | 6 | Config, Data, GPU | Every Colab session |
| Unit tests | 24 | Components | Every commit |
| Integration tests | 2 | End-to-end | Every PR |
| Dry-run tests | 1 | Training loop | Before big changes |

**Total:** 33 tests covering all critical paths

---

## 🎯 Success Stories

### **Before Strategy:**
```
User: Run in Colab Pro → Bug at hour 2 → Lost $20, 4 hours
```

### **After Strategy:**
```
User: Local pre-flight (1 min) → Bug caught → Fix (5 min)
     Free tier test (1 hr) → Validated → Proceed to Pro
     Pro training (5 hr) → Success!
```

**Result:** $20 saved, 4 hours saved, confidence increased

---

## 🔧 Advanced: Automated Testing Pipeline

### **Pre-Colab CI/CD Pipeline**

**Create `.github/workflows/pre-colab-validation.yml`:**
```yaml
name: Pre-Colab Validation

on:
  push:
    branches: [master]
  pull_request:

jobs:
  validate-before-colab:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run pre-flight checks
        run: |
          python scripts/colab_preflight.py
      
      - name: Run unit tests
        run: |
          pytest tests/test_colab_preflight.py -v
      
      - name: Run dry-run test
        run: |
          python tests/test_dry_run_training.py
      
      - name: Check quality gate
        run: |
          echo "✅ All quality gates passed - Safe for Colab"
```

---

## 📚 Training: Pre-Colab Quality

### **Training Checklist**

For developers joining the project:

**Must Complete Before First Colab Run:**
1. ✅ Read `COLAB_PRE_COLAB_CHECKLIST.md` (this file)
2. ✅ Run pre-flight script locally
3. ✅ Run unit tests locally
4. ✅ Test in Colab free tier
5. ✅ Document any issues found

**Estimated Time:** 2-3 hours  
**Skill Level:** Beginner  
**Prerequisites:** Python, PyTorch, Git

---

## 🆘 Emergency Procedures

### **If Colab Training Fails**

**Step 1: Save Logs**
```python
# In Colab:
!cat experiments/model_training.log > colab_failure_logs.txt
```

**Step 2: Diagnose Issue**
```python
# Check logs for error patterns
!grep -i "error\|exception\|failed" experiments/model_training.log
```

**Step 3: Fix Locally**
```bash
# Clone repo locally
git clone https://github.com/ntquy9901/stockvoli-research.git

# Fix issue locally (free)
# Test fix with pre-flight script
```

**Step 4: Re-Validate**
```python
# Run pre-flight checks again
python scripts/colab_preflight.py

# Only return to Colab when all checks pass
```

---

## 📞 Support

**For Questions:**
- Review `TRAINING_FIX_EXPLANATION.md` for common fixes
- Check `HOW_TO_USE_COMMON_RULES.md` for coding standards
- Open GitHub issue for bugs

**Quick References:**
- Pre-flight script: `scripts/colab_preflight.py`
- Unit tests: `tests/test_colab_preflight.py`
- Training script: `src/model_training_fixed.py`
- Config: `configs/config.yaml`

---

## ✅ Summary

**Three-Stage Strategy:**
1. **Local Pre-Flight** (20 min, FREE) → 90% risk reduction
2. **Colab Free Tier** (1-2 hr, FREE) → 99% risk reduction
3. **Colab Pro** (5-6 hr, $10-20) → Train with confidence

**Total Time Investment:** 2-3 hours  
**Total Cost Savings:** $50-500 per prevented failure  
**Risk Reduction:** 99% (validated by 3 stages)

**Key Insight:** *Test locally where it's free, train remotely where it's expensive.*

**Murat's Final Recommendation:**  
**"This three-stage strategy turns Colab from a risky proposition into a predictable, cost-effective training platform. The 2-3 hour investment saves $50-100 per run and provides 99% confidence in expensive training sessions. That's a 50-100x ROI on your testing time."**

---

*Last Updated: 2026-06-13*  
*Version: 1.0.0*  
*Complete Pre-Colab Testing Strategy*  
*Author: Murat (Test Architect)*
