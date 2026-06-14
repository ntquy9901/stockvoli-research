# Archived Files - TimesFM VN30 Project

**Date Archived:** 2026-06-09  
**Reason:** Architecture cleanup - removing obsolete implementations before starting proper TimesFM fine-tuning

## What's Archived Here

### 🔴 **Old Implementations (4 Failed Attempts)**

These files represent previous implementation attempts that were identified as **technically incorrect** in the research findings:

1. **Root Level Scripts:**
   - `actual_timesfm_finetune.py` - Attempted TimesFM implementation
   - `advanced_finetune.py` - Advanced fine-tuning attempt
   - `real_finetune.py` - Real data fine-tuning attempt
   - `timesfm_correct_finetuning.py` - Correction attempt
   - `timesfm_finetuning.py` - Initial TimesFM fine-tuning
   - `timesfm_quick_finetune.py` - Quick fine-tuning demo
   - `timesmf_real_finetune.py` - **CRITICAL: Used custom transformer, NOT actual TimesFM**
   - `timesmf_transformer_finetune.py` - **CRITICAL: Custom transformer architecture**

2. **Source Files:**
   - `src/actual_finetune.py` - Old actual fine-tuning implementation
   - `src/data_preprocessing.py` - Old data processing approach
   - `src/model_evaluation_deployment.py` - Old evaluation/deployment code
   - `src/statistical_validation.py` - Old statistical validation
   - `src/timesfm_incremental.py` - Old incremental learning approach
   - `src/utils.py` - Old utility functions

3. **Test Files:**
   - `tests/test_data_preprocessing.py` - Tests for old data processing
   - `tests/test_evaluation_deployment.py` - Tests for old evaluation
   - `tests/test_statistical_validation.py` - Tests for old validation
   - `tests/test_timesfm.py` - Tests for old TimesFM implementation

### 🟡 **Demos & Utilities**

- `quick_demo.py` - Quick demonstration script
- `quick_finetune_demo.py` - Quick fine-tuning demo
- `run_simple_test.py` - Simple test runner
- `START_HERE.py` - Old starter file
- `calculate_r2.py` - Old R² calculation (superseded by proper metrics)
- `QUICK_START.md` - Old quick start guide

### 🟠 **Old Models & Artifacts**

- `models/demo_finetuned_model.pt` - Demo fine-tuned model
- `models/real_finetuned_model.pkl` - Real fine-tuned model (pickle)
- `models/timesfm_finetuned.pt` - Large fine-tuned model (2.7MB)
- `models/timesfm_finetuned/` - Directory with fine-tuned artifacts
- `models/demo_training_history.json` - Demo training history
- `models/real_finetuned_metadata.json` - Real model metadata
- `models/timesfm_info.json` - TimesFM model info
- `vcb_analysis.png` - Old VCB analysis chart

### 📊 **Old Configuration**

- `config_old.yaml` - Old configuration file with:
  - ❌ AdamW optimizer (should be SGD)
  - ❌ Learning rate 0.00001 (should be 1e-4)
  - ❌ Complex incremental learning setup
  - ❌ Moirai 2.0 configuration (not needed)
  - ❌ MLFlow tracking (should be simple JSON)

### 📓 **Old Notebooks**

- `notebooks/complete_system_demo.ipynb` - Old system demo notebook

## Why These Were Archived

### **Critical Architecture Issues Identified:**

1. **❌ Custom Transformers vs Actual TimesFM**
   - Most implementations used `TimesFMTransformer(nn.Module)` custom classes
   - **Required:** Actual `google/timesfm-2.5-200m-transformers` from HuggingFace
   - **Impact:** No access to Google's pre-trained time series patterns

2. **❌ Wrong Optimizer Choice**
   - Old implementations used AdamW
   - **Required:** SGD with momentum=0.9 for financial data
   - **Impact:** Poor convergence on noisy financial time series

3. **❌ Missing Financial Data Handling**
   - No log transformation for extreme events
   - Missing realized volatility calculations
   - No financial-specific preprocessing
   - **Impact:** NaN losses during market crashes, poor model stability

4. **❌ Single Time Series vs Multi-Stock**
   - Old code treated all stocks as one long series
   - **Required:** Each stock as separate univariate series
   - **Impact:** Improper handling of Vietnamese market structure

5. **❌ Missing Statistical Validation**
   - No Diebold-Mariano statistical testing
   - Limited performance metrics
   - **Impact:** Cannot prove statistical significance of results

## What's NOT Archived (Still Active)

### **✅ Core Project Structure:**
- `data/` - Raw OHLCV data (30 stocks, 100,365 observations)
- `_bmad-output/planning-artifacts/` - Complete architecture documents
- `docs/paper/` - Research papers (useful references)
- `.claude/` - Claude configuration and skills
- `_bmad/` - BMAD framework files

### **✅ Active Documentation:**
- `README.md` - Project overview (may need update)
- `claude.md` - **Comprehensive coding guidelines for TimesFM VN30**
- `_bmad-output/planning-artifacts/architecture/` - Complete architecture

### **✅ Architecture & Planning:**
- `_bmad-output/planning-artifacts/research/` - Technical research findings
- `_bmad-output/planning-artifacts/architecture.md` - Main architecture document
- `_bmad-output/planning-artifacts/architecture/project-structure.md` - Project structure
- `_bmad-output/planning-artifacts/architecture/timesfm-vn30-*.md` - Implementation plans

## New Implementation Approach

Based on the comprehensive architecture work, the new implementation will follow:

1. **Phase 1:** Foundation Setup (TimesFM 2.5 loading, GPU validation)
2. **Phase 2:** Data Engineering (Financial preprocessing, Vietnamese features)
3. **Phase 3:** Model Implementation (LoRA adapters, SGD optimizer)
4. **Phase 4:** Validation & Testing (Statistical testing, performance metrics)
5. **Phase 5:** Production Deployment (Model export, inference pipeline)

## Reference Value

These archived files are kept for:
1. **Learning** - Understanding what didn't work
2. **Reference** - Possible useful code snippets
3. **Comparison** - Seeing how the new approach differs
4. **Documentation** - Complete project history

## Don't Use These Files For

❌ New TimesFM VN30 implementation  
❌ Production model deployment  
❌ Further development without major fixes  

**Use these files only for reference and learning purposes.**

---

**Archive Status:** Complete ✅  
**Next Step:** Begin Phase 1 Implementation with clean architecture  
**Architecture Quality:** 9.5/10 (based on new design)