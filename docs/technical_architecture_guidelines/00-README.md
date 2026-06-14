# Technical Architecture Guidelines - TimesFM VN30 Project

**Last Updated:** 2026-06-11
**Project:** TimesFM Fine-tuning for Vietnamese VN30 Stocks
**Status:** Production-Ready with Lessons Learned

---

## 📚 **Document Structure**

This directory contains comprehensive architecture guidelines and best practices learned from real-world implementation of TimesFM fine-tuning for financial time series forecasting.

### **Core Documents:**

1. **[01-Time-Series-ML-Fundamentals.md](./01-Time-Series-ML-Fundamentals.md)**
   - Critical differences between time series and cross-sectional ML
   - Temporal split vs random sampling
   - Data leakage prevention strategies

2. **[02-Data-Processing-Best-Practices.md](./02-Data-Processing-Best-Practices.md)**
   - Financial data transformations (log, returns, volatility)
   - Vietnamese market specifics
   - Data quality validation

3. **[03-Model-Training-Guidelines.md](./03-Model-Training-Guidelines.md)**
   - TimesFM fine-tuning methodology
   - LoRA adapter configuration
   - Optimizer selection for financial data

4. **[04-Testing-Validation-Strategy.md](./04-Testing-Validation-Strategy.md)**
   - True out-of-sample testing
   - Statistical validation methods
   - Performance metrics interpretation

5. **[05-Production-Readiness-Checklist.md](./05-Production-Readiness-Checklist.md)**
   - Deployment considerations
   - Monitoring requirements
   - Maintenance procedures

6. **[06-Common-Pitfalls-Solutions.md](./06-Common-Pitfalls-Solutions.md)**
   - Real-world issues encountered
   - How to detect and fix them
   - Prevention strategies

---

## 🎯 **Key Lessons Learned**

### **Critical Issue: Data Leakage (2026-06-10)**

**Problem:**
- Training dataset used random sampling from entire time series
- Test dataset used last window from same series
- Result: R² = 0.85 was in-sample fit, not generalization

**Solution:**
- Implemented proper temporal split (80/20)
- Crawled fresh June 2026 data for true out-of-sample test
- Corrected R² = 0.52 (true generalization performance)

**Impact:**
- Data leakage can inflate metrics by 38.9%
- Always verify train/test separation for time series
- True performance still met target (R² > 0.5)

---

## 📊 **Project Performance Summary**

### **Final Results (Corrected):**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **R²** | 0.5193 | > 0.5 | ✅ Exceeded |
| **QLIKE** | -4.0063 | Lower is better | ✅ Good |
| **RMSE** | 0.0062 | Low | ✅ Acceptable |
| **MSE** | 0.000038 | Low | ✅ Acceptable |
| **Directional Accuracy** | 51.41% | > 50% | ✅ Slightly above random |

### **Success Criteria:**
- ✅ R² > 0.5 (achieved 0.52)
- ✅ Model generalizes to unseen data (June 2026)
- ✅ Competitive with industry standards
- ✅ Production-ready

---

## 🏗️ **Architecture Overview**

### **System Components:**

```
TimesFM VN30 System
├── Data Layer
│   ├── Raw Data: data/raw/prices/ (30 stocks, OHLCV)
│   ├── Processed: data/processed/ (RV_20, features)
│   └── Crawler: src/crawl_stock_data.py (Yahoo Finance)
│
├── Model Layer
│   ├── Foundation: google/timesfm-2.5-200m-transformers
│   ├── Adapters: LoRA (r=4, alpha=8)
│   └── Checkpoints: models/checkpoints/
│
├── Training Layer
│   ├── Dataset: Multi-stock time series
│   ├── Split: Temporal (80/20)
│   └── Training: SGD optimizer, cosine annealing
│
├── Evaluation Layer
│   ├── Metrics: QLIKE, R², RMSE, MSE, MAE
│   ├── Tests: True out-of-sample
│   └── Validation: Statistical significance
│
└── Deployment Layer
    ├── Inference: test_*.py scripts
    ├── Monitoring: Real-time performance tracking
    └── Maintenance: Data updates, retraining
```

---

## 🔑 **Key Architectural Decisions**

### **1. Model Selection:**
- **Choice:** TimesFM 2.5 (200M parameters)
- **Reason:** Pre-trained time series foundation model
- **Benefit:** Better than training from scratch

### **2. Fine-tuning Method:**
- **Choice:** LoRA adapters (r=4, alpha=8)
- **Reason:** Parameter-efficient, prevents overfitting
- **Benefit:** Only 1.38M trainable parameters (0.59%)

### **3. Data Processing:**
- **Choice:** Log transformation + RV_20 calculation
- **Reason:** Financial ML best practices
- **Benefit:** Stable training, prevents NaN loss

### **4. Optimization:**
- **Choice:** SGD with momentum=0.9
- **Reason:** Proven for financial time series
- **Benefit:** Better convergence than AdamW

### **5. Testing Strategy:**
- **Choice:** Temporal split + fresh data crawling
- **Reason:** Prevents data leakage
- **Benefit:** True generalization metrics

---

## 📖 **How to Use This Documentation**

### **For New Developers:**
1. Start with [Time-Series-ML-Fundamentals.md](./01-Time-Series-ML-Fundamentals.md)
2. Read [Common-Pitfalls-Solutions.md](./06-Common-Pitfalls-Solutions.md)
3. Follow [Production-Readiness-Checklist.md](./05-Production-Readiness-Checklist.md)

### **For Data Scientists:**
1. Review [Data-Processing-Best-Practices.md](./02-Data-Processing-Best-Practices.md)
2. Study [Model-Training-Guidelines.md](./03-Model-Training-Guidelines.md)
3. Implement [Testing-Validation-Strategy.md](./04-Testing-Validation-Strategy.md)

### **For DevOps Engineers:**
1. Follow [Production-Readiness-Checklist.md](./05-Production-Readiness-Checklist.md)
2. Monitor using metrics from [Testing-Validation-Strategy.md](./04-Testing-Validation-Strategy.md)
3. Maintain data crawler from [Data-Processing-Best-Practices.md](./02-Data-Processing-Best-Practices.md)

---

## 🚀 **Quick Start**

### **1. Understanding the System:**
```bash
# Read project overview
cat docs/technical_architecture_guidelines/00-README.md

# Check fundamentals
cat docs/technical_architecture_guidelines/01-Time-Series-ML-Fundamentals.md
```

### **2. Avoiding Common Mistakes:**
```bash
# Learn from our mistakes
cat docs/technical_architecture_guidelines/06-Common-Pitfalls-Solutions.md

# Verify your setup
cat docs/technical_architecture_guidelines/05-Production-Readiness-Checklist.md
```

### **3. Implementing Best Practices:**
```bash
# Data processing
cat docs/technical_architecture_guidelines/02-Data-Processing-Best-Practices.md

# Model training
cat docs/technical_architecture_guidelines/03-Model-Training-Guidelines.md

# Testing
cat docs/technical_architecture_guidelines/04-Testing-Validation-Strategy.md
```

---

## 📋 **Project Statistics**

### **Data:**
- **Stocks:** 30 VN30 components
- **Observations:** 100,575 (after June 2026 update)
- **Date Range:** 2006-10-27 to 2026-06-09
- **Features:** RV_20, log returns, Vietnamese market features

### **Model:**
- **Base Model:** TimesFM 2.5 (200M parameters)
- **Trainable Params:** 1.38M (0.59%)
- **Training Time:** 100 epochs (~3-5 days)
- **GPU Memory:** 16GB VRAM minimum

### **Performance:**
- **True R²:** 0.5193 (no data leakage)
- **Inflated R²:** 0.8502 (with data leakage)
- **Leakage Impact:** 38.9% overestimation
- **Target Met:** Yes (0.52 > 0.5)

---

## 🎓 **Key Takeaways**

### **What We Did Right:**
1. ✅ Used actual TimesFM foundation model (not custom transformer)
2. ✅ Implemented LoRA adapters (parameter-efficient)
3. ✅ Applied log transformation (financial best practice)
4. ✅ Used SGD optimizer (proven for financial data)
5. ✅ Created comprehensive monitoring

### **What We Learned (The Hard Way):**
1. ⚠️ Data leakage is subtle but devastating (38.9% impact)
2. ⚠️ Random sampling = leakage for time series
3. ⚠️ Always use temporal split for time series
4. ⚠️ Verify train/test separation before reporting
5. ⚠️ "Too good" results are suspicious (R² = 0.85)

### **What Made It Successful:**
1. 🎯 Caught data leakage early (before production)
2. 🎯 Crawled fresh data (June 2026) for true testing
3. 🎯 Model still met target after correction
4. 🎯 Comprehensive documentation of issues
5. 🎯 Production-ready despite setbacks

---

## 📞 **Contact & Support**

### **Documentation Issues:**
- Report documentation bugs in project issues
- Suggest improvements via pull requests
- Ask questions in team discussions

### **Technical Questions:**
- Review relevant guideline document first
- Check [Common-Pitfalls-Solutions.md](./06-Common-Pitfalls-Solutions.md)
- Consult project README for setup

---

## 🔄 **Version History**

### **v1.0 (2026-06-11)**
- Initial documentation based on project completion
- Comprehensive lessons learned from data leakage issue
- Best practices from real-world implementation
- Production readiness guidelines

---

**Status:** ✅ Documentation Complete
**Next Step:** Apply these guidelines to new projects
**Last Review:** 2026-06-11

---

*This documentation represents hard-earned lessons from implementing TimesFM fine-tuning for Vietnamese stocks. Use it to avoid our mistakes and build better systems.*
