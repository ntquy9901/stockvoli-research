# TimesFM VN30 Architecture Diagrams
**Project:** Vietnamese Stock Volatility Forecasting System
**Date:** 2026-06-09
**Architect:** Winston (System Architect)

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TIMESFM VN30 SYSTEM ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA INGESTION LAYER                                │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   VCB Data   │  │   VIC Data   │  │   VNM Data   │  │   HPG Data   │   │
│  │  (4,222 days)│  │  (4,659 days)│  │  (4,880 days)│  │  (4,618 days)│   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│         │                  │                  │                  │             │
│         └──────────────────┬──────────────────┴──────────────────┘          │
│                            ↓                                               │
│                   ┌───────────────────┐                                    │
│                   │ 30 Stocks Total   │                                    │
///                  │ 100,365 obs total  │                                    │
│                   └───────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FINANCIAL PROCESSING LAYER                              │
│                                                                              │
│  Step 1: Log Transformation                     Step 2: Realized Volatility │
│  ┌───────────────────────────────────┐        ┌─────────────────────────┐ │
│  │ log_close = log(close)             │        │ RV_5   = 5-day std      │ │
│  │ log_returns = log_close.diff()     │  →     │ RV_10  = 10-day std     │ │
│  │                                    │        │ RV_20  = 20-day std     │ │
│  │ Prevents NaN during extreme events │        │ RV_30  = 30-day std     │ │
│  └───────────────────────────────────┘        └─────────────────────────┘ │
│                            ↓                                                 │
│  Step 3: Vietnamese Market Features            Step 4: Financial Clipping  │
│  ┌───────────────────────────────────┐        ┌─────────────────────────┐ │
│  │ • TET Holiday detection            │        │ Clip RV_20: [-5, 5]       │ │
│  │ • Day-of-week patterns             │  →     │ Prevents extreme values   │ │
│  │ • Month-end effects                │        │ Maintains stability      │ │
│  │ • Earnings season flags           │        └─────────────────────────┘ │
│  └───────────────────────────────────┘                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MULTI-STOCK DATASET LAYER                             │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │              VN30MultiStockDataset (Channel-Independent)              │  │
│  │                                                                       │  │
│  │  Each Stock → Separate Time Series                                   │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                │  │
│  │  │   VCB   │  │   VIC   │  │   VNM   │  │   HPG   │  ... (26 more)  │  │
│  │  │ Series  │  │ Series  │  │ Series  │  │ Series  │                 │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘                 │  │
│  │       │            │            │            │                         │  │
│  │       └────────────┴────────────┴────────────┘                         │  │
│  │                    ↓                                                    │  │
│  │         ┌──────────────────┐                                           │  │
│  │         │ Random Window    │                                           │  │
│  │         │ Sampling         │                                           │  │
│  │         │ (5,000+ samples) │                                           │  │
│  │         └──────────────────┘                                           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Training: 69,673 samples (80%) | Testing: 12,024 samples (20%)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TIMESFM MODEL ARCHITECTURE                           │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   TIMESFM 2.5 FOUNDATION MODEL                        │   │
│  │                    (200M Parameters - Frozen)                         │   │
│  │                                                                       │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │     TimesFM Transformer Blocks (20 layers, 8 heads)           │   │   │
│  │  │                                                               │   │   │
│  │  │  Input: [batch, context_len=128, features=7]                 │   │   │
│  │  │  Positional Encoding → Multi-Head Attention → FFN          │   │   │
│  │  │  Context Understanding → Pattern Recognition → Prediction    │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                ↓                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │              LORA ADAPTERS (Trainable)                        │   │   │
│  │  │              (r=4, α=8, ~1M params)                           │   │   │
│  │  │                                                               │   │   │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐             │   │   │
│  │  │  │ LoRA-rank-4│  │ LoRA-rank-4│  │ LoRA-rank-4│  ...        │   │   │
│  │  │  │   Layer 1  │  │   Layer 2  │  │   Layer 3  │             │   │   │
│  │  │  └────────────┘  └────────────┘  └────────────┘             │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                ↓                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │              Volatility Prediction Head                        │   │   │
│  │  │              Output: Next day's RV_20                         │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Model: google/timesfm-2.5-200m-transformers + LoRA adapters                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING INFRASTRUCTURE                             │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    OPTIMIZER & SCHEDULING                               │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │  SGD Optimizer (Financial-Standard)                            │  │  │
│  │  │  • Learning Rate: 1e-4 (Conservative)                          │  │  │
│  │  │  • Momentum: 0.9 (High stability)                              │  │  │
│  │  │  • Nesterov: True (Faster convergence)                         │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │                              ↓                                       │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Cosine Annealing LR Schedule                                  │  │  │
│  │  │  • Warmup: 5 epochs                                           │  │  │
│  │  │  • Decay: 95 epochs                                           │  │  │
│  │  │  • Min LR: 1e-6                                               │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      TRAINING CONFIGURATION                            │  │
│  │  • Batch Size: 32 (GPU-optimized)                                   │  │
│  │  • Gradient Clipping: max_norm=1.0                                   │  │
│  │  • Epochs: 100 (5 warmup + 95 decay)                                │  │
│  │  • Hardware: GPU (CUDA) with 16GB+ VRAM                             │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          VALIDATION & TESTING LAYER                            │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    STATISTICAL VALIDATION                             │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Diebold-Mariano Test                                          │  │  │
│  │  │  • Null: Model and benchmark equal accuracy                    │  │  │
│  │  │  • Alternative: Model superior to benchmark                    │  │  │
│  │  │  • Significance: p < 0.05                                      │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │                              ↓                                       │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Performance Metrics                                           │  │  │
│  │  │  • Core: R², MAE, RMSE, Correlation                            │  │  │
│  │  │  • Volatility: Direction accuracy, regime classification        │  │  │
│  │  │  • Financial: Sharpe ratio, max drawdown                        │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      BACKTESTING FRAMEWORK                             │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Mock Trading Strategy                                         │  │  │
│  │  │  • Market neutral: Long low vol, Short high vol                │  │  │
│  │  │  • Portfolio: VN30 equal-weighted                               │  │  │
│  │  │  • Transaction costs: 0.20% per trade                          │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │                              ↓                                       │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Portfolio Metrics                                            │  │  │
│  │  │  • Total return, Sharpe ratio, Sortino ratio                   │  │  │
│  │  │  • Maximum drawdown, Win rate                                   │  │  │
│  │  │  • Per-stock performance breakdown                             │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PRODUCTION DEPLOYMENT LAYER                           │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    MODEL SERVING INFRASTRUCTURE                       │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │  FastAPI Service                                                 │  │  │
│  │  │  • POST /predict-volatility                                     │  │  │
│  │  │  • Batch processing: VN30 stocks                                │  │  │
│  │  │  • Response time: < 1s per stock                                │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │                              ↓                                       │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Model Management                                               │  │  │
│  │  │  • Version: vn30_timesfm_v1.0                                   │  │  │
│  │  │  • Checkpoints: Best + Last                                    │  │  │
│  │  │  • Metadata: Training config, performance metrics               │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    MONITORING & ALERTING                               │  │
│  │  • Prediction drift detection                                        │  │
│  │  • Model performance monitoring                                       │  │
│  │  • Data quality alerts                                               │  │
│  │  • Automatic retraining triggers                                     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA FLOW DIAGRAM                                   │
└─────────────────────────────────────────────────────────────────────────────┘

[Raw OHLCV Files]
        ↓
[Load & Validate] → Data Quality Checks → Missing Data Handling
        ↓
[Financial Transformations]
    ├── Log Transformation → Prevents NaN during extreme events
    ├── Realized Volatility → Multi-horizon calculations (5,10,20,30)
    ├── Vietnamese Features → TET holidays, trading patterns
    └── Financial Clipping → Range [-5, 5] for stability
        ↓
[Multi-Stock Dataset Creation]
    ├── Per-stock Series → Each stock as independent time series
    ├── Random Window Sampling → 5,000+ samples across all stocks
    ├── Train/Test Split → Time-based 80/20 (no leakage)
    └── Normalization → Per-stock scaling for different price ranges
        ↓
[Model Training Pipeline]
    ├── TimesFM 2.5 Base → Pre-trained patterns (frozen)
    ├── LoRA Adapters → Vietnamese market adaptation (trainable)
    ├── SGD Optimizer → Financial-standard training
    └── Gradient Clipping → Training stability
        ↓
[Model Evaluation]
    ├── Statistical Tests → Diebold-Mariano significance
    ├── Performance Metrics → R², MAE, RMSE, Correlation
    ├── Backtesting → Mock trading with VN30 portfolio
    └── Production Checks → Deployment readiness validation
        ↓
[Production Deployment]
    ├── Model Export → LoRA adapters + metadata
    ├── Inference API → Real-time predictions
    ├── Monitoring → Performance drift detection
    └── Retraining → Continuous adaptation pipeline
```

---

## Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      COMPONENT INTERACTION ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────┐
│  Data Processing Module  │
│  (vn30_financial_data_  │
│   processor.py)         │
└──────────┬──────────────┘
           │
           ├──→ [Load CSV Files]
           ├──→ [Validate Data Quality]
           ├──→ [Log Transform]
           ├──→ [Calculate RV]
           ├──→ [Add VN Features]
           └── [Output: Processed Data]
                  ↓
┌──────────────────────────┐
│  Dataset Creation Module  │
│  (vn30_multi_stock_      │
│   dataset.py)            │
└──────────┬──────────────┘
           │
           ├──→ [Create Stock Series]
           ├──→ [Random Window Sample]
           ├──→ [Train/Test Split]
           └── [Output: DataLoader]
                  ↓
┌──────────────────────────┐
│  Model Training Module   │
│  (timesfm_vn30_          │
│   finetuner.py)          │
└──────────┬──────────────┘
           │
           ├──→ [Load TimesFM 2.5]
           ├──→ [Add LoRA Adapters]
           ├──→ [Configure SGD]
           ├──→ [Training Loop]
           └── [Output: Fine-tuned Model]
                  ↓
┌──────────────────────────┐
│  Validation Module       │
│  (vn30_validation.py)    │
└──────────┬──────────────┘
           │
           ├──→ [DM Statistical Test]
           ├──→ [Performance Metrics]
           ├──→ [Backtesting Framework]
           └── [Output: Validation Report]
                  ↓
┌──────────────────────────┐
│  Deployment Module      │
│  (production_inference. │
│   py)                   │
└──────────┬──────────────┘
           │
           ├──→ [Load Fine-tuned Model]
           ├──→ [Inference API]
           ├──→ [Monitoring Setup]
           └── [Output: Production Service]
```

---

## Technology Stack Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TECHNOLOGY STACK LAYERS                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  APPLICATION LAYER                                                           │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐   │
│  │   Training Pipeline  │  │  Validation Testing  │  │ Production API   │   │
│  └──────────────────────┘  └──────────────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  MODEL LAYER                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    TimesFM 2.5 + LoRA Adapters                        │  │
│  │                   (google/timesfm-2.5-200m)                           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  ML FRAMEWORK LAYER                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Transformers │  │     PEFT     │  │  Accelerate  │  │ PyTorch 2.0+ │   │
│  │   (Hugging  │  │   (LoRA)     │  │   (GPU)      │  │   (Backend)  │   │
│  │    Face)    │  │              │  │              │  │              │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  DATA PROCESSING LAYER                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │    Pandas    │  │    NumPy     │  │  Scikit-learn│  │    SciPy     │   │
│  │  (Dataframe) │  │  (Arrays)    │  │  (Metrics)   │  │  (Stats)     │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  HARDWARE LAYER                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    GPU (CUDA) + 16GB+ VRAM                            │  │
│  │                    TimesFM requires GPU for training                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Risk Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RISK MITIGATION ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────────────────┘

[Technical Risks]
    ├── GPU Memory Issues → Mitigation: Gradient accumulation, batch size reduction
    ├── Model Loading Failure → Mitigation: Early testing, fallback to TimesFM 1.0
    ├── NaN Training Loss → Mitigation: Log transform, gradient clipping, data validation
    └── Poor Convergence → Mitigation: Conservative LR, proven SGD config

[Data Risks]
    ├── Insufficient Quality → Mitigation: Data validation pipeline, outlier handling
    ├── Missing Stock Data → Mitigation: Flexible dataset, graceful degradation
    └── Date Inconsistencies → Mitigation: Time-based splitting, per-stock processing

[Business Risks]
    ├── Performance Targets Missed → Mitigation: Iterative testing, hyperparameter tuning
    ├── Statistical Insignificance → Mitigation: Multiple testing approaches
    └── Deployment Issues → Mitigation: Comprehensive testing, gradual rollout
```

---

## Success Criteria Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SUCCESS CRITERIA CHECKPOINTS                            │
└─────────────────────────────────────────────────────────────────────────────┘

[Phase 1: Foundation] ✓
    ├── TimesFM 2.5 loads successfully
    ├── All 30 stocks validated
    └── GPU environment functional

[Phase 2: Data Engineering] ✓
    ├── 5,000+ training samples generated
    ├── No NaN values in processed data
    └── Proper train/test split (no leakage)

[Phase 3: Model Implementation] ✓
    ├── Model trains without NaN losses
    ├── Loss decreases consistently
    └── Checkpoints save/load correctly

[Phase 4: Validation] ✓
    ├── R² > 0.5 on test set
    ├── DM p-value < 0.05 (significant)
    └── Sharpe ratio improvement vs baseline

[Phase 5: Production] ✓
    ├── Model loads in production
    ├── Inference time < 1s per stock
    └── Documentation complete
```

---

**Document Status:** Architecture Diagrams Complete
**Next Phase:** Implementation Readiness Check
**Owner:** Winston (System Architect)
**Review Date:** 2026-06-09