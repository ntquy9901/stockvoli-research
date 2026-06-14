# TimesFM VN30 Implementation Architecture Plan
**Project:** Vietnamese Stock Volatility Forecasting with TimesFM
**Date:** 2026-06-09
**Architect:** Winston (System Architect)
**Status:** Design Phase

---

## Executive Summary

This plan implements **proper TimesFM foundation model fine-tuning** for Vietnamese VN30 stocks using the proven methodology from `pfnet-research/timesfm_fin`. The architecture replaces custom transformer approaches with actual TimesFM LoRA fine-tuning, targeting 25-35% prediction accuracy improvement and Sharpe ratios of 0.8-1.5.

### Key Architecture Decisions

| Decision | Rationale | Impact |
|----------|-----------|---------|
| **Use actual TimesFM 2.5** | Access Google's pre-trained time series patterns | Leverages 100B+ data points foundation |
| **LoRA adapters (r=4)** | Parameter-efficient fine-tuning | Reduces VRAM by 90%, prevents catastrophic forgetting |
| **Log transformation** | Financial standard for handling extreme events | Prevents NaN loss during market crashes |
| **SGD optimizer** | Proven superior for financial time series | Better convergence on noisy financial data |
| **Multi-stock data loaders** | Vietnamese market portfolio structure | Proper handling of 30 stocks with different characteristics |

---

## Current System Analysis

### Assets Present ✅
- **30 Vietnamese stocks** with complete OHLCV data (100,365 observations)
- **Date range:** 2006-2026 (20 years of market data)
- **Data location:** `data/raw/prices/` (30 CSV files)
- **Failed attempts:** 4 previous fine-tuning approaches identified as incorrect

### Critical Issues Identified ❌
1. **All current implementations use custom transformers** - NOT TimesFM fine-tuning
2. **Missing financial data preprocessing** - no log transformation, improper normalization
3. **Wrong optimizer choice** - AdamW instead of SGD for financial data
4. **Single time series approach** - doesn't handle multi-stock portfolio structure
5. **No proper validation methodology** - missing Diebold-Mariano statistical tests

### Architecture Gap Analysis
```
Current State → Target State
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Custom Transformer → Actual TimesFM 2.5 Foundation Model
AdamW Optimizer → SGD with Momentum (lr=1e-4)
Raw OHLCV Data → Log-transformed Financial Features  
Single Series → Multi-Stock Data Loaders
Basic MAE/RMSE → Statistical Validation (DM Tests)
Zero-shot → Fine-tuned (30% loss reduction expected)
```

---

## Technical Architecture

### System Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                     │
│  • 30 VN30 Stocks (OHLCV) → Financial Preprocessing Pipeline      │
│  • Log Transformation → Realized Volatility → Normalization        │
└────────────────────────────────────────────────────────────────────┘
                                ↓
┌────────────────────────────────────────────────────────────────────┐
│                  MODEL ARCHITECTURE LAYER                          │
│  • TimesFM 2.5 Foundation (200M parameters)                        │
│  • LoRA Adapters (r=4, α=8) - Only trainable parameters          │
│  • Multi-stock attention mechanism                                │
└────────────────────────────────────────────────────────────────────┘
                                ↓
┌────────────────────────────────────────────────────────────────────┐
│                    TRAINING LAYER                                  │
│  • SGD Optimizer (lr=1e-4, momentum=0.9)                         │
│  • Gradient Clipping (max_norm=1.0)                               │
│  • Cosine Annealing LR Schedule                                  │
│  • Multi-stock DataLoader (channel-independent)                   │
└────────────────────────────────────────────────────────────────────┘
                                ↓
┌────────────────────────────────────────────────────────────────────┐
│                 VALIDATION LAYER                                   │
│  • Diebold-Mariano Statistical Tests                              │
│  • Sharpe Ratio Calculation                                       │
│  • Backtesting with Mock Trading                                  │
│  • Per-stock and Portfolio-level Metrics                          │
└────────────────────────────────────────────────────────────────────┘
```

### Component Architecture

#### 1. Data Processing Pipeline
```python
class VN30FinancialDataProcessor:
    """
    Vietnamese market-specific data processing following pfnet-research methodology
    
    Key transformations:
    - Log transformation for extreme events
    - Multi-horizon realized volatility  
    - Vietnamese market features (TET holiday, trading patterns)
    """
    
    def process_stock_data(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        # Step 1: Log transformation (CRITICAL for financial data)
        df['log_close'] = np.log(df['close'])
        df['log_returns'] = df['log_close'].diff()
        
        # Step 2: Realized volatility at multiple horizons
        for window in [5, 10, 20, 30]:
            df[f'RV_{window}'] = df['log_returns'].rolling(window).std()
        
        # Step 3: Vietnamese market features
        df['is_tet_period'] = self._detect_tet_holiday(df.index)
        df['day_of_week'] = df.index.dayofweek
        
        # Step 4: Financial-specific clipping
        df['RV_20'] = np.clip(df['RV_20'], -5, 5)  # Prevent extreme values
        
        return df
```

#### 2. Multi-Stock Dataset Architecture
```python
class VN30MultiStockDataset(Dataset):
    """
    Multi-stock dataset treating each stock as separate univariate series
    
    Architecture follows Issue #230 resolution for multiple time series:
    - Each stock (VCB, VIC, VNM, etc.) is independent series
    - Random window sampling across all stocks
    - Proper time-based train/test split (no leakage)
    """
    
    def __init__(self, stock_data_dict: Dict[str, pd.DataFrame], 
                 context_len: int = 128, horizon_len: int = 13):
        self.series_list = []
        
        # Process each stock separately
        for stock_name, df in stock_data_dict.items():
            log_returns = np.log(df['close']).diff()
            realized_vol = log_returns.rolling(20).std()
            
            # Each stock becomes separate series
            if len(realized_vol) >= context_len + horizon_len:
                self.series_list.append({
                    'name': stock_name,
                    'data': realized_vol.dropna().values
                })
        
        # Create random window samples
        self.samples = self._create_random_windows(
            self.series_list, context_len, horizon_len, num_samples=5000
        )
```

#### 3. Model Fine-tuning Architecture
```python
class TimesFMVN30Finetuner:
    """
    Actual TimesFM foundation model fine-tuning with LoRA adapters
    
    Architecture:
    - Base: TimesFM 2.5 (200M parameters, frozen)
    - Adapters: LoRA layers (r=4, trainable only)
    - Optimizer: SGD with momentum=0.9
    - Training: Financial-specific methodology
    """
    
    def __init__(self):
        # Load actual TimesFM 2.5 foundation model
        self.base_model = TimesFm2_5ModelForPrediction.from_pretrained(
            "google/timesfm-2.5-200m-transformers",
            torch_dtype=torch.bfloat16
        )
        
        # Configure LoRA adapters (parameter-efficient fine-tuning)
        lora_config = LoraConfig(
            r=4,                    # Low rank for efficiency
            lora_alpha=8,           # Scaling factor
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none"
        )
        self.model = get_peft_model(self.base_model, lora_config)
        
        # Financial-specific optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=1e-4,        # Conservative learning rate
            momentum=0.9,   # High momentum for stability
            nesterov=True   # Nesterov momentum
        )
```

### Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| **Base Model** | TimesFM 2.5 (200M) | Google's foundation model, proven on 100B+ time series |
| **Fine-tuning** | PEFT LoRA (r=4) | Parameter-efficient, prevents catastrophic forgetting |
| **Optimizer** | SGD (momentum=0.9) | Financial standard, superior to AdamW for time series |
| **Data Processing** | Pandas + NumPy | Financial data manipulation |
| **Validation** | scipy.stats | Diebold-Mariano statistical testing |
| **Hardware** | GPU (CUDA) | TimesFM requires GPU for training |

---

## Implementation Roadmap

### Phase 1: Foundation Setup (Week 1)
**Goal:** Establish proper technical foundation

**Tasks:**
1. **Repository Setup**
   - Clone `pfnet-research/timesfm_fin` for reference implementation
   - Study their financial configuration and data pipeline
   - Install required dependencies: `transformers`, `peft`, `accelerate`

2. **Data Validation**
   - Verify all 30 stocks have complete OHLCV data
   - Check date ranges and data quality
   - Validate minimum length requirements (≥100 trading days per stock)

3. **Environment Setup**
   - GPU availability check (TimesFM requires CUDA)
   - HuggingFace authentication setup
   - Memory requirements validation (16GB+ VRAM recommended)

**Deliverables:**
- ✅ All 30 stocks validated and ready
- ✅ TimesFM base model loads successfully
- ✅ GPU environment functional

**Success Criteria:**
- TimesFM 2.5 model loads and runs inference on sample data
- All 30 stocks pass quality checks
- Memory usage within limits

---

### Phase 2: Data Engineering (Week 2)
**Goal:** Build financial-specific data pipeline

**Tasks:**
1. **Financial Data Preprocessing**
   - Implement log transformation for all stocks
   - Calculate multi-horizon realized volatility (5, 10, 20, 30 days)
   - Add Vietnamese market features (TET holiday, day-of-week patterns)
   - Implement financial-specific clipping (-5, 5 range)

2. **Multi-Stock Data Loader**
   - Create `VN30MultiStockDataset` class
   - Implement random window sampling across stocks
   - Time-based train/test split (80/20, no leakage)
   - Data validation and quality checks

3. **Normalization Strategy**
   - Per-stock normalization implementation
   - Handle different price ranges (VCB ~90K VND vs HPG ~20K VND)
   - Consistent scaling across all stocks

**Deliverables:**
- ✅ `vn30_financial_data_loader.py` - Multi-stock dataset
- ✅ Processed dataset with 5,000+ training samples
- ✅ Normalization strategy for Vietnamese stocks

**Success Criteria:**
- 5,000+ training samples generated
- No NaN values in processed data
- Proper train/test split with no temporal leakage

---

### Phase 3: Model Implementation (Week 3-4)
**Goal:** Implement actual TimesFM fine-tuning

**Tasks:**
1. **Base Model Integration**
   - Load TimesFM 2.5 from HuggingFace
   - Verify pre-trained weights functionality
   - Test zero-shot performance baseline

2. **LoRA Adapter Configuration**
   - Implement LoRA (r=4, α=8)
   - Verify trainable parameters count (~1M vs 200M total)
   - Test adapter forward pass

3. **Financial Training Loop**
   - Implement SGD optimizer with momentum=0.9
   - Add gradient clipping (max_norm=1.0)
   - Cosine annealing learning rate schedule
   - Multi-stock batch processing

4. **Training Pipeline**
   - Single-stock fine-tuning test (VCB)
   - Expand to VN30 portfolio
   - Checkpoint management and model versioning

**Deliverables:**
- ✅ `timesfm_vn30_finetuner.py` - Complete fine-tuning pipeline
- ✅ LoRA-adapted TimesFM model for VN30
- ✅ Training loop with financial methodology

**Success Criteria:**
- Model trains without NaN losses
- Loss decreases consistently over epochs
- Checkpoints save and load correctly

---

### Phase 4: Validation & Testing (Week 5)
**Goal:** Comprehensive statistical validation

**Tasks:**
1. **Statistical Testing**
   - Implement Diebold-Mariano test for forecast accuracy
   - Compare fine-tuned vs baseline TimesFM
   - Calculate statistical significance (p < 0.05)

2. **Performance Metrics**
   - Core metrics: MAE, RMSE, R², Correlation
   - Volatility metrics: Direction accuracy, regime classification
   - Financial metrics: Sharpe ratio, maximum drawdown

3. **Backtesting Framework**
   - Mock trading strategy implementation
   - Portfolio-level evaluation
   - Per-stock performance analysis

4. **Production Readiness Checks**
   - Model quality validation (R² > 0.5 target)
   - Statistical significance confirmation
   - Deployment readiness assessment

**Deliverables:**
- ✅ `vn30_validation.py` - Statistical testing framework
- ✅ Performance report with all metrics
- ✅ Backtesting results for VN30 portfolio

**Success Criteria:**
- R² > 0.5 on test set
- Diebold-Mariano p-value < 0.05 (statistically significant)
- Sharpe ratio improvement vs baseline

---

### Phase 5: Production Deployment (Week 6)
**Goal:** Deploy model for live forecasting

**Tasks:**
1. **Model Export**
   - Save fine-tuned LoRA adapters
   - Create model metadata and versioning
   - Document configuration and hyperparameters

2. **Inference Pipeline**
   - Real-time prediction API
   - Batch processing for VN30 stocks
   - Performance monitoring setup

3. **Documentation**
   - Technical architecture documentation
   - Usage guide and API reference
   - Performance benchmarking results

**Deliverables:**
- ✅ Production-ready fine-tuned model
- ✅ Inference pipeline for daily forecasts
- ✅ Complete documentation package

**Success Criteria:**
- Model loads and predicts in production environment
- Inference time < 1 second per stock
- Documentation complete and accurate

---

## Testing Strategy

### Unit Testing
```python
# Test individual components
def test_financial_data_preprocessing():
    """Verify log transformation and volatility calculation"""
    sample_data = generate_sample_ohlcv()
    processed = VN30FinancialDataProcessor().process(sample_data)
    
    assert 'log_close' in processed.columns
    assert 'RV_20' in processed.columns
    assert not processed['RV_20'].isna().any()

def test_multi_stock_dataset():
    """Verify multi-stock data loader"""
    dataset = VN30MultiStockDataset(stock_data_dict, context_len=128, horizon_len=13)
    
    assert len(dataset) > 1000
    assert all('context' in sample and 'target' in sample for sample in dataset)
```

### Integration Testing
```python
# Test end-to pipeline
def test_timesfm_finetuning_pipeline():
    """Verify complete fine-tuning workflow"""
    finetuner = TimesFMVN30Finetuner()
    train_loader = create_vn30_dataloader()
    
    # Single epoch test
    metrics = finetuner.train_one_epoch(train_loader)
    
    assert metrics['loss'] < float('inf')
    assert metrics['loss'] > 0
```

### Statistical Validation Testing
```python
# Test statistical significance
def test_diebold_mariano_significance():
    """Verify statistical validation framework"""
    actual, model_pred, baseline = generate_test_data()
    dm_result = diebold_mariano_test(actual, model_pred, baseline)
    
    assert 'p_value' in dm_result
    assert 'significant' in dm_result
    assert 0 <= dm_result['p_value'] <= 1
```

---

## Success Criteria & Validation

### Technical Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Model Architecture** | Actual TimesFM 2.5 + LoRA | Code review + model inspection |
| **Data Processing** | Log-transformed, multi-stock | Unit tests + data validation |
| **Training Stability** | No NaN losses, consistent convergence | Training logs + loss curves |
| **Statistical Significance** | p < 0.05 (Diebold-Mariano) | Statistical test results |

### Performance Success Metrics

| Metric | Baseline | Target | Measurement Method |
|--------|----------|--------|-------------------|
| **R² Score** | 0.0 (zero-shot) | > 0.5 | Test set evaluation |
| **Sharpe Ratio** | 0.42 (TimesFM baseline) | 0.8 - 1.5 | Backtesting framework |
| **Loss Reduction** | 0% | 25-35% | Training/validation loss comparison |
| **Statistical Significance** | N/A | p < 0.05 | Diebold-Mariano test |

### Business Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Prediction Accuracy** | > 70% direction accuracy | Test set evaluation |
| **Portfolio Performance** | Positive Sharpe ratio | Backtesting results |
| **Production Readiness** | Deploy-ready model | Deployment checklist |

---

## Risk Analysis & Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|--------|-------------------|
| **TimesFM loading failure** | Medium | High | Test HuggingFace download early, have fallback to 1.0 model |
| **GPU memory issues** | Medium | High | Implement gradient accumulation, reduce batch size |
| **NaN loss during training** | Low | High | Log transformation, gradient clipping, data validation |
| **Poor convergence** | Medium | Medium | Conservative learning rate, proven SGD configuration |

### Data Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|--------|-------------------|
| **Insufficient data quality** | Low | Medium | Data validation pipeline, outlier handling |
| **Missing stocks data** | Low | Medium | Flexible dataset handles missing stocks |
| **Date range inconsistencies** | Low | Low | Time-based splitting, per-stock processing |

### Business Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|--------|-------------------|
| **Performance doesn't meet targets** | Medium | High | Iterative testing, hyperparameter tuning |
| **Statistical insignificance** | Low | Medium | Multiple testing approaches, benchmark comparison |
| **Deployment issues** | Low | Medium | Comprehensive testing, gradual rollout |

---

## Architecture Trade-offs

### Decision 1: LoRA vs Full Fine-tuning
**Chosen:** LoRA adapters (r=4)

**Trade-offs:**
- ✅ **Pros:** 90% less VRAM, prevents catastrophic forgetting, faster training
- ❌ **Cons:** Slightly lower capacity vs full fine-tuning
- **Decision:** Financial data benefits from pre-trained patterns; LoRA preserves them

### Decision 2: SGD vs AdamW Optimizer
**Chosen:** SGD with momentum=0.9

**Trade-offs:**
- ✅ **Pros:** Superior for financial time series, proven by pfnet-research
- ❌ **Cons:** Slower convergence than adaptive methods
- **Decision:** Financial data characteristics favor SGD's stability

### Decision 3: Multi-Stock vs Single Stock Models
**Chosen:** Multi-stock dataset with independent series

**Trade-offs:**
- ✅ **Pros:** Portfolio context, efficient use of data, single model deployment
- ❌ **Cons:** More complex data pipeline
- **Decision:** Vietnamese market requires portfolio approach for practical use

---

## Implementation Dependencies

### External Dependencies
```bash
# Core TimesFM and fine-tuning
pip install transformers>=4.35.0
pip install peft>=0.5.0
pip install accelerate>=0.24.0

# Data processing
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install scikit-learn>=1.3.0

# Statistical testing
pip install scipy>=1.10.0

# Financial data
pip install yfinance>=0.2.0
```

### Internal Dependencies
```
vn30_financial_data_loader.py (Phase 2)
  ↓
timesfm_vn30_finetuner.py (Phase 3)
  ↓
vn30_validation.py (Phase 4)
  ↓
production_inference.py (Phase 5)
```

---

## Next Actions

### Immediate (This Week)
1. **Clone pfnet-research/timesfm_fin** repository
2. **Test TimesFM 2.5 loading** on your system
3. **Validate all 30 stocks** data quality
4. **Set up GPU environment** with required dependencies

### Architecture Review Questions
1. **GPU Resources:** Do you have access to GPU with 16GB+ VRAM?
2. **Timeline Constraints:** Is the 6-week timeline suitable or do you need faster results?
3. **Production Requirements:** Should we prioritize single-stock accuracy or portfolio performance?
4. **Backtesting Needs:** Do you need specific trading strategy validation?

### Stakeholder Alignment Needed
- **Technical Team:** GPU availability and deployment infrastructure
- **Business Team:** Performance targets and success criteria
- **Data Team:** Additional data sources or features needed

---

## Appendix: Architectural Principles Applied

1. **Boring Technology** - Use proven TimesFM foundation, avoid experimental architectures
2. **Developer Productivity** - LoRA adapters reduce training time and complexity
3. **Rule of Three** - Don't abstract until we have three working implementations
4. **Financial Standards** - Follow pfnet-research methodology proven on major markets

---

**Document Status:** Ready for Review
**Next Phase:** Implementation Readiness Check
**Owner:** Winston (System Architect)
**Review Date:** 2026-06-09