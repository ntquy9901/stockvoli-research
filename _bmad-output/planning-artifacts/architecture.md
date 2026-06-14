---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8]
workflowType: 'architecture'
lastStep: 8
status: 'complete'
completedAt: '2026-06-09'
inputDocuments:
  - file: "_bmad-output/planning-artifacts/architecture/timesfm-vn30-implementation-plan.md"
    type: "implementation-plan"
    loaded: "2026-06-09"
  - file: "_bmad-output/planning-artifacts/research/technical-timesfm-financial-finetune-research-2026-06-07.md"
    type: "research"
    loaded: "2026-06-07"
  - file: "_bmad-output/planning-artifacts/architecture/timesfm-vn30-architecture-diagrams.md"
    type: "architecture-diagrams"
    loaded: "2026-06-09"
  - file: "_bmad-output/planning-artifacts/architecture/timesfm-vn30-testing-strategy.md"
    type: "testing-strategy"
    loaded: "2026-06-09"
workflowType: 'architecture'
project_name: 'stockvoli-research'
user_name: 'QUY'
date: '2026-06-09'
---

# Architecture Decision Document

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

## Document Setup Complete

**Workspace:** Architecture workspace initialized for TimesFM VN30 fine-tuning system

**Input Documents Loaded:**
- ✅ Implementation Plan: Complete technical architecture with 6-phase roadmap
- ✅ Research: TimesFM financial fine-tuning research (pfnet-research methodology)
- ✅ Architecture Diagrams: System architecture, data flows, component interactions
- ✅ Testing Strategy: Comprehensive testing framework from unit to production

**Foundation Established:**
Using existing implementation plan as requirements foundation for proper TimesFM foundation model fine-tuning of Vietnamese VN30 stocks.

**Ready to begin architectural decision making._

---

## Project Context Analysis

### Requirements Overview

**Functional Requirements:**

1. **Data Processing Pipeline** - Financial transformations for 30 VN30 stocks (100,365 observations from 2006-2026)
   - Log transformation for extreme event handling
   - Multi-horizon realized volatility calculation (5, 10, 20, 30 days)
   - Vietnamese market-specific features (TET holidays, trading patterns)
   - Financial clipping (-5, 5 range) for stability

2. **TimesFM Integration** - Load and configure actual TimesFM 2.5 foundation model
   - HuggingFace model loading: `google/timesfm-2.5-200m-transformers`
   - LoRA adapter configuration (r=4, α=8) for parameter-efficient fine-tuning
   - Multi-stock attention mechanism handling

3. **Multi-Stock Training Architecture** - Treat each stock as separate univariate time series
   - Channel-independent approach for 30 VN30 stocks
   - Random window sampling across all stocks (5,000+ samples)
   - Time-based train/test split (80/20) with no temporal leakage

4. **Financial Methodology Implementation** - Follow pfnet-research proven approach
   - SGD optimizer with momentum=0.9 (NOT AdamW)
   - Gradient clipping (max_norm=1.0) for training stability
   - Cosine annealing learning rate schedule (5 warmup + 95 decay epochs)

5. **Statistical Validation Framework** - Rigorous model evaluation
   - Diebold-Mariano test for statistical significance (p < 0.05 target)
   - Sharpe ratio calculation for financial performance assessment
   - Backtesting framework with mock trading strategies

6. **Production Deployment Pipeline** - Real-time inference capabilities
   - Batch processing for all 30 VN30 stocks
   - Model serving infrastructure with FastAPI
   - Performance monitoring and drift detection

**Non-Functional Requirements:**

**Performance Requirements:**
- **Model Accuracy:** R² > 0.5 on test set, 25-35% loss reduction vs zero-shot
- **Financial Performance:** Sharpe ratio 0.8-1.5 vs baseline 0.42
- **Inference Speed:** < 1 second per stock for production predictions
- **Prediction Quality:** > 70% direction accuracy, statistical significance (p < 0.05)

**Reliability & Stability:**
- **Training Stability:** No NaN losses, consistent convergence over epochs
- **Extreme Event Handling:** Robust to market crashes through log transformation
- **Catastrophic Forgetting Prevention:** LoRA adapters preserve pre-trained patterns
- **Data Quality:** Handle missing stocks, inconsistent date ranges, varying price ranges

**Technical Constraints:**
- **Hardware Requirements:** GPU with 16GB+ VRAM for TimesFM training
- **Model Dependencies:** TimesFM 2.5, PEFT LoRA framework, HuggingFace transformers
- **Methodology Constraints:** Must follow pfnet-research financial fine-tuning approach
- **Optimization Requirements:** SGD optimizer (financial standard), specific hyperparameters

**Scale & Complexity:**

- **Primary Domain:** Machine Learning / Financial Time Series / Statistical Analysis
- **Complexity Level:** HIGH (foundation model fine-tuning, financial domain expertise, statistical validation)
- **Estimated Architectural Components:** 5 major layers, 12+ components
- **Implementation Timeline:** 6 weeks across 5 phases (Foundation → Data → Model → Validation → Production)

### Technical Constraints & Dependencies

**Critical Technical Dependencies:**
- **TimesFM 2.5 Model:** `google/timesfm-2.5-200m-transformers` (200M parameters, bfloat16)
- **Fine-tuning Framework:** PEFT LoRA (r=4, α=8), transformers ≥4.35.0, accelerate ≥0.24.0
- **Financial Libraries:** scipy.stats (statistical testing), pandas/numpy (financial data processing)
- **Reference Implementation:** pfnet-research/timesfm_fin methodology and configuration

**Hardware & Infrastructure Constraints:**
- **GPU Requirements:** CUDA-compatible GPU, 16GB+ VRAM recommended
- **Memory Requirements:** Sufficient RAM for 30 stocks dataset (100,365 observations)
- **Storage:** Checkpoint management for multiple model versions and training states
- **Network:** HuggingFace model download access, potential API dependencies

**Data Constraints:**
- **Market Coverage:** 30 VN30 stocks spanning 20 years (2006-2026)
- **Data Quality:** Must handle missing data points, varying stock histories, price range differences
- **Temporal Constraints:** Time-based splitting required (no future data leakage)
- **Financial Specifics:** Vietnamese market calendar (TET holidays), trading patterns

**Methodology Constraints:**
- **Optimizer Choice:** SGD with momentum=0.9 (NOT AdamW) - financial standard
- **Data Processing:** Log transformation required (prevents NaN during extreme events)
- **Multi-Stock Architecture:** Channel-independent approach (NOT concatenated single series)
- **Validation Rigor:** Statistical significance testing mandatory (Diebold-Mariano p < 0.05)

### Cross-Cutting Concerns Identified

**1. Financial Data Integrity**
- **Impact:** All architecture layers
- **Concern:** Extreme market events can cause training instability
- **Mitigation:** Log transformation, financial clipping (-5,5), gradient clipping
- **Components:** Data processing, training loop, validation metrics

**2. Statistical Validation Rigor**
- **Impact:** Model acceptance criteria, testing strategy
- **Concern:** Financial ML requires statistical significance, not just accuracy
- **Mitigation:** Diebold-Mariano testing framework, Sharpe ratio calculation
- **Components:** Validation layer, backtesting framework, production monitoring

**3. Vietnamese Market Specifics**
- **Impact:** Data processing, feature engineering, model expectations
- **Concern:** Local market patterns differ from global training data
- **Mitigation:** TET holiday features, day-of-week patterns, local price range normalization
- **Components:** Data processing layer, feature engineering, inference pipeline

**4. Catastrophic Forgetting Prevention**
- **Impact:** Model architecture, training strategy
- **Concern:** Fine-tuning may destroy pre-trained time series patterns
- **Mitigation:** LoRA adapters (r=4) freeze most parameters, conservative learning rate
- **Components:** Model architecture, training loop, hyperparameter configuration

**5. Multi-Stock Coordination**
- **Impact:** Data loading, training batching, inference pipeline
- **Concern:** 30 stocks with different characteristics need unified approach
- **Mitigation:** Channel-independent dataset architecture, per-stock normalization
- **Components:** Data loading, training infrastructure, production serving

**6. Production Readiness & Monitoring**
- **Impact:** Deployment architecture, operational requirements
- **Concern:** Financial models require continuous performance monitoring
- **Mitigation:** Performance tracking, drift detection, retraining triggers
- **Components:** Production deployment, monitoring layer, inference pipeline

**7. Previous Implementation Issues**
- **Impact:** Architecture decisions, validation approach
- **Concern:** 4 previous failed attempts using wrong architectures
- **Mitigation:** Strict adherence to TimesFM methodology, comprehensive testing
- **Components:** All architectural layers, testing strategy, validation framework

### Architecture Gap Analysis

**Current State → Target State Transformations:**

```
Custom Transformer → Actual TimesFM 2.5 Foundation Model
AdamW Optimizer → SGD with Momentum (lr=1e-4)
Raw OHLCV Data → Log-transformed Financial Features  
Single Series → Multi-Stock Data Loaders
Basic MAE/RMSE → Statistical Validation (DM Tests)
Zero-shot Performance → Fine-tuned (30% loss reduction expected)
Manual Testing → Comprehensive Testing Strategy (Unit → Integration → E2E)
No Deployment → Production Inference Pipeline
```

**Critical Architecture Corrections Needed:**
1. **Model Architecture:** Replace custom transformers with TimesFM 2.5 + LoRA
2. **Data Processing:** Add financial-specific transformations (log, volatility, clipping)
3. **Training Strategy:** Implement SGD optimizer with financial methodology
4. **Multi-Stock Handling:** Channel-independent dataset architecture
5. **Validation Rigor:** Statistical significance testing (Diebold-Mariano)
6. **Production Capability:** Real-time inference pipeline for 30 stocks

---

**Status:** Project context analysis complete - ready for architectural decision making

---

## Starter Template Evaluation

### Primary Technology Domain

**Machine Learning / Financial Time Series** based on project requirements analysis.

This is **not** a traditional web/mobile application project. It's a specialized ML/Financial Engineering project requiring research-based methodology and proven financial fine-tuning approaches.

### Starter Options Considered

**Traditional Starter Templates:** Not applicable - this is a specialized ML project, not a web/mobile application.

**Research-Based "Starter" Options Evaluated:**

#### Option 1: pfnet-research/timesfm_fin (Proven Financial Methodology)
- **Source:** [pfnet-research/timesfm_fin - GitHub](https://github.com/pfnet-research/timesfm_fin)
- **Research:** [Financial Fine-tuning a Large Time Series Model](https://arxiv.org/html/2412.09880v1) (December 2024)
- **Status:** ✅ Latest research with proven financial methodology
- **Results:** 30% loss reduction on S&P500 vs baseline TimesFM

**Provides:**
- Proven SGD optimizer configuration (momentum=0.9, NOT AdamW)
- Financial data handling (log transformation, clipping, normalization)
- Pre-configured hyperparameters for financial time series
- Mock trading utilities and backtesting framework

#### Option 2: TimesFM 2.5 + HuggingFace Transformers (Latest Base Model)
- **Source:** [google-research/timesfm](https://github.com/google-research/timesfm)
- **Model:** [google/timesfm-2.5-200m-transformers](https://huggingface.co/google/timesfm-2.5-200m-transformers)
- **Status:** ✅ Latest base model (September 2025)
- **Integration:** Native HuggingFace Transformers support (v4.51.3+)

**Provides:**
- Latest TimesFM 2.5 with 200M parameters, 16K context length
- Official HuggingFace integration with LoRA support
- Modern framework with active maintenance
- Production-ready deployment capabilities

### Selected Starter: Hybrid Approach (Recommended)

**Rationale for Selection:**

The hybrid approach combines the **latest TimesFM 2.5 technology** with **proven financial methodology** from pfnet-research, adapted for **Vietnamese market specifics**.

**Key Advantages:**
- **Latest Model:** TimesFM 2.5 capabilities with modern framework
- **Proven Methodology:** Financial fine-tuning best practices (30% improvement demonstrated)
- **Market Adaptation:** Vietnamese stock-specific features (TET holidays, local patterns)
- **Production Ready:** Research-backed approach with modern infrastructure
- **Architecture Alignment:** Matches your implementation plan's correction of previous failures

**Initialization Commands:**

```bash
# Install latest TimesFM 2.5 infrastructure
pip install transformers>=4.51.3
pip install torch>=2.0.0
pip install peft>=0.5.0
pip install accelerate>=0.24.0

# Clone reference implementation for proven methodology
git clone https://github.com/pfnet-research/timesfm_fin.git

# Install financial-specific dependencies
pip install scipy>=1.10.0  # Statistical testing
pip install pandas>=2.0.0   # Financial data processing
pip install scikit-learn>=1.3.0  # Metrics and validation
```

**Architectural Decisions Provided by Starter:**

**Language & Runtime:**
- Python 3.10+ with ML ecosystem (PyTorch, NumPy, Pandas)
- GPU-accelerated training (CUDA) with bfloat16 precision
- Type hints and modern Python practices

**Core Framework:**
- **Base Model:** TimesFM 2.5 (200M parameters, 16K context length)
- **Fine-tuning:** PEFT LoRA (r=4, α=8) for parameter-efficient training
- **Integration:** HuggingFace Transformers (v4.51.3+)

**Financial Methodology (from pfnet-research):**
- **Optimizer:** SGD with momentum=0.9 (NOT AdamW - financial standard)
- **Learning Rate:** 1e-4 with cosine annealing (5 warmup + 95 decay)
- **Gradient Clipping:** max_norm=1.0 for training stability
- **Data Processing:** Log transformation, financial clipping (-5,5 range)

**Build Tooling:**
- **Training:** PyTorch with accelerate for distributed training
- **Validation:** scipy.stats for statistical testing (Diebold-Mariano)
- **Experiment Tracking:** MLflow or Weights & Biases integration
- **Production:** FastAPI for inference serving

**Code Organization:**
```
stockvoli-research/
├── data/
│   ├── raw/prices/           # Existing 30 stocks data
│   ├── processed/            # Financial transformations
│   └── features/              # Vietnamese market features
├── src/
│   ├── data_processing.py     # Financial data pipeline
│   ├── vn30_dataset.py        # Multi-stock dataset
│   ├── timesfm_finetuner.py   # Model training
│   ├── validation.py          # Statistical testing
│   └── inference.py          # Production serving
├── configs/
│   └── config.yaml            # System configuration
├── tests/                     # Comprehensive test suite
└── models/                    # Checkpoints and exports
```

**Development Experience:**
- **GPU Development:** CUDA-enabled training with memory optimization
- **Experiment Tracking:** Training logs, checkpoint management
- **Testing Framework:** Unit → Integration → Statistical validation
- **Documentation:** Research-backed methodology with clear rationale

**Key Architectural Patterns Established:**
1. **Financial-First Design:** All decisions prioritize financial data characteristics
2. **Statistical Rigor:** Validation framework requires significance testing
3. **Catastrophic Forgetting Prevention:** LoRA adapters preserve pre-trained patterns
4. **Multi-Stock Architecture:** Channel-independent approach for portfolio context
5. **Production Readiness:** Monitoring, deployment, and continuous validation

**Note:** This hybrid starter provides the foundation for correcting the 4 failed implementations identified in your research. The reference implementation addresses the exact technical issues (wrong optimizer, missing log transformation, custom transformers vs actual TimesFM).

**Next Steps:**
1. Verify GPU environment meets requirements (16GB+ VRAM)
2. Test TimesFM 2.5 model loading with sample data
3. Implement financial data processing pipeline
4. Configure LoRA adapters with proven hyperparameters
5. Establish statistical validation framework

---

**Status:** Starter template evaluation complete - hybrid approach selected for implementation

---

## Core Architectural Decisions

### Decision Priority Analysis

**Critical Decisions (Block Implementation):**
1. ✅ **Multi-Stock Dataset Architecture** - Required for training pipeline (Channel-Independent approach)
2. ✅ **Local GPU Environment** - Required for TimesFM 2.5 training (8GB+ VRAM minimum)
3. ✅ **Simple Inference Script** - Required for validation and production testing

**Important Decisions (Shape Architecture):**
4. ✅ **JSON Experiment Tracking** - Shapes training workflow and result organization
5. ✅ **Basic Metrics Logging** - Shapes validation approach and success criteria tracking

**Deferred Decisions (Post-MVP):**
- Web API deployment (can add FastAPI later if needed for production serving)
- Advanced monitoring (can upgrade to MLflow/Prometheus if model goes to production)
- Cloud GPU scaling (can migrate to AWS/GCP if local GPU insufficient)

### Data Architecture

**Decision:** **Channel-Independent Multi-Stock Dataset**

**Rationale:**
- Proven by pfnet-research methodology (30% loss reduction on S&P500 demonstrated)
- Handles different Vietnamese stock price ranges properly (VCB ~90K vs HPG ~20K VND)
- Prevents cross-stock contamination during training
- Aligns with financial time series best practices
- Follows Issue #230 resolution for multiple time series

**Implementation:**
```python
# Each stock as separate univariate time series
stock_data_dict = {
    'VCB': vcb_realized_volatility,
    'VIC': vic_realized_volatility, 
    'VNM': vnm_realized_volatility,
    # ... all 30 stocks independently
}
```

**Technical Considerations:**
- Per-stock normalization for different price ranges
- Random window sampling across all stocks (5,000+ samples)
- Time-based train/test split (80/20) with no temporal leakage
- Vietnamese market feature integration (TET holidays, trading patterns)

**Affects:** Data loading pipeline, training batching strategy, validation approach, model architecture design

**Provided by Starter:** Partially - methodology from pfnet-research, adapted for Vietnamese market

---

### Training Infrastructure

**Decision:** **Custom JSON + File-Based Experiment Tracking**

**Rationale:**
- Lightweight, no external service dependencies (MLflow/W&B not needed for research phase)
- Direct control over experiment metadata and hyperparameters
- Perfect for single-developer research workflow
- Easy to version control with git
- Faster iteration without infrastructure setup

**Implementation:**
```python
# Simple JSON-based experiment tracking
import json
from datetime import datetime

experiment_log = {
    "experiment_id": "vn30_timesfm_v1",
    "timestamp": datetime.now().isoformat(),
    "config": {
        "model": "timesfm-2.5-200m",
        "lora_r": 4,
        "lora_alpha": 8,
        "learning_rate": 1e-4,
        "optimizer": "SGD",
        "momentum": 0.9
    },
    "training": {
        "epoch": epoch,
        "train_loss": float(train_loss),
        "eval_loss": float(eval_loss),
        "learning_rate": float(current_lr)
    },
    "metrics": {
        "r2_score": float(r2_score),
        "mae": float(mae),
        "sharpe_ratio": float(sharpe_ratio),
        "dm_significance": bool(dm_significant)
    }
}

# Save to JSON file
with open(f'experiments/experiment_{epoch}.json', 'w') as f:
    json.dump(experiment_log, f, indent=2)
```

**Technical Considerations:**
- Store experiments in `experiments/` directory with timestamp-based naming
- Include hyperparameters, metrics, and model checkpoints in each experiment log
- Simple Python scripts to analyze results across experiments
- JSON structure allows for easy comparison and plotting

**Affects:** Training loop implementation, checkpoint management, result analysis workflow

**Provided by Starter:** No - custom choice for simplicity and research focus

---

### Model Serving & Inference

**Decision:** **Simple Python Script - Basic Inference, No Web Interface**

**Rationale:**
- Direct model access for research and validation
- No web framework overhead (FastAPI/Flask not needed for prototype)
- Batch processing for 30 stocks daily predictions
- Sub-second response time achievable with direct inference
- Focus on ML accuracy over API complexity

**Implementation:**
```python
# Simple inference script for VN30 volatility predictions
import torch
import numpy as np
from transformers import TimesFm2_5ModelForPrediction

def predict_vn30_volatility(model_path, stock_data_dict):
    """
    Generate volatility predictions for all VN30 stocks
    
    Args:
        model_path: Path to fine-tuned LoRA adapters
        stock_data_dict: Dictionary of stock contexts
        
    Returns:
        Dictionary of volatility predictions per stock
    """
    # Load fine-tuned model
    model = TimesFm2_5ModelForPrediction.from_pretrained(model_path)
    model.eval()
    
    predictions = {}
    with torch.no_grad():
        for stock_name, stock_context in stock_data_dict.items():
            # Prepare context for each stock
            context = torch.tensor(stock_context).unsqueeze(0)
            
            # Generate prediction
            prediction = model(past_values=context)
            
            # Store prediction
            predictions[stock_name] = prediction.item()
    
    return predictions

# Usage example
if __name__ == "__main__":
    model_path = "models/vn30_timesfm_finetuned"
    stock_data = load_vn30_contexts()  # Your data loading function
    
    predictions = predict_vn30_volatility(model_path, stock_data)
    
    for stock, pred in predictions.items():
        print(f"{stock}: {pred:.6f}")
```

**Technical Considerations:**
- Supports batch processing for all 30 stocks
- Model loading with fine-tuned LoRA adapters
- GPU acceleration if available, CPU fallback
- Simple command-line interface for daily predictions
- Results can be saved to CSV/JSON for further analysis

**Affects:** Production deployment strategy, user interface, integration capabilities

**Provided by Starter:** No - research-focused choice prioritizing simplicity

---

### Monitoring & Observability

**Decision:** **Basic Metrics - R², MAE, Sharpe Ratio in Logs Only**

**Rationale:**
- Focuses on financial ML critical metrics from success criteria
- No complex observability infrastructure (Prometheus/Grafana not needed)
- Direct tracking of key performance indicators
- Simple log file analysis for performance trends
- Aligns with implementation plan validation requirements

**Implementation:**
```python
# Basic financial metrics logging
import logging
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def log_financial_metrics(epoch, actuals, predictions, returns=None):
    """
    Log core financial ML metrics
    
    Args:
        epoch: Current training epoch
        actuals: Actual volatility values
        predictions: Model predictions
        returns: Optional returns for Sharpe ratio calculation
    """
    # Core metrics
    r2 = r2_score(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    
    # Financial metrics
    sharpe_ratio = calculate_sharpe_ratio(predictions, returns) if returns else None
    direction_accuracy = calculate_direction_accuracy(actuals, predictions)
    
    # Log comprehensive metrics
    logging.info(f"Epoch {epoch:3d} | R²: {r2:.4f} | MAE: {mae:.6f} | RMSE: {rmse:.6f}")
    logging.info(f"           | Sharpe: {sharpe_ratio:.4f if sharpe_ratio else 'N/A'} | Dir Acc: {direction_accuracy:.2%}")
    
    return {
        'r2': r2,
        'mae': mae, 
        'rmse': rmse,
        'sharpe_ratio': sharpe_ratio,
        'direction_accuracy': direction_accuracy
    }

def calculate_sharpe_ratio(predictions, returns, risk_free_rate=0.02):
    """Calculate annualized Sharpe ratio for volatility predictions"""
    if returns is None:
        return None
    
    # Simple Sharpe ratio calculation
    excess_returns = returns - risk_free_rate / 252  # Daily adjustment
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    annualized_sharpe = sharpe * np.sqrt(252)  # Annualize
    
    return annualized_sharpe
```

**Technical Considerations:**
- Log to both file (`training.log`) and console
- Track success criteria metrics (R² > 0.5, Sharpe 0.8-1.5)
- Simple CSV extraction from logs for plotting and analysis
- Diebold-Mariano test results logged when available
- Per-stock performance tracking for portfolio analysis

**Affects:** Performance monitoring, model quality tracking, alerting, success criteria validation

**Provided by Starter:** No - aligned with financial ML requirements from implementation plan

---

### GPU Infrastructure & Deployment

**Decision:** **Local Laptop GPU Training**

**Rationale:**
- No cloud costs or dependencies for research phase
- Direct hardware control for experimentation
- Suitable for prototype/research timeline (6-week implementation)
- Scales from development to validation without environment changes
- Lower barrier to rapid iteration

**Technical Requirements:**
- **GPU:** 8GB+ VRAM minimum (16GB+ preferred for TimesFM 2.5)
- **CUDA:** NVIDIA GPU with CUDA 11.8+ support
- **RAM:** 16GB+ system RAM recommended
- **Storage:** 50GB+ free space for models, checkpoints, and datasets
- **OS:** Windows 11/Linux with GPU driver support

**Implementation:**
```python
# GPU environment setup and validation
import torch

def setup_gpu_environment():
    """Configure and validate GPU training environment"""
    # Check CUDA availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"GPU detected: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.2f} GB")
        
        # Validate memory requirements (8GB minimum)
        if gpu_memory < 8:
            print("⚠️  WARNING: GPU memory below 8GB - training may be slow")
        
        return device
    else:
        print("❌ No CUDA GPU detected - training on CPU (not recommended)")
        return torch.device('cpu')

# Use bfloat16 for memory efficiency
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# Load model with proper device configuration
model = TimesFm2_5ModelForPrediction.from_pretrained(
    "google/timesfm-2.5-200m-transformers",
    torch_dtype=dtype,
    device_map="auto"
)
```

**Technical Considerations:**
- Batch size adjustment based on available VRAM
- Gradient accumulation if GPU memory limited
- Mixed precision training (bfloat16) for memory efficiency
- Checkpoint management to prevent training loss
- Temperature monitoring for long training sessions

**Affects:** Training batch size, model size limits, development workflow, iteration speed

**Provided by Starter:** No - chosen based on available hardware and project scope

---

### Decision Impact Analysis

**Implementation Sequence:**

1. **Week 1 (Foundation):** Verify local GPU meets TimesFM 2.5 requirements → Test model loading
2. **Week 2 (Data Engineering):** Build channel-independent dataset pipeline → Validate Vietnamese stock processing
3. **Week 3-4 (Model Implementation):** Implement training with JSON experiment tracking → Monitor convergence via basic logging
4. **Week 5 (Validation):** Add Diebold-Mariano testing to metrics logging → Validate statistical significance
5. **Week 6 (Production):** Create simple inference script → Test daily batch predictions

**Cross-Component Dependencies:**

- **GPU capability** → **Training batch size** → **Model convergence quality**
- **Channel-independent dataset** → **Training pipeline** → **Validation approach** → **Statistical testing framework**
- **JSON experiment tracking** → **Checkpoint management** → **Best model selection** → **Final performance**
- **Simple inference** → **Production testing** → **Business validation** → **Success criteria measurement**
- **Basic logging** → **Performance tracking** → **Model quality assessment** → **Deployment readiness**

**Risk Mitigation:**

- **GPU Memory Constraints:** Implement gradient accumulation, reduce batch size if needed
- **Simple Infrastructure:** Upgrade to MLflow/FastAPI if project scales to production needs
- **Local Development:** Cloud GPU backup option if local hardware insufficient
- **Basic Monitoring:** Enhanced observability if model performance issues detected

**Technology Stack Summary:**

**Core ML Framework:**
- Python 3.10+ with PyTorch 2.0+ (bfloat16 support)
- Transformers ≥4.51.3 (TimesFM 2.5 integration)
- PEFT ≥0.5.0 (LoRA adapters, r=4, α=8)

**Financial ML Stack:**
- NumPy ≥1.24.0, Pandas ≥2.0.0 (data processing)
- SciPy ≥1.10.0 (Diebold-Mariano statistical testing)
- Scikit-learn ≥1.3.0 (R², MAE, Sharpe ratio calculation)

**Development Environment:**
- JSON file-based experiment tracking (no MLflow/W&B)
- Python logging for basic metrics (no Prometheus/Grafana)
- Git version control for experiment reproducibility
- Local GPU development (no cloud dependencies)

---

**Status:** Core architectural decisions complete - ready for implementation patterns

---

## Implementation Patterns & Consistency Rules

### Pattern Categories Defined

**Critical Conflict Points Identified:** 6 areas where AI agents could make different choices that would complicate experimentation

**Philosophy:** Maximum simplicity for rapid experimentation - focus on model performance metrics (QLIKE, R², RMSE, MSE) over infrastructure complexity

### Naming Patterns

**Python Code Naming:**
- **Functions:** `snake_case` (e.g., `calculate_volatility`, `train_model`)
- **Variables:** `snake_case` (e.g., `stock_data`, `model_predictions`)
- **Classes:** `PascalCase` (e.g., `VolatilityCalculator`, `ModelTrainer`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `BATCH_SIZE`, `LEARNING_RATE`)

**File Naming:**
- **Scripts:** `snake_case.py` (e.g., `data_processing.py`, `model_training.py`)
- **Notebooks:** `descriptive_name.ipynb` (e.g., `volatility_analysis.ipynb`)
- **Data files:** `descriptive_name.csv` (e.g., `vcb_processed.csv`)

**Experiment Naming:**
- **Checkpoints:** `model_epoch_{N}_{metric}.pt` (e.g., `model_epoch_15_r2_0.65.pt`)
- **Results:** `experiment_{date}_{config}.json` (e.g., `experiment_20250609_sgd.json`)

**Example:**
```python
# Good naming conventions
def calculate_realized_volatility(prices, window=20):
    batch_size = 32
    learning_rate = 1e-4
    MAX_EPOCHS = 100
```

### Structure Patterns

**Project Organization (Simple Structure):**
```
stockvoli-research/
├── data_processing.py      # Financial data transformations
├── model_training.py         # TimesFM training script  
├── model_evaluation.py       # Metrics calculation (QLIKE, R², RMSE, MSE)
├── inference.py              # Simple prediction script
├── utils.py                  # Shared utility functions
├── experiments/              # JSON experiment logs
├── models/                   # Model checkpoints
├── configs/                  # Configuration files
└── data/                     # Raw and processed data
```

**File Organization Rules:**
- **All processing scripts** in root directory (no complex subdirectories)
- **Experiments** go in `experiments/` with JSON format
- **Models** saved to `models/` with descriptive checkpoint names
- **Tests** (if any) in `tests/` directory co-located with source

### Data Processing Patterns

**🎯 Simple Functional Approach:**

```python
# Simple function-based data processing
def apply_log_transformation(prices):
    """Apply log transformation to prevent extreme values"""
    return np.log(prices)

def calculate_realized_volatility(returns, window=20):
    """Calculate realized volatility over specified window"""
    return returns.rolling(window).std()

def add_vietnamese_features(data):
    """Add Vietnamese market-specific features"""
    data['is_tet_period'] = detect_tet_holidays(data.index)
    data['day_of_week'] = data.index.dayofweek
    return data

# Processing pipeline
def process_stock_data(stock_file):
    """Complete processing pipeline for single stock"""
    df = pd.read_csv(stock_file)
    df['log_close'] = apply_log_transformation(df['close'])
    df['log_returns'] = df['log_close'].diff()
    df['RV_20'] = calculate_realized_volatility(df['log_returns'], 20)
    df = add_vietnamese_features(df)
    return df
```

**Pattern:** All transformations as simple, testable functions

### Model Training Patterns

**🎯 Simple Script-Based Training:**

```python
# Simple training script structure
def train_model(model, train_loader, config):
    """Simple training loop with metrics tracking"""
    optimizer = configure_optimizer(model, config)
    
    for epoch in range(config['epochs']):
        # Training
        train_loss = train_one_epoch(model, train_loader, optimizer)
        
        # Evaluation
        eval_loss, metrics = evaluate_model(model, test_loader)
        
        # Log simple metrics
        print(f"Epoch {epoch}: Loss={train_loss:.4f}, R²={metrics['r2']:.4f}")
        
        # Save checkpoint if best
        if metrics['r2'] > best_r2:
            save_checkpoint(model, f"model_epoch_{epoch}_r2_{metrics['r2']:.3f}.pt")

# Usage
if __name__ == "__main__":
    model = load_timesfm_model()
    train_model(model, train_loader, training_config)
```

**Pattern:** Script-based training with simple progress tracking

### Experiment Tracking Patterns

**🎯 Simple JSON Logging:**

```python
# Simple experiment logging
def log_metrics(epoch, metrics_dict):
    """Log metrics to simple JSON format"""
    experiment_log = {
        'timestamp': datetime.now().isoformat(),
        'epoch': epoch,
        'metrics': metrics_dict  # {'r2': 0.65, 'rmse': 0.023, 'mse': 0.0005, 'qlike': 0.045}
    }
    
    with open(f'experiments/epoch_{epoch}.json', 'w') as f:
        json.dump(experiment_log, f, indent=2)

# Usage during training
metrics = {
    'r2': r2_score(actuals, predictions),
    'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
    'mse': mean_squared_error(actuals, predictions),
    'qlike': calculate_qlike(actuals, predictions)
}
log_metrics(epoch, metrics)
```

**Pattern:** Simple JSON files per epoch, flat structure

### Evaluation Metrics Patterns

**🎯 Standard Financial Metrics Functions:**

```python
# Core metrics calculation functions
def calculate_qlike(actuals, predictions):
    """Calculate QLIKE metric for volatility forecasting"""
    return np.sum(actuals/predictions + np.log(predictions) - 1)

def calculate_r2(actuals, predictions):
    """Calculate R² score"""
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    return 1 - (ss_res / ss_tot)

def calculate_rmse(actuals, predictions):
    """Calculate Root Mean Square Error"""
    return np.sqrt(np.mean((actuals - predictions) ** 2))

def calculate_mse(actuals, predictions):
    """Calculate Mean Square Error"""
    return np.mean((actuals - predictions) ** 2)

# Comprehensive evaluation
def evaluate_all_metrics(actuals, predictions):
    """Calculate all key metrics"""
    return {
        'qlike': calculate_qlike(actuals, predictions),
        'r2': calculate_r2(actuals, predictions),
        'rmse': calculate_rmse(actuals, predictions),
        'mse': calculate_mse(actuals, predictions)
    }
```

**Pattern:** Standard metric functions, consistent naming, return simple dictionaries

### Error Handling Patterns

**🎯 Simple Validation-First Approach:**

```python
# Simple validation and error handling
def validate_gpu_memory():
    """Check GPU memory availability before training"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory < 8:  # Minimum 8GB requirement
            raise ValueError(f"Insufficient GPU memory: {gpu_memory:.1f}GB < 8GB required")
    return True

def validate_data_quality(data):
    """Validate data has no critical issues"""
    if data.isnull().any().any():
        raise ValueError("Data contains NaN values")
    if len(data) < 100:
        raise ValueError(f"Insufficient data points: {len(data)} < 100 minimum")
    return True

# Simple error handling in training
try:
    validate_gpu_memory()
    validate_data_quality(train_data)
    train_model(model, train_data)
except ValueError as e:
    print(f"Validation failed: {e}")
except Exception as e:
    print(f"Training error: {e}")
```

**Pattern:** Pre-flight validation, simple try/except blocks

### Configuration Patterns

**🎯 Keep Existing config.yaml:**

```python
# Simple configuration loading
import yaml

def load_config():
    """Load configuration from existing config.yaml"""
    with open('configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)

# Usage
config = load_config()
stocks = config['data']['stocks']  # List of 30 stocks
training_params = config['incremental_learning']['training']
```

**Pattern:** Use existing config.yaml structure, no new configuration systems

### Dataset Organization Patterns

**🎯 Simple Dictionary-Based Approach:**

```python
# Simple dataset organization
def load_all_stocks():
    """Load all 30 stocks into simple dictionary"""
    stock_data = {}
    for stock in STOCKS:  # From config
        file_path = f"data/raw/prices/{stock}_ohlcv.csv"
        stock_data[stock] = process_stock_data(file_path)
    return stock_data

# Usage
stock_data_dict = load_all_stocks()
for stock_name, processed_data in stock_data_dict.items():
    print(f"{stock_name}: {len(processed_data)} observations")
```

**Pattern:** Dictionary-based dataset, simple processing loop

### Enforcement Guidelines

**All AI Agents MUST:**

1. **Use snake_case** for all Python functions and variables
2. **Save experiments** as JSON files in `experiments/` directory
3. **Calculate standard metrics**: QLIKE, R², RMSE, MSE using provided function names
4. **Use simple functions** over classes for data processing
5. **Validate before training** (GPU memory, data quality)
6. **Log progress** with simple print statements showing key metrics
7. **Save checkpoints** with descriptive names including epoch and key metrics

**Pattern Enforcement:**

- **Verification:** Run `model_evaluation.py` to check all metrics are calculated correctly
- **Documentation:** All functions must have docstrings explaining inputs/outputs
- **Testing:** If tests exist, they must use same metric functions as training code

**Quick Pattern Reference:**
```python
# DO: Use simple functions with standard names
def calculate_rmse(actuals, predictions):
    return np.sqrt(np.mean((actuals - predictions) ** 2))

# DON'T: Create complex classes for simple operations
class RMSECalculator:
    def calculate(self, actuals, predictions):
        return np.sqrt(np.mean((actuals - predictions) ** 2))

# DO: Use consistent JSON structure for experiments
experiment_log = {'epoch': 1, 'metrics': {'r2': 0.65, 'rmse': 0.023, 'mse': 0.0005, 'qlike': 0.045}}

# DON'T: Use inconsistent metric names or structures
experiment_log = {'epoch': 1, 'R2': 0.65, 'RootMeanSquareError': 0.023}  # Wrong names!
```

### Pattern Examples

**Good Examples:**
```python
# Simple, consistent metric calculation
metrics = evaluate_all_metrics(actual_volatility, predicted_volatility)
print(f"R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.6f}, MSE: {metrics['mse']:.6f}, QLIKE: {metrics['qlike']:.6f}")

# Consistent experiment logging
log_metrics(epoch, metrics)
save_checkpoint(model, f"model_epoch_{epoch}_r2_{metrics['r2']:.3f}.pt")

# Simple data processing pipeline
for stock in stocks:
    stock_data[stock] = process_stock_data(f"data/raw/prices/{stock}_ohlcv.csv")
```

**Anti-Patterns to Avoid:**
```python
# DON'T: Inconsistent metric names
metrics = {'R2': 0.65, 'root_mean_square_error': 0.023}  # Wrong!

# DON'T: Complex class structures for simple operations  
class DataProcessorFactory:
    def create_processor(self, type): ...  # Overcomplicated!

# DON'T: Inconsistent experiment naming
torch.save(model, f"model_{epoch}.pt")  # Missing metrics in name!

# DON'T: Custom exception classes for simple validation
class InsufficientGPUMemoryError(Exception): ...  # Overcomplicated!
```

**Implementation Priority:**
1. **Week 1:** Implement standard metric functions (QLIKE, R², RMSE, MSE)
2. **Week 2:** Create simple data processing functions following naming patterns
3. **Week 3-4:** Build training script with consistent checkpoint naming
4. **Week 5:** Implement evaluation script using standard metric functions
5. **Week 6:** Simple inference script with consistent output format

---

**Status:** Implementation patterns defined for rapid experimentation with focus on key financial metrics

---

## Project Structure & Boundaries

### Complete Project Directory Structure

```
stockvoli-research/
├── README.md                          # Project overview and quick start
├── requirements.txt                   # Python dependencies (PyTorch, Transformers, etc.)
├── .gitignore                         # Git ignore patterns
├── configs/
│   └── config.yaml                    # System configuration (30 stocks, parameters)
├── data/
│   ├── raw/prices/                    # Original OHLCV data (30 stocks CSV files)
│   ├── processed/                     # Financial transformed data
│   └── features/                      # Vietnamese market features
├── models/                            # Model checkpoints and exports
│   ├── timesfm/                       # TimesFM 2.5 base model cache
│   ├── checkpoints/                   # Training checkpoints
│   └── final_models/                  # Best models for production
├── experiments/                       # JSON experiment logs
│   ├── experiment_20250609_sgd.json   # Per-epoch metrics (QLIKE, R², RMSE, MSE)
│   └── results_summary.json          # Best results across experiments
├── src/                               # Core source code (simple functional approach)
│   ├── data_processing.py            # Financial data transformations
│   ├── vn30_dataset.py                # Multi-stock dataset creation
│   ├── model_training.py              # TimesFM training script
│   ├── model_evaluation.py           # Metrics calculation (QLIKE, R², RMSE, MSE)
│   ├── statistical_tests.py           # Diebold-Mariano statistical testing
│   ├── inference.py                   # Simple prediction script
│   └── utils.py                       # Shared utility functions
├── tests/                             # Test suite
│   ├── test_data_processing.py        # Test financial transformations
│   ├── test_metrics.py                # Test metric calculations
│   └── test_training.py               # Test training pipeline
├── setup/                             # Setup and validation scripts
│   ├── check_environment.py           # GPU and dependency validation
│   ├── download_timesfm.py           # TimesFM 2.5 model download
│   └── test_data_loading.py          # Test 30 stocks data loading
├── docs/                              # Documentation
│   ├── methodology.md                # Financial fine-tuning methodology
│   ├── results.md                     # Experiment results and performance
│   └── deployment_guide.md           # Simple inference usage
└── notebooks/                         # Jupyter notebooks for exploration
    ├── data_exploration.ipynb        # Vietnamese stock data analysis
    ├── model_experiments.ipynb       # Quick model testing
    └── results_visualization.ipynb   # Performance visualization
```

### Architectural Boundaries

**Component Communication Boundaries:**

```
Data Processing → Model Training → Model Evaluation → Simple Inference
     (CSV files)      (PyTorch)       (Metrics)        (Predictions)
         ↓                ↓                ↓                  ↓
    Processed      Trained         QLIKE/R²/RMSE    Daily stock
      Data         Models           /MSE          Volatility
                                       Forecasts
```

**Integration Boundaries:**

1. **Data Layer** → **Processing Layer**
   - **Boundary:** Raw OHLCV CSV files → Processed financial features
   - **Communication:** File I/O, pandas DataFrame operations
   - **Data Flow:** `data/raw/prices/*.csv` → `data_processing.py` → `data/processed/`

2. **Processing Layer** → **Training Layer**
   - **Boundary:** Processed financial data → Multi-stock training dataset
   - **Communication:** Dictionary-based dataset passing
   - **Data Flow:** `vn30_dataset.py` → `model_training.py` → `models/checkpoints/`

3. **Training Layer** → **Evaluation Layer**
   - **Boundary:** Trained model → Metric calculations
   - **Communication:** Model checkpoint loading, numpy arrays
   - **Data Flow:** `models/checkpoints/` → `model_evaluation.py` → `experiments/*.json`

4. **Evaluation Layer** → **Inference Layer**
   - **Boundary:** Validated model → Production predictions
   - **Communication:** Model loading, dictionary inputs/outputs
   - **Data Flow:** `models/final_models/` → `inference.py` → Daily predictions

### Requirements to Structure Mapping

**Phase → Directory Mapping:**

**Phase 1: Foundation Setup → `setup/` directory**
- GPU Environment Check → `setup/check_environment.py`
- TimesFM 2.5 Loading → `setup/download_timesfm.py`
- Data Validation → `setup/test_data_loading.py`

**Phase 2: Data Engineering → `src/data_processing.py` + `src/vn30_dataset.py`**
- Financial Transformations → `data_processing.py` functions
- Multi-Stock Dataset → `vn30_dataset.py` dataset class
- Vietnamese Features → `data_processing.py` feature functions

**Phase 3: Model Implementation → `src/model_training.py`**
- TimesFM 2.5 Loading → Training script model initialization
- LoRA Configuration → Training script adapter setup
- Training Loop → `model_training.py` main function
- Checkpoint Management → Training script save/load functions

**Phase 4: Validation & Testing → `src/model_evaluation.py` + `src/statistical_tests.py`**
- Financial Metrics → `model_evaluation.py` metric functions
- Statistical Testing → `statistical_tests.py` Diebold-Mariano
- Backtesting Framework → `model_evaluation.py` Sharpe ratio calculation

**Phase 5: Production Deployment → `src/inference.py`**
- Model Loading → Inference script load function
- Batch Predictions → Inference script predict function
- Results Export → Inference script save results

### Integration Points

**Internal Communication:**

**Data Flow Pipeline:**
```
Raw CSV Data → Processing Functions → Processed Dict → Training Dataset → Trained Model → Evaluation Metrics → Experiment Logs
```

**Component Communication:**
- **Data → Training:** Dictionary-based dataset passing
- **Training → Evaluation:** Model checkpoint file paths
- **Evaluation → Experiment:** JSON file appending
- **All → Config:** Read from `configs/config.yaml`

**External Integrations:**

**HuggingFace Integration:**
- **Point:** Model download and loading
- **Implementation:** `setup/download_timesfm.py`, `src/model_training.py`
- **Boundary:** HuggingFace API calls, model caching in `models/timesfm/`

**Vietnamese Market Data Integration:**
- **Point:** Local CSV files with OHLCV data
- **Implementation:** `data_processing.py` reads from `data/raw/prices/`
- **Boundary:** File I/O operations, no external APIs

### File Organization Patterns

**Configuration Files:**
- **Location:** `configs/` directory
- **Structure:** Single `config.yaml` with all parameters
- **Pattern:** Centralized configuration, no environment files needed

**Source Code Organization:**
- **Location:** `src/` directory (root level for simplicity)
- **Structure:** Functional scripts, no complex package hierarchy
- **Pattern:** One script per major function, simple function-based approach

**Test Organization:**
- **Location:** `tests/` directory (root level)
- **Structure:** Simple test files mirroring source scripts
- **Pattern:** Test file per source file, simple assert-based tests

### Development Workflow Integration

**Development Workflow:**
```
1. Run setup scripts → Validate environment
2. Process stock data → Create financial features
3. Train model → Generate checkpoints
4. Evaluate model → Calculate QLIKE/R²/RMSE/MSE
5. Analyze experiments → Review JSON logs
6. Deploy inference → Generate daily predictions
```

**Iteration Pattern:**
- **Experiment:** Modify parameters in `config.yaml`
- **Train:** Run `src/model_training.py`
- **Evaluate:** Run `src/model_evaluation.py`
- **Compare:** Review `experiments/` JSON files
- **Iterate:** Update config based on results

---

**Status:** Complete project structure defined for rapid experimentation workflow

---

## Architecture Validation Results

### Coherence Validation ✅

**Decision Compatibility:**
All architectural decisions work together seamlessly:
- **Technology Stack Compatibility:** Python 3.10+ → PyTorch 2.0+ → Transformers ≥4.51.3 → PEFT ≥0.5.0 (Compatible versions)
- **Model + Training Approach:** TimesFM 2.5 + LoRA adapters + SGD optimizer → Local GPU training (Compatible approach)
- **Simple Infrastructure:** JSON logging + Basic metrics + Simple scripts (All aligned with simplicity goal)
- **No Contradictions:** All decisions support the experiment-focused, simple infrastructure approach

**Pattern Consistency:**
Implementation patterns fully support architectural decisions:
- **Naming Conventions:** `snake_case` functions → Matches Python ML standards ✅
- **File Organization:** Root-level scripts → Aligns with simplicity and rapid iteration ✅
- **Metric Functions:** Standard QLIKE/R²/RMSE/MSE functions → Consistent across all components ✅
- **Communication Patterns:** Dictionary-based datasets → File-based checkpoints → JSON experiment logs ✅

**Structure Alignment:**
Project structure perfectly supports all architectural decisions:
- **Simple Script Organization:** Root-level `src/` scripts enable functional approach ✅
- **Clear Data Flow:** CSV → Processing → Training → Evaluation → JSON logs pipeline ✅
- **Well-Defined Integration Points:** File I/O boundaries, dictionary passing, model checkpoints ✅

### Requirements Coverage Validation ✅

**Epic/Feature Coverage:**
All 6 implementation phases are architecturally supported:
- **Phase 1 (Foundation):** `setup/` directory with environment validation scripts ✅
- **Phase 2 (Data Engineering):** `src/data_processing.py` + `src/vn30_dataset.py` ✅
- **Phase 3 (Model Implementation):** `src/model_training.py` with TimesFM 2.5 + LoRA ✅
- **Phase 4 (Validation):** `src/model_evaluation.py` + `src/statistical_tests.py` ✅
- **Phase 5 (Production):** `src/inference.py` for daily predictions ✅

**Functional Requirements Coverage:**
All functional requirements are architecturally supported:
- **30 Vietnamese Stocks:** ✅ Configured in `configs/config.yaml`
- **Financial Transformations:** ✅ Log transformation, volatility calculation functions
- **TimesFM 2.5 Integration:** ✅ Model loading and LoRA adapter configuration
- **Statistical Validation:** ✅ Diebold-Mariano test framework, significance testing
- **QLIKE/R²/RMSE/MSE Metrics:** ✅ Standard metric functions with exact signatures

**Non-Functional Requirements Coverage:**
All performance and validation requirements addressed:
- **Model Performance (R² > 0.5):** ✅ Metric calculation and JSON logging
- **Statistical Significance (p < 0.05):** ✅ Diebold-Mariano test implementation
- **Training Stability:** ✅ Basic metrics logging, NaN prevention through log transformation
- **GPU Training (8GB+ VRAM):** ✅ Setup validation script, bfloat16 precision
- **Simple Infrastructure:** ✅ JSON logging, basic metrics, no complex frameworks

### Implementation Readiness Validation ✅

**Decision Completeness:**
All critical architectural decisions are fully documented with versions:
- **Technology Stack:** Python 3.10+, PyTorch 2.0+, Transformers ≥4.51.3, PEFT ≥0.5.0 ✅
- **Model Architecture:** TimesFM 2.5 (200M params), LoRA (r=4, α=8) ✅
- **Optimizer:** SGD with momentum=0.9, lr=1e-4, gradient clipping=1.0 ✅
- **Metrics:** QLIKE, R², RMSE, MSE function signatures fully specified ✅
- **Data Processing:** Log transformation, financial clipping (-5,5) ✅

**Structure Completeness:**
Complete project structure with all files and directories defined:
- **All Source Scripts:** `data_processing.py`, `vn30_dataset.py`, `model_training.py`, `model_evaluation.py`, `statistical_tests.py`, `inference.py`, `utils.py` ✅
- **Configuration:** `configs/config.yaml` with 30 stocks setup ✅
- **Data Organization:** `data/raw/prices/`, `data/processed/`, `data/features/` ✅
- **Model Storage:** `models/timesfm/`, `models/checkpoints/`, `models/final_models/` ✅
- **Experiment Logging:** `experiments/` with JSON format ✅

**Pattern Completeness:**
Comprehensive implementation patterns prevent AI agent conflicts:
- **Naming Conventions:** `snake_case` functions, `UPPER_SNAKE_CASE` constants ✅
- **Metric Functions:** Standard function names enforced across all agents ✅
- **Error Handling:** Validation-first approach, simple try/except blocks ✅
- **Checkpoint Naming:** `model_epoch_{N}_r2_{VALUE}.pt` descriptive format ✅

### Gap Analysis Results

**Critical Gaps:** None ✅ All blocking elements are addressed

**Important Gaps:**
- **Source File Creation:** The `src/` scripts are architecturally defined but not yet created (expected for implementation phase)
- **Statistical Test Implementation:** Diebold-Mariano test framework defined, scipy.stats provides implementation
- **GPU Validation Details:** Environment checking defined, implementation uses PyTorch CUDA detection

**Nice-to-Have Gaps:**
- **Advanced Visualization:** Results visualization could be enhanced (matplotlib/seaborn integration)
- **Backup Strategy:** Experiment/results backup approach not specified (git version control recommended)
- **Extended Documentation:** Technical methodology could be more detailed

### Architecture Completeness Checklist

**Requirements Analysis**
- [x] Project context thoroughly analyzed ✅
- [x] Scale and complexity assessed ✅
- [x] Technical constraints identified ✅
- [x] Cross-cutting concerns mapped ✅

**Architectural Decisions**
- [x] Critical decisions documented with versions ✅
- [x] Technology stack fully specified ✅
- [x] Integration patterns defined ✅
- [x] Performance considerations addressed ✅

**Implementation Patterns**
- [x] Naming conventions established ✅
- [x] Structure patterns defined ✅
- [x] Communication patterns specified ✅
- [x] Process patterns documented ✅

**Project Structure**
- [x] Complete directory structure defined ✅
- [x] Component boundaries established ✅
- [x] Integration points mapped ✅
- [x] Requirements to structure mapping complete ✅

### Implementation Handoff

**AI Agent Guidelines:**

**All AI Agents MUST:**
- Follow architectural decisions exactly as documented (TimesFM 2.5 + LoRA, SGD optimizer, etc.)
- Use implementation patterns consistently (snake_case naming, standard metric functions, etc.)
- Respect project structure and boundaries (scripts in `src/`, data in `data/`, etc.)
- Calculate financial metrics using exact function names: `calculate_qlike()`, `calculate_r2()`, `calculate_rmse()`, `calculate_mse()`
- Save experiments to `experiments/` as JSON files with timestamp format
- Name checkpoints descriptively: `model_epoch_{N}_r2_{VALUE}.pt`
- Use `configs/config.yaml` as single source of truth for all parameters

**First Implementation Priority:**
1. **Setup Phase:** Run `setup/check_environment.py` to validate GPU and dependencies
2. **Data Processing:** Implement `src/data_processing.py` with financial transformations
3. **Dataset Creation:** Implement `src/vn30_dataset.py` with channel-independent approach
4. **Model Training:** Implement `src/model_training.py` with TimesFM 2.5 + LoRA
5. **Evaluation:** Implement `src/model_evaluation.py` with standard metric functions
6. **Statistical Testing:** Implement `src/statistical_tests.py` with Diebold-Mariano

**Architecture Document Reference:**
- For all architectural questions, refer to this document
- Technology stack decisions → Section "Core Architectural Decisions"
- Project structure questions → Section "Project Structure & Boundaries"
- Implementation patterns → Section "Implementation Patterns & Consistency Rules"
- Validation criteria → Section "Success Criteria & Validation"

### Architecture Readiness Assessment

**Overall Status:** ✅ **READY FOR IMPLEMENTATION**

**Confidence Level:** **HIGH** - Architecture is coherent, complete, and experiment-focused with clear patterns

**Key Strengths:**
1. **Research-Focused Design:** Simple patterns enable rapid experimentation with financial ML
2. **Technical Correctness:** Uses actual TimesFM 2.5 (not custom transformers) with proven methodology
3. **Financial ML Rigor:** Proper statistical validation (Diebold-Mariano) and financial metrics (QLIKE)
4. **Technology Alignment:** TimesFM 2.5 + pfnet-research methodology integration for maximum effectiveness
5. **Clear Boundaries:** Well-defined file I/O boundaries and simple communication patterns
6. **Experimentation Ready:** JSON logging, standard metrics, quick iteration approach
7. **GPU-Optimized:** Local laptop GPU training with memory-efficient approach

**Areas for Future Enhancement:**
- **Advanced Monitoring:** Could upgrade from basic logging to MLflow/Prometheus for production
- **Web API Deployment:** Could add FastAPI web interface if production serving needed
- **Distributed Training:** Could implement multi-GPU training if single GPU insufficient
- **Enhanced Visualization:** Could integrate plotting libraries for better results analysis
- **Cloud Backup:** Could add cloud storage backup for experiments and models

**Implementation Timeline Alignment:**
- **Week 1:** Setup validation + data processing implementation ✅ Structurally supported
- **Week 2:** Multi-stock dataset creation ✅ Patterns defined
- **Week 3-4:** Model training implementation ✅ Architecture specified
- **Week 5:** Statistical validation implementation ✅ Framework provided
- **Week 6:** Simple inference deployment ✅ Script structure defined

---

**Status:** Architecture validation complete - READY FOR IMPLEMENTATION