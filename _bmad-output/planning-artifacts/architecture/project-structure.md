# TimesFM VN30 Project Structure & Boundaries

## Complete Project Directory Structure

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

## Architectural Boundaries

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

### Data Access Boundaries

**Raw Data Access:**
- **Location:** `data/raw/prices/{STOCK}_ohlcv.csv`
- **Format:** CSV with columns: date, open, high, low, close, volume
- **Access Pattern:** Read-only via `pandas.read_csv()`
- **Boundary:** No modification of raw data, only read operations

**Processed Data Access:**
- **Location:** `data/processed/{STOCK}_processed.csv`
- **Format:** Financial features (log_returns, RV_20, Vietnamese features)
- **Access Pattern:** Write via `data_processing.py`, read by training/evaluation
- **Boundary:** Clear separation - processing creates, training consumes

**Model Checkpoint Access:**
- **Location:** `models/checkpoints/model_epoch_{N}_r2_{VALUE}.pt`
- **Format:** PyTorch state dictionaries with metadata
- **Access Pattern:** Write during training, read for evaluation/inference
- **Boundary:** Training writes, evaluation reads, inference loads best model

**Experiment Results Access:**
- **Location:** `experiments/experiment_{TIMESTAMP}_{CONFIG}.json`
- **Format:** JSON with epoch, metrics (QLIKE, R², RMSE, MSE), config
- **Access Pattern:** Append-only during training, read for analysis
- **Boundary:** Training logs, analysis scripts read

### Service Boundaries

**Simple Script-Based Services:**

**1. Data Processing Service** (`data_processing.py`)
```python
# Service boundary: Input CSV files → Output processed data
def process_stock_data(stock_file):
    # Input: CSV file path
    # Output: Processed pandas DataFrame
    # No external dependencies beyond pandas/numpy
```

**2. Model Training Service** (`model_training.py`)
```python
# Service boundary: Input processed data → Output trained models
def train_model(processed_data_dict, config):
    # Input: Dictionary of processed stock data
    # Output: Trained model checkpoints
    # Dependencies: TimesFM 2.5, PyTorch, GPU
```

**3. Evaluation Service** (`model_evaluation.py`)
```python
# Service boundary: Input model + data → Output metrics
def evaluate_model(model, test_data):
    # Input: Trained model, test dataset
    # Output: Metrics dict {qlike, r2, rmse, mse}
    # Dependencies: NumPy, SciPy (for statistical tests)
```

**4. Inference Service** (`inference.py`)
```python
# Service boundary: Input model + contexts → Output predictions
def predict_volatility(model, stock_contexts):
    # Input: Trained model, stock context dictionaries
    # Output: Predictions dictionary {stock: volatility_prediction}
    # Dependencies: PyTorch, TimesFM
```

## Requirements to Structure Mapping

**Feature/Phase Mapping:**

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

**Cross-Cutting Concerns:**

**Metric Calculation System:**
- **Location:** `src/model_evaluation.py`
- **Functions:** `calculate_qlike()`, `calculate_r2()`, `calculate_rmse()`, `calculate_mse()`
- **Usage:** Called by training script, evaluation script, statistical tests
- **Integration:** All agents MUST use these exact function names and signatures

**Configuration Management:**
- **Location:** `configs/config.yaml`
- **Usage:** All scripts read configuration from this file
- **Integration:** Single source of truth for 30 stocks, hyperparameters, paths
- **Pattern:** Load once at script startup, pass to functions as needed

**Experiment Tracking:**
- **Location:** `experiments/` directory
- **Format:** JSON files per experiment with timestamp
- **Integration:** Training script appends, analysis scripts read
- **Pattern:** `experiments/experiment_{date}_{config_description}.json`

## Integration Points

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

**Error Handling Communication:**
- **GPU Issues:** Raised in training script, caught in setup validation
- **Data Issues:** Raised in processing script, caught in data validation
- **Model Issues:** Raised in training/evaluation, logged to experiments JSON

**External Integrations:**

**HuggingFace Integration:**
- **Point:** Model download and loading
- **Implementation:** `setup/download_timesfm.py`, `src/model_training.py`
- **Boundary:** HuggingFace API calls, model caching in `models/timesfm/`

**pfnet-research Integration:**
- **Point:** Financial methodology reference
- **Implementation:** Training script follows their hyperparameters
- **Boundary:** No direct code integration, methodology adaptation only

**Vietnamese Market Data Integration:**
- **Point:** Local CSV files with OHLCV data
- **Implementation:** `data_processing.py` reads from `data/raw/prices/`
- **Boundary:** File I/O operations, no external APIs

## File Organization Patterns

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

**Asset Organization:**
- **Location:** `data/`, `models/`, `experiments/` directories
- **Structure:** Functional subdirectories (raw/, processed/, checkpoints/)
- **Pattern:** Organized by data/function, not by component or layer

## Development Workflow Integration

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

**File Modifications During Development:**
1. **Data Processing:** Modify `src/data_processing.py` for new features
2. **Training Parameters:** Edit `configs/config.yaml`
3. **New Metrics:** Add functions to `src/model_evaluation.py`
4. **Results Analysis:** Create notebooks in `notebooks/` directory

## Architecture Boundaries Summary

**Key Integration Boundaries:**
- **Data Layer:** File-based CSV I/O (raw → processed)
- **Model Layer:** PyTorch checkpoint files (training → evaluation)
- **Metrics Layer:** JSON experiment logs (evaluation → analysis)
- **Inference Layer:** Dictionary-based predictions (model → output)

**Component Communication Rules:**
- **All scripts** read from `configs/config.yaml`
- **Data flows** through simple dictionaries and files
- **No complex APIs** - direct function calls and file I/O
- **Error handling** through exceptions and logging to files

**Enforcement Guidelines:**
- **All scripts** use standard metric functions from `model_evaluation.py`
- **All experiments** logged to `experiments/` as JSON
- **All data** flows through `data_processing.py` for consistency
- **All models** saved to `models/` with descriptive checkpoint names

---

**Status:** Complete project structure defined for rapid experimentation workflow