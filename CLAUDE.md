# AI Coding Guidelines for TimesFM VN30 Data Science Project

## 1. Nguyên tắc cơ bản (Foundational Rules)

- **Ngôn ngữ & Chuẩn mực:** Sử dụng Python 3.10+ với strict adherence to PEP 8.
- **Tính trọn vẹn:** Code sinh ra phải hoàn chỉnh và có thể chạy được (no placeholders like `# TODO: implement this`).
- **Typing:** Bắt buộc sử dụng Type Hints (`typing`) cho mọi function và method.
- **Docstrings:** Sử dụng Google Style Docstring để giải thích input, output và ý nghĩa của hàm.

## 2. Common ML/DS Research Rules (Clean Code Principles)

**These are universal clean code rules for machine learning and data science research projects. They apply to ALL ML/DS work, not just this project.**

### 2.1. Core Principles (The "Why")

- **Code is read much more than written:** Most software cost is in maintenance, not initial development
- **Leave code better than you found it:** The Boy Scouts rule - always improve code when you touch it
- **Keep it simple:** Use half the brain cells when writing code vs reading code
- **Match code quality to maturity:** Don't over-engineer quick POCs, don't leave production code messy

### 2.2. Naming Conventions (The "What")

**Variable Names:**
- **Descriptive over concise:** `learning_rate` not `lr`, `train_set` not `df`
- **Pronounceable and searchable:** `is_valid` not `flag`, `user_count` not `uc`
- **Plurals for collections:** `students` not `student_list`, `predictions` not `pred_list`
- **No abbreviations:** `classification_threshold` not `clf_thresh`
- **Better long than ambiguous:** `volatility_forecasting_model` not `model`

**Function Names:**
- **Verb or verb phrase:** `evaluate_model`, `load_data`, `calculate_metrics`
- **Specific and intention-revealing:** `load_training_data` not `load`, `train_timesfm_model` not `train`
- **One concept per name:** Don't use `get` for different purposes (fetching vs computing)
- **Private functions prefix with `_`:` `_internal_helper_function`

**Class Names:**
- **Noun or noun phrase:** `Classifier`, `DataHandler`, `ModelTrainer`
- **Avoid generic names:** Don't use `Manager`, `Processor`, `Handler`

**Constants:**
- **ALL_CAPS with underscores:** `MAX_EPOCHS = 100`, `LEARNING_RATE = 0.001`
- **Descriptive names:** `CLASSIFICATION_THRESHOLD = 0.8` not `THRESH = 0.8`
- **Place at top of file or in config:** Never hard-code magic numbers

### 2.3. Function Design (The "How")

**Size and Scope:**
- **Small functions:** No longer than 20-30 lines
- **One thing only:** Each function should do ONE thing well
- **Either do or answer:** Don't calculate AND print (no side effects)
- **Extract when complex:** If you search for lines within a function, extract them

**Parameters:**
- **Few arguments:** Avoid more than 3 parameters when possible
- **Use data structures:** Group related params into dicts/configs
- **Named parameters:** Always use parameter names when calling functions

**Structure:**
- **No side effects:** Don't modify unexpected data
- **Return early:** Avoid deep nesting with early returns
- **Don't repeat yourself:** Extract repeated code into functions
- **Encapsulate conditionals:** Use functions for complex boolean logic

### 2.4. Code Organization (The "Where")

**File Structure:**
- **One concept per file:** Don't mix data processing, model training, and evaluation
- **Related code close together:** Vertical proximity matters
- **High-level at top:** Main functions first, details below (newspaper principle)
- **Logical grouping:** Group imports, constants, functions, classes

**Script Layout (Newspaper Principle):**
```python
# 1. Imports (grouped: stdlib, third-party, local)
# 2. Constants (ALL_CAPS)
# 3. High-level functions (main orchestration)
# 4. Mid-level functions (specific logic)
# 5. Low-level functions (helpers, utilities)
# 6. if __name__ == "__main__": entry point
```

**Module Organization:**
- **Flat over nested:** Prefer simple function organization over complex class hierarchies
- **Functional over OOP:** Use functions when possible, classes only when beneficial
- **Separate concerns:** Data processing, model training, evaluation in separate files

### 2.5. Comments and Documentation (The "When")

**Comment Philosophy:**
- **Code should explain itself:** 90% of the time, use good names instead of comments
- **Explain WHY not HOW:** Comments should explain reasoning, not mechanics
- **Keep comments short:** One or two sentences max
- **Delete outdated comments:** Never leave comments that don't match the code
- **Don't comment out code:** Just delete it, git has your back

**When to Comment:**
- **Why decisions:** "Using SGD because financial time series benefit from momentum"
- **Non-obvious logic:** "Clip to [-5, 5] to prevent numerical instability"
- **Workarounds:** "Temporary fix until GPU memory is upgraded"
- **Financial/statistical context:** "20-day window = one trading month in Vietnam"

**Code Quality Levels:**
1. **Total mess:** Quick POC, testing ideas (acceptable for exploration)
2. **Readable:** Others can understand (minimum for sharing)
3. **Production:** Clean, tested, documented (required for deployment)

**Refactoring Rules:**
- **Never leave messy code:** Refactor before moving on
- **Small improvements:** Better code happens gradually
- **Tests first:** Write tests before refactoring when possible
- **Boy scouts rule:** Always leave code better than you found it

---

## 3. Tư duy Data Science - Financial ML Enhanced

### 2.1. Khả năng tái lập (Reproducibility)
- **Random Seeds:** Bắt buộc set random seeds ở đầu mỗi script:
```python
import random
import numpy as np
import torch

def set_random_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed(seed))
        torch.cuda.manual_seed_all(seed(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Gọi ở đầu mỗi training script
set_random_seed(config.get('random_seed', 42))
```

### 2.2. Xử lý Dữ liệu Financial
- **Vectorized Operations:** Ưu tiên operations với `pandas` và `numpy` thay vì vòng lặp `for`
- **Financial Transformations:** LUÔN LUÔN sử dụng log transformation cho stock data:
```python
# ✅ CORRECT - Financial standard
df['log_close'] = np.log(df['close'])
df['log_returns'] = df['log_close'].diff()

# ❌ WRONG - Sẽ gặp vấn đề với extreme events
df['returns'] = df['close'].pct_change()
```

- **Data Leakage Prevention:** Tách Train/Test trước khi apply bất kỳ scaling/imputation:
```python
# ✅ CORRECT - No data leakage
train_data, test_data = temporal_split(data, test_size=0.2)
train_mean = train_data.mean()
test_data_scaled = (test_data - train_mean) / train_std()

# ❌ WRONG - Data leakage
global_mean = data.mean()
data_scaled = (data - global_mean) / global_std()
```

### 2.3. Logging & Metadata
- **No Print Statements:** Không dùng `print()` trong production code, sử dụng `logging`:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments/training.log'),
        logging.StreamHandler()
    ]
)

# Log với financial metrics
logging.info(f"Epoch {epoch}: R²={r2:.4f}, QLIKE={qlike:.6f}, RMSE={rmse:.6f}")
```

- **Experiment Metadata:** Luôn log đầy đủ thông tin:
```python
experiment_log = {
    'timestamp': datetime.now().isoformat(),
    'model': 'timesfm-2.5-200m',
    'config': {
        'lora_r': 4,
        'lora_alpha': 8,
        'learning_rate': 1e-4,
        'optimizer': 'SGD'
    },
    'metrics': {
        'qlike': qlike_score,
        'r2': r2_score,
        'rmse': rmse_score,
        'mse': mse_score
    }
}
```

## 3. Kiến trúc TimesFM-Specific

### 3.1. Model Architecture Patterns
- **Sử dụng TimesFM 2.5 Foundation Model:** KHÔNG tự viết transformer architecture:
```python
# ✅ CORRECT - Actual TimesFM 2.5
from transformers import TimesFm2_5ModelForPrediction
from peft import LoraConfig, get_peft_model

model = TimesFm2_5ModelForPrediction.from_pretrained(
    "google/timesfm-2.5-200m-transformers",
    torch_dtype=torch.bfloat16
)

# ❌ WRONG - Custom transformer (not TimesFM!)
class TimesFMTransformer(nn.Module):
    def __init__(self):
        # Custom transformer architecture
```

- **LoRA Adapters Configuration:** Sử dụng exactly cấu trúc từ research:
```python
lora_config = LoraConfig(
    r=4,                    # Rank như đã xác định
    lora_alpha=8,           # Alpha như đã xác định
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(base_model, lora_config)
```

### 3.2. Financial-Specific Training
- **Optimizer:** LUÔN LUÔN dùng SGD cho financial data:
```python
# ✅ CORRECT - Financial standard
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-4,        # Conservative learning rate
    momentum=0.9,   # High momentum for stability
    nesterov=True   # Nesterov momentum
)

# ❌ WRONG - AdamW không optimal cho financial time series
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

- **Gradient Clipping:** Bắt buộc apply gradient clipping:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

- **Learning Rate Schedule:** Cosine annealing cho financial data:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,         # Full training cycle
    eta_min=1e-6       # Minimum learning rate
)
```

## 4. Vietnamese Market Specifics

### 4.1. Data Features for Vietnamese Stocks
- **TET Holiday Detection:**
```python
def is_tet_period(date_index):
    """Detect Vietnamese TET holiday period (Jan-Feb)"""
    return date_index.month in [1, 2]  # January, February
```

- **Day of Week Patterns:**
```python
data['day_of_week'] = data.index.dayofweek  # 0=Monday, 4=Friday
data['is_monday'] = (data['day_of_week'] == 0).astype(int)
data['is_friday'] = (data['day_of_week'] == 4).astype(int)
```

- **Price Range Normalization:**
```python
# Xử lý different price ranges (VCB ~90K vs HPG ~20K)
def normalize_for_vietnamese_stocks(stock_data_dict):
    """Normalize each stock independently"""
    normalized = {}
    for stock, data in stock_data_dict.items():
        mean = data.mean()
        std = data.std()
        normalized[stock] = (data - mean) / std
    return normalized
```

### 4.2. Financial Data Processing Pipeline
- **Pipeline Order:** Tuân thủ đúng thứ tự:
```python
# 1. Log transformation (ngăn extreme events)
df['log_close'] = np.log(df['close'])

# 2. Log returns (stable hơn raw returns)
df['log_returns'] = df['log_close'].diff()

# 3. Realized volatility
df['RV_20'] = df['log_returns'].rolling(20).std()

# 4. Vietnamese features
df['is_tet_period'] = is_tet_period(df.index)

# 5. Financial clipping (ngăn extreme values)
df['RV_20'] = np.clip(df['RV_20'], -5, 5)
```

## 5. Metrics & Validation

### 5.1. Financial Metrics - REQUIRED FUNCTIONS
- **Standard Metric Functions:** LUÔN LUÔN dùng exact function names:
```python
def calculate_qlike(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """Calculate QLIKE metric for volatility forecasting"""
    return np.sum(actuals/predictions + np.log(predictions) - 1)

def calculate_r2(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """Calculate R² score"""
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    return 1 - (ss_res / ss_tot)

def calculate_rmse(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """Calculate Root Mean Square Error"""
    return np.sqrt(np.mean((actuals - predictions) ** 2))

def calculate_mse(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """Calculate Mean Square Error"""
    return np.mean((actuals - predictions) ** 2)
```

- **Diebold-Mariano Statistical Test:**
```python
from scipy import stats

def diebold_mariano_test(actual: np.ndarray, model_pred: np.ndarray, 
                        bench_pred: np.ndarray) -> dict:
    """
    Diebold-Mariano test for forecast accuracy
    
    Returns:
        dict with 'dm_statistic', 'p_value', 'significant'
    """
    # Loss differential
    loss_diff = (actual - bench_pred)**2 - (actual - model_pred)**2
    
    # Test statistic
    mean_loss_diff = np.mean(loss_diff)
    var_loss_diff = np.var(loss_diff, ddof=1)
    
    dm_statistic = mean_loss_diff / np.sqrt(var_loss_diff / len(loss_diff))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_statistic)))
    
    return {
        'dm_statistic': dm_statistic,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

### 5.2. Evaluation Standards
- **Batch Evaluation:** Evaluate all metrics together, không từng metric riêng lẻ:
```python
# ✅ CORRECT - Comprehensive evaluation
metrics = {
    'qlike': calculate_qlike(actuals, predictions),
    'r2': calculate_r2(actuals, predictions),
    'rmse': calculate_rmse(actuals, predictions),
    'mse': calculate_mse(actuals, predictions)
}

# ❌ WRONG - Incomplete evaluation
print(f"R2: {r2}")  # Missing other metrics
```

## 6. Code Organization & Structure

### 6.1. Simple Functional Approach
- **Functional over OOP:** Ưu tiên simple functions thay complex classes:
```python
# ✅ RECOMMENDED - Simple functions
def process_vn30_data(stock_file: str) -> pd.DataFrame:
    """Process single Vietnamese stock data"""
    df = pd.read_csv(stock_file)
    df['log_close'] = np.log(df['close'])
    return df

# ❌ AVOID - Over-engineered classes
class VN30DataProcessorFactory:
    def create_processor(self, type): ...
```

### 6.2. File Naming & Organization
- **Script Naming:** Use `snake_case.py` for all scripts:
```
src/
├── data_processing.py      # NOT DataProcessing.py
├── model_training.py         # NOT model_training.py  
├── model_evaluation.py       # NOT ModelEvaluation.py
├── statistical_tests.py       # NOT StatisticalTests.py
└── inference.py               # NOT Inference.py
```

- **Checkpoint Naming:** Descriptive checkpoint names:
```python
# ✅ CORRECT - Descriptive naming
torch.save(model, f"models/checkpoints/model_epoch_{epoch}_r2_{r2:.3f}.pt")

# ❌ WRONG - Non-descriptive naming
torch.save(model, f"models/checkpoint_{epoch}.pt")
```

### 6.3. Configuration Management
- **Centralized Configuration:** Sử dụng `configs/config.yaml` duy nhất:
```python
# Load configuration
import yaml
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Use in code
stocks = config['data']['stocks']
learning_rate = config['incremental_learning']['training']['learning_rate']
```

- **NO Hard-coded Parameters:** Không hard-code values:
```python
# ❌ WRONG - Hard-coded
learning_rate = 0.0001
batch_size = 32

# ✅ CORRECT - From config
learning_rate = config['training']['learning_rate']
batch_size = config['training']['batch_size']
```

## 7. Training & Experimentation

### 7.1 Training Loop Patterns
- **Progress Tracking:** Log metrics at mỗi epoch với đầy đủ thông tin:
```python
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    eval_metrics = evaluate_model(model, test_loader)
    
    # Log comprehensive metrics
    logging.info(f"Epoch {epoch+1}/{num_epochs}")
    logging.info(f"  Train Loss: {train_loss:.6f}")
    logging.info(f"  R²: {eval_metrics['r2']:.4f}")
    logging.info(f"  QLIKE: {eval_metrics['qlike']:.6f}")
    logging.info(f"  RMSE: {eval_metrics['rmse']:.6f}")
    logging.info(f"  MSE: {eval_metrics['mse']:.6f}")
```

### 7.2. Checkpoint Management
- **Save Best Models:** Chỉ save khi cải thiện:
```python
best_r2 = 0.0
for epoch in range(num_epochs):
    # Training and evaluation...
    current_r2 = evaluate_r2(model, test_data)
    
    if current_r2 > best_r2:
        best_r2 = current_r2
        save_checkpoint(model, f"models/best_model_r2_{current_r2:.3f}.pt")
        logging.info(f"  New best model saved! R² = {current_r2:.4f}")
```

### 7.3. Experiment Tracking
- **JSON Logging:** Save experiment metadata as JSON:
```python
import json

def log_experiment(epoch, metrics, config):
    """Log experiment metadata to JSON"""
    experiment_log = {
        'timestamp': datetime.now().isoformat(),
        'epoch': epoch,
        'config': config,
        'metrics': metrics
    }
    
    with open(f'experiments/experiment_epoch_{epoch}.json', 'w') as f:
        json.dump(experiment_log, f, indent=2)
```

### 7.4. File Management and Archiving
- **Archive Old Files:** Khi fix hoặc tạo file mới, bắt buộc move file cũ vào `archived/` folder:
```python
# ✅ CORRECT - Archive old files
# Cấu trúc thư mục:
project/
├── src/                    # Code hiện hành (active code)
│   ├── model_training.py
│   ├── data_processing.py
│   └── model_evaluation.py
└── archived/               # Code cũ (reference only)
    ├── train_old_2025-06-13_refactored.py
    ├── processing_v1.py
    └── evaluate_old_2025-06-13_bugfix.py

# Quy tắc đặt tên archived files:
# filename_old_DATE_REASON.py
# Ví dụ: model_training_old_2025-06-13_refactored.py
# Ví dụ: data_processing_old_2025-06-13_bugfix_v1.py

# Khi fix file:
# 1. Move file cũ sang archived/
mv src/train.py archived/train_old_2025-06-13_refactored.py

# 2. Tạo file mới với tên improved
# src/train.py → src/model_training.py (better name)
```

- **Tại sao phải archive:**
  - Tránh clutter trong thư mục src/
  - Dễ biết file nào đang được sử dụng
  - Giữ reference cho việc debug và học hỏi
  - Track evolution của codebase

### 7.5. Learning Curves (MANDATORY)
- **BẮT BUỘC Vẽ Learning Curves:** Mọi lần training đều phải vẽ learning curves để detect overfitting:
```python
import matplotlib.pyplot as plt
from pathlib import Path

def plot_learning_curves(train_losses, val_losses, save_path, epoch=None):
    """
    Vẽ learning curves để detect overfitting.
    
    MANDATORY cho tất cả training runs:
    - Training loss vs validation loss
    - Detect overfitting: val loss tăng trong khi train loss giảm
    - Detect underfitting: cả train và val loss đều cao
    - Visual convergence behavior
    
    Args:
        train_losses: List training losses per epoch
        val_losses: List validation losses per epoch  
        save_path: Path để save plot
        epoch: Current epoch number (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Learning Curves - TimesFM Training', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Check for overfitting
    if len(val_losses) > 10 and epoch is not None:
        recent_val = val_losses[-5:]
        if all(recent_val[i] > recent_val[i-1] for i in range(1, len(recent_val))):
            logging.warning(f"⚠️ Epoch {epoch}: Validation loss increasing - possible OVERFITTING!")
            logging.warning(f"   Recent val losses: {recent_val}")
    
    if epoch is not None:
        logging.info(f"💾 Learning curve saved: {save_path}")

# During training
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_metrics = evaluate_model(model, val_loader)
    val_loss = val_metrics['val_loss']
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # Log metrics
    logging.info(f"Epoch {epoch+1}/{num_epochs}")
    logging.info(f"  Train Loss: {train_loss:.6f}")
    logging.info(f"  Val Loss: {val_loss:.6f}")
    
    # Vẽ learning curves mỗi 10 epochs
    if (epoch + 1) % 10 == 0:
        plot_path = f'experiments/learning_curves_epoch_{epoch+1}.png'
        plot_learning_curves(train_losses, val_losses, plot_path, epoch+1)
    
    # Early stopping nếu overfitting
    if len(val_losses) > 10:
        recent_val_trend = val_losses[-5:]
        is_increasing = all(recent_val_trend[i] > recent_val_trend[i-1] 
                           for i in range(1, len(recent_val_trend)))
        if is_increasing:
            logging.warning(f"🚨 Overfitting detected at epoch {epoch+1}")
            logging.warning(f"   Applying early stopping...")
            break

# Final learning curves (MANDATORY)
final_plot_path = 'experiments/learning_curves_final.png'
plot_learning_curves(train_losses, val_losses, final_plot_path)
logging.info(f"✅ Final learning curves saved: {final_plot_path}")

# Lưu learning curves vào experiment metadata
experiment_log['learning_curves_path'] = final_plot_path
```

- **Khi nào overfitting xảy ra:**
  - Validation loss tăng trong khi training loss giảm
  - Gap lớn giữa training và validation loss
  - Training accuracy ~100% nhưng validation thấp hơn nhiều

- **Giải pháp overfitting:**
  1. Thêm regularization (dropout, weight decay)
  2. Giảm model complexity
  3. Tăng training data
  4. Data augmentation
  5. Early stopping dựa trên validation loss

- **Rules:**
  - MANDATORY: Vẽ learning curves cho TẤT CẢ training runs
  - Plot mỗi N epochs (ví dụ: mỗi 10 epochs)
  - Luôn save final learning curve
  - Include learning curves trong experiment reports
  - Sử dụng learning curves để detect overfitting sớm

## 8. GPU Training Considerations

### 8.1. GPU Memory Management
- **Memory Validation:** Check GPU memory trước training:
```python
def validate_gpu_memory():
    """Check GPU memory availability"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory < 8:  # Minimum 8GB requirement
            raise ValueError(f"Insufficient GPU memory: {gpu_memory:.1f}GB < 8GB")
        logging.info(f"GPU Memory: {gpu_memory:.1f}GB - Sufficient")
```

- **Batch Size Adjustment:** Adjust batch size nếu cần thiết:
```python
# Start with config value, reduce if OOM
batch_size = config['training']['batch_size']
try:
    # Training code
    pass
except torch.cuda.OutOfMemoryError:
    logging.warning(f"OOM at batch_size={batch_size}, reducing by half")
    batch_size = batch_size // 2
    # Retry with smaller batch size
```

### 8.2. Mixed Precision Training
- **Use bfloat16:** Sử dụng bfloat16 để tiết kiệm memory:
```python
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

model = TimesFm2_5ModelForPrediction.from_pretrained(
    "google/timesfm-2.5-200m-transformers",
    torch_dtype=dtype
)
```

## 9. Error Handling & Validation

### 9.1. Pre-flight Validation
- **Validate Inputs:** Kiểm tra data quality trước expensive operations:
```python
def validate_financial_data(data: pd.DataFrame) -> bool:
    """Validate financial data quality"""
    if data.isnull().any().any():
        raise ValueError("Data contains NaN values")
    if len(data) < 100:
        raise ValueError(f"Insufficient data: {len(data)} < 100 observations")
    if (data <= 0).any().any():
        raise ValueError("Data contains non-positive values (can't log transform)")
    return True
```

### 9.2. Training Error Handling
- **Graceful Degradation:** Xử lý errors một cách graceful:
```python
try:
    validate_gpu_memory()
    validate_financial_data(train_data)
    train_model(model, train_data)
    
except ValueError as e:
    logging.error(f"Validation failed: {e}")
    # Exit gracefully, save partial results
    
except torch.cuda.OutOfMemoryError:
    logging.error("GPU OOM - try smaller batch size")
    # Fallback strategy
    
except Exception as e:
    logging.error(f"Unexpected error: {e}")
    # Save current state before exiting
```

## 10. Documentation & Comments

### 10.1. Financial ML Docstrings
- **Explain Financial Logic:** Docstrings phải giải thích ý nghĩa tài chính:
```python
def calculate_realized_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate realized volatility using rolling standard deviation.
    
    In financial ML, realized volatility is a key measure of asset price variability.
    For Vietnamese stocks, we use 20-day rolling window as market standard.
    
    Args:
        returns: Log returns of stock prices (NOT raw returns)
        window: Rolling window size (default: 20 trading days)
        
    Returns:
        pd.Series: Realized volatility time series
        
    Financial Note:
        - Using log returns instead of raw returns for better statistical properties
        - 20-day window represents approximately one trading month in Vietnam
        - This metric is target variable for TimesFM fine-tuning
        
    Example:
        >>> log_returns = np.log(prices).diff()
        >>> rv = calculate_realized_volatility(log_returns)
        >>> print(f"Mean RV: {rv.mean():.6f}")
    """
```

### 10.2. Comments for Complex Logic
- **Explain Counter-intuitive Financial Logic:**
```python
# Financial Note: We use log returns instead of raw returns because:
# 1. Log returns are more normally distributed
# 2. They handle extreme market events better (flash crashes)
# 3. They allow us to apply log transformation which prevents NaN during training
log_returns = np.log(prices).diff()

# Statistical Note: Clipping to [-5, 5] range prevents extreme values
# from causing numerical instability during training while preserving
# most of the information content of volatility measurements
rv_clipped = np.clip(realized_volatility, -5, 5)
```

## 11. Testing & Quality Assurance

### 11.1. Data Quality Tests
```python
def test_financial_data_quality(data: pd.DataFrame):
    """Test data quality for financial ML"""
    # Test no NaN after processing
    assert not data.isnull().any().any(), "Data contains NaN values"
    
    # Test reasonable volatility ranges
    assert (data['RV_20'] > 0).all(), "Volatility must be positive"
    assert (data['RV_20'] <= 5).all(), "Volatility clipping failed"
    
    # Test log transformation
    assert 'log_close' in data.columns, "Log transformation missing"
    assert 'log_returns' in data.columns, "Log returns missing"
```

### 11.2. Metric Calculation Tests
```python
def test_metric_functions():
    """Test that metric calculations are correct"""
    # Create sample data
    actuals = np.array([0.01, 0.02, 0.015])
    predictions = np.array([0.011, 0.019, 0.016])
    
    # Test R² calculation
    r2 = calculate_r2(actuals, predictions)
    assert 0 <= r2 <= 1, "R² must be between 0 and 1"
    
    # Test MSE calculation
    mse = calculate_mse(actuals, predictions)
    assert mse >= 0, "MSE must be non-negative"
```

## 12. Anti-Patterns to Avoid

### 12.1. Architecture Anti-Patterns
- **❌ Custom Transformers:** Đừng viết TimesFM transformer từ đầu
- **❌ AdamW Optimizer:** Đừng dùng AdamW cho financial time series
- **❌ Raw Returns:** Đừng dùng raw returns (pct_change) cho financial ML
- **❌ Single Dataset:** Đừng gộp tất cả stocks thành một time series

### 12.2. Data Processing Anti-Patterns
- **❌ Global Scaling:** Đừng scale toàn bộ dataset trước train/test split
- **❌ Skip Log Transformation:** Đừng bỏ qua log transformation cho financial data
- **❌ Ignore Financial Clipping:** Đừng skip financial clipping (-5,5 range)

### 12.3. Training Anti-Patterns
- **❌ Print Statements:** Đừng dùng print() cho experiment tracking
- **❌ Inconsistent Metrics:** Đừng dùng different metric names
- **❌ Missing Seeds:** Đừng quên random seeds cho reproducibility
- **❌ Hard-coded Parameters:** Đừng hard-code learning rates, batch sizes

## 13. Google Colab Notebook Best Practices

### 13.1. CRITICAL: Package Name Rules

**MANDATORY Package Names:**
```python
# ✅ CORRECT - Use exact package names
!pip install pyyaml      # NOT yaml
!pip install accelerate  # NOT accel
!pip install transformers # NOT transformer

# ❌ WRONG - Common mistakes
!pip install yaml         # Package doesn't exist
!pip install transformer  # Wrong name
```

**Why This Matters:**
- `yaml` package doesn't exist → `ERROR: No matching distribution found for yaml`
- Correct name is `pyyaml` → Always use exact package names
- This is especially important in Colab where error messages can be confusing

### 13.2. CRITICAL: Torchao Version Conflict

**MANDATORY: Uninstall torchao before training:**
```python
# ✅ CORRECT - Always uninstall torchao first
!pip uninstall -y torchao

# Setup memory after uninstall
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ❌ WRONG - Skip this step
# Error: Found an incompatible version of torchao. Found version 0.10.0, 
#        but only versions above 0.16.0 are supported
# LoRA setup will FAIL
```

**Why This Matters:**
- Colab installs torchao by default
- TimesFM/PEFT requires torchao > 0.16.0, but Colab has 0.10.0
- Version conflict causes LoRA adapter setup to fail
- Must uninstall before any model training

### 13.3. CRITICAL: Directory Creation

**MANDATORY: Create directories before training:**
```python
# ✅ CORRECT - Create all necessary directories
!mkdir -p experiments models

# ❌ WRONG - Assume directories exist
# Training will FAIL if directories don't exist
# Model checkpoints can't be saved
```

**Why This Matters:**
- Training scripts assume `experiments/` and `models/` directories exist
- Missing directories cause file write failures during training
- Use `-p` flag to create parent directories if needed

### 13.4. CRITICAL: Persistent Directory Change

**MANDATORY: Use os.chdir() for persistence:**
```python
# ✅ CORRECT - Use os.chdir() for persistent directory change
import os
os.chdir('stockvoli-research')

# ❌ WRONG - %cd doesn't always persist
%cd stockvoli-research  # May not persist across cells
```

**Why This Matters:**
- `%cd` is a Colab magic command that may not persist
- `os.chdir()` is proper Python and persists across all cells
- Training scripts need to be in correct directory to find data files
- Prevents "file not found" errors during training

### 13.5. MANDATORY: Memory Optimization Setup

**CRITICAL: Setup CUDA memory optimization:**
```python
# ✅ CORRECT - Setup memory optimization BEFORE any model operations
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ❌ WRONG - Skip this step
# May cause CUDA memory errors during training
# Especially important for TimesFM 2.5 with large models
```

**Why This Matters:**
- TimesFM 2.5 is a large model (232M parameters)
- Without memory optimization, may cause CUDA allocation failures
- Must be set BEFORE loading model or starting training
- Same setting used in working G4 notebook

### 13.6. Notebook Structure Best Practices

**MANDATORY: Complete notebook structure (13 cells minimum):**

1. **GPU Check** - Verify hardware availability
2. **Clone + Setup** - Repository + **mkdir + os.chdir (CRITICAL)**
3. **Dependencies** - **Torchao uninstall + memory setup (CRITICAL)**
4. **Import Verify** - Test all packages work
5. **Config Verify** - Load and display settings (CRITICAL)
6. **Data Verify** - Check OHLC features exist
7. **Run Test** - Execute training script
8. **Monitor Header** - Section divider (optional)
9. **Training Monitor** - Logs + GPU utilization (CRITICAL)
10. **Results Check** - Enhanced analysis (CRITICAL)
11. **Summary** - Expected outcomes
12. **Documentation** - Complete context
13. **Backup Commands** - Alternative execution method

**Why Each Cell Matters:**
- **Cell 2:** Directory setup prevents file not found errors
- **Cell 3:** Torchao fix prevents LoRA setup failures
- **Cell 5:** Config verification ensures correct parameters
- **Cell 9:** Monitoring allows tracking training progress
- **Cell 10:** Results checking validates experiment success

### 13.7. JSON Notebook Format Rules

**MANDATORY: Valid JSON structure:**
```python
# ✅ CORRECT - Valid JSON notebook
{
  "cells": [
    {
      "cell_type": "code",
      "source": ["print('hello')"]  # Array of strings
    }
  ],
  "metadata": {...},
  "nbformat": 4,
  "nbformat_minor": 4
}

# ❌ WRONG - Invalid JSON
{
  "cells": [
    {
      "source": "print('hello')"  # String, not array
    }
  ]
}

# ❌ WRONG - Unicode issues
# Avoid special Unicode characters in code cells
# Use ASCII-safe strings in markdown
```

**Why This Matters:**
- Colab parses notebooks as JSON
- Invalid JSON causes parsing errors
- "SyntaxError: Unexpected token" errors indicate JSON issues
- Always validate JSON before pushing to git

### 13.8. Dependency Installation Strategy

**PREFERRED: Use pre-installed Colab packages:**
```python
# ✅ RECOMMENDED - Disable install, use pre-installed
# Install dependencies (DISABLED - Already installed in Colab)
# !pip install -q transformers peft torch pandas numpy pyyaml accelerate

print("Using pre-installed Colab packages")
print("If you get import errors, uncomment pip install lines above")

# ❌ AVOID - Always reinstall
# Wastes time, may cause version conflicts
# Colab already has most packages installed
```

**When to Enable Installation:**
- Only if you get specific import errors
- If you need specific versions
- For testing new packages
- If Colab environment changes

### 13.9. Training Monitoring Requirements

**MANDATORY: Monitor training during execution:**
```python
# ✅ CORRECT - Monitor training progress
print("Training logs:")
!tail -20 experiments/model_training.log

print("GPU utilization:")
!nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv

import torch
allocated = torch.cuda.memory_allocated(0) / 1e9
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU Memory: {allocated:.1f}/{total:.1f} GB")
```

**Why This Matters:**
- Training takes hours (3-8 hours typical)
- Need to detect failures early
- GPU utilization indicates training health
- Memory usage helps prevent OOM errors
- Same monitoring as working G4 notebook

### 13.10. Results Validation Requirements

**MANDATORY: Comprehensive results checking:**
```python
# ✅ CORRECT - Complete results validation
results_file = Path('experiments/feature_comparison_results_fixed.json')

if results_file.exists():
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Check status
    if 'RV_20' in results and 'overnight' in results:
        rv20_result = results['RV_20']
        overnight_result = results['overnight']
        
        # Validate success
        if rv20_result['status'] == 'success' and overnight_result['status'] == 'success':
            # Calculate improvement
            improvement = (rv20_loss - overnight_loss) / rv20_loss * 100
            
            # Check success criteria
            if improvement > 0:
                print("SUCCESS: Overnight volatility is better!")
            else:
                print("Baseline still better")
```

**Why This Matters:**
- Training may complete but produce invalid results
- Need to validate both features succeeded
- Statistical significance matters
- Must check for error messages
- Same validation as G4 notebook

### 13.11. Common Colab Issues & Solutions

**ISSUE 1: "SyntaxError: Unexpected token"**
- **Cause:** Invalid JSON in notebook file
- **Solution:** Validate JSON with `python -c "import json; json.load(open('notebook.ipynb'))"`
- **Prevention:** Use Write tool correctly, don't edit JSON manually

**ISSUE 2: "No matching distribution found for yaml"**
- **Cause:** Wrong package name `yaml` instead of `pyyaml`
- **Solution:** Use `!pip install pyyaml`
- **Prevention:** Always use exact package names

**ISSUE 3: "torchao version incompatible"**
- **Cause:** torchao 0.10.0 conflicts with PEFT
- **Solution:** `!pip uninstall -y torchao` before training
- **Prevention:** Always include torchao uninstall cell

**ISSUE 4: "File not found" during training**
- **Cause:** Wrong directory or missing directories
- **Solution:** Use `os.chdir()` and `!mkdir -p experiments models`
- **Prevention:** Always create directories before training

**ISSUE 5: "CUDA out of memory"**
- **Cause:** Insufficient GPU memory for batch size
- **Solution:** Reduce batch size in config.yaml
- **Prevention:** Use G4-optimized settings (batch_size: 12)

### 13.12. Notebook Creation Checklist

**MANDATORY: Check all items before considering notebook complete:**

- [ ] Package names are correct (pyyaml, NOT yaml)
- [ ] Torchao uninstall included
- [ ] Memory optimization setup included
- [ ] Directory creation included
- [ ] Persistent directory change (os.chdir)
- [ ] Config verification included
- [ ] Data verification included
- [ ] Training monitoring included
- [ ] Results validation included
- [ ] JSON format validated
- [ ] All 13 cells present
- [ ] Backup commands provided
- [ ] Expected timeline documented

### 13.13. G4 Notebook Compatibility Rules

**MANDATORY: Match G4 notebook structure exactly:**

**G4 Notebook Elements to Include:**
1. ✅ `!pip uninstall -y torchao` (Cell 4)
2. ✅ `!mkdir -p experiments models` (Cell 6)
3. ✅ `os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'` (Cell 4)
4. ✅ `os.chdir('stockvoli-research')` (Cell 6)
5. ✅ Config load and verification (Cell 8)
6. ✅ Training logs monitoring (Cell 12)
7. ✅ GPU utilization checks (Cell 12)
8. ✅ Enhanced results validation (Cell 14)

**Why G4 Compatibility Matters:**
- G4 notebook is proven to work
- Contains solutions to all common issues
- Tested on actual hardware
- Optimized for memory and performance
- Reference for all future notebooks

### 13.14. Quick Reference - Common Commands

**Package Installation:**
```python
# CORRECT
!pip install pyyaml accelerate transformers peft torch pandas numpy

# WRONG
!pip install yaml  # Wrong name
```

**Torchao Fix:**
```python
# ALWAYS include this
!pip uninstall -y torchao
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

**Directory Setup:**
```python
# ALWAYS include this
!mkdir -p experiments models
import os
os.chdir('stockvoli-research')
```

**Config Verification:**
```python
# ALWAYS include this
import yaml
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(f"Batch size: {config['training']['batch_size']}")
```

**Monitoring:**
```python
# ALWAYS include this
!tail -20 experiments/model_training.log
!nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total
```

## 14. Quick Reference for TimesFM VN30 Project

### 14.1. Mandatory Function Names
```python
# Metrics (LUÔN LUÔN exact names này)
calculate_qlike(actuals, predictions)
calculate_r2(actuals, predictions)
calculate_rmse(actuals, predictions)
calculate_mse(actuals, predictions)

# Data Processing
apply_log_transformation(prices)
calculate_realized_volatility(returns, window=20)
is_tet_period(date_index)

# Model
load_timesfm_model()
setup_lora_adapters(model)
train_model(model, data_loader, optimizer)

# Validation
diebold_mariano_test(actual, model_pred, bench_pred)
```

### 14.2. File Structure Quick Reference
```
src/
├── data_processing.py      # Financial transformations
├── vn30_dataset.py        # Multi-stock dataset
├── model_training.py      # TimesFM training
├── model_evaluation.py   # Metrics (QLIKE, R², RMSE, MSE)
├── statistical_tests.py   # Diebold-Mariano
└── inference.py          # Simple predictions
```

### 14.3. Configuration Reference
```yaml
# From configs/config.yaml
data:
  stocks: [VCB, VIC, VNM, ...]  # 30 stocks
  raw_path: "data/raw/prices"

incremental_learning:
  window_size: 90
  training:
    batch_size: 32
    learning_rate: 0.0001  # 1e-4
    optimizer: "SGD"
    epochs_per_window: 1
```

---

## Final Notes

**Mission:** This project implements **proper TimesFM foundation model fine-tuning** for Vietnamese VN30 stocks using proven financial ML methodology.

**Key Success Metrics:** R² > 0.5, Diebold-Mariano p < 0.05, 25-35% loss improvement vs zero-shot.

**Philosophy:** Simple, reproducible, experiment-focused code that prioritizes financial ML rigor over infrastructure complexity.

**Remember:** 
- ✅ Use actual TimesFM 2.5 (NOT custom transformers)
- ✅ Log transformation is mandatory for financial data
- ✅ SGD optimizer (NOT AdamW) for financial time series
- ✅ All agents must use exact metric function names
- ✅ Vietnamese market features (TET holidays, trading patterns)
- ✅ Statistical validation (Diebold-Mariano) is required

**Architecture Reference:** See `_bmad-output/planning-artifacts/architecture.md` for complete technical decisions and implementation patterns.

---

*Last Updated: 2026-06-15*
*Project: TimesFM VN30 Financial Fine-tuning*
*Based on: Architecture Workflow Completion*
*Colab Notebook Best Practices Added: Section 13*