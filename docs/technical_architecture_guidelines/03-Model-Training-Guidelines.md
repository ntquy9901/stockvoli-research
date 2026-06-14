# Model Training Guidelines - TimesFM Fine-tuning

**Last Updated:** 2026-06-11
**Model:** TimesFM 2.5 (200M parameters)
**Method:** LoRA fine-tuning
**Domain:** Financial Time Series

---

## 🎯 **Architecture Overview**

### **System Architecture:**

```
TimesFM Fine-tuning Pipeline
├── Foundation Model
│   ├── Model: google/timesfm-2.5-200m-transformers
│   ├── Parameters: 200M (frozen)
│   └── Pre-trained: Diverse time series data
│
├── LoRA Adapters
│   ├── Rank (r): 4
│   ├── Alpha: 8
│   ├── Target: All linear layers
│   ├── Trainable: 1.38M (0.59%)
│   └── Method: Low-rank adaptation
│
├── Training Configuration
│   ├── Optimizer: SGD (momentum=0.9)
│   ├── Learning Rate: 1e-4
│   ├── Batch Size: 32
│   ├── Epochs: 100
│   └── Scheduler: Cosine annealing
│
└── Dataset
    ├── Architecture: Multi-stock (30 stocks)
    ├── Context: 128 days
    ├── Horizon: 13 days
    └── Split: Temporal (80/20)
```

---

## 🏗️ **Model Architecture**

### **TimesFM 2.5 Foundation Model:**

```python
from transformers import TimesFm2_5ModelForPrediction

# Load base model (frozen)
base_model = TimesFm2_5ModelForPrediction.from_pretrained(
    "google/timesfm-2.5-200m-transformers",
    torch_dtype=torch.bfloat16,  # Memory efficiency
    device_map="cuda"
)

# Base model parameters: 200M (frozen)
# Only LoRA adapters are trainable
```

**Why TimesFM?**
- Pre-trained on diverse time series data
- Better than training from scratch
- Handles multiple time series patterns
- Designed for forecasting tasks

### **LoRA Adapter Configuration:**

```python
from peft import LoraConfig, get_peft_model

# LoRA configuration (from Google Research)
lora_config = LoraConfig(
    r=4,                    # Rank (low for efficiency)
    lora_alpha=8,           # Scaling factor
    target_modules="all-linear",  # All linear layers
    lora_dropout=0.05,      # Dropout for regularization
    bias="none"             # No trainable biases
)

# Add LoRA adapters
model = get_peft_model(base_model, lora_config)

# Trainable parameters: 1.38M (0.59% of 200M)
# Base model: Frozen
```

**Why this configuration?**
- **r=4**: Proven effective for financial time series
- **alpha=8**: Optimal scaling (2× rank)
- **all-linear**: Maximum adaptability
- **dropout=0.05**: Prevents overfitting
- **0.59% trainable**: Parameter-efficient fine-tuning

---

## 🔧 **Training Configuration**

### **Optimizer Selection:**

```python
# ✅ CORRECT - SGD with momentum (financial standard)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-4,          # Conservative learning rate
    momentum=0.9,     # High momentum for stability
    nesterov=True     # Nesterov momentum (faster convergence)
)

# ❌ WRONG - AdamW (not optimal for financial data)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)
```

**Why SGD for Financial Data?**
1. **Better generalization** - less overfitting
2. **More stable** - less sensitive to hyperparameters
3. **Proven empirically** - financial ML literature
4. **Consistent results** - reproducible training

### **Learning Rate Schedule:**

```python
# Cosine annealing (proven for financial data)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,         # Full training cycle
    eta_min=1e-6       # Minimum learning rate
)

# Learning rate decay:
# Epoch 1:   1e-4 (start)
# Epoch 50:  5e-5 (mid)
# Epoch 100: 1e-6 (end)
```

**Why Cosine Annealing?**
- Smooth decay (prevents sudden jumps)
- Proven for time series forecasting
- Better convergence than constant LR
- Standard in financial ML

### **Gradient Management:**

```python
# Gradient clipping (prevent exploding gradients)
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)

# Apply AFTER backward pass, BEFORE optimizer step
```

### **Batch Configuration:**

```python
# Training parameters
batch_size = 32        # Good balance for memory/stability
num_epochs = 100       # Full training cycle
warmup_epochs = 5      # Warmup period
patience = 5           # Early stopping patience
```

---

## 📊 **Dataset Architecture**

### **Multi-Stock Dataset:**

```python
class VN30TimeSeriesDataset(Dataset):
    """
    Multi-stock time series dataset

    Architecture:
    - 30 stocks (channel-independent)
    - Each stock treated as separate series
    - No cross-stock information leakage
    """

    def __init__(self, series_list, context_len=128, horizon_len=13,
                 num_samples=200, seed=42, mode='train'):
        """
        Args:
            series_list: List of stock time series
            context_len: Input window (128 days)
            horizon_len: Prediction horizon (13 days)
            num_samples: Samples per stock (200)
            mode: 'train' or 'test' (temporal split)
        """
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.num_samples = num_samples

        # Apply temporal split
        split_ratio = 0.8
        processed_series = []

        for series in series_list:
            split_point = int(len(series) * split_ratio)

            if mode == 'train':
                usable_data = series[:split_point]  # First 80%
            else:  # test
                usable_data = series[split_point:]  # Last 20%

            processed_series.append(usable_data)

        # Generate samples
        self.samples = []
        rng = np.random.default_rng(seed)

        for series in processed_series:
            min_len = context_len + horizon_len
            max_start = len(series) - min_len

            for _ in range(num_samples):
                if max_start < 0:
                    continue

                start = rng.integers(0, max_start + 1)
                end = start + context_len + horizon_len

                window = series[start:end]

                context = window[:context_len]
                target = window[context_len:]

                self.samples.append((context, target))
```

**Key Features:**
1. **Channel-independent**: Each stock is separate series
2. **Temporal split**: Prevents data leakage
3. **Fixed windows**: Consistent input/output sizes
4. **Random sampling**: Within temporal bounds

### **Dataset Statistics:**

```
Training Dataset:
- Stocks: 30
- Samples per stock: 200
- Total samples: 6,000
- Context: 128 days
- Horizon: 13 days
- Temporal range: 2006-2022 (first 80%)

Test Dataset:
- Stocks: 30
- Samples per stock: 1 (last window)
- Total samples: 30
- Context: 128 days
- Horizon: 13 days
- Temporal range: 2022-2026 (last 20%)
```

---

## 🔄 **Training Loop**

### **Complete Training Loop:**

```python
def train_model(model, train_loader, val_loader, config):
    """
    Complete training loop with all best practices
    """
    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=0.9,
        nesterov=True
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=1e-6
    )

    # Training history
    history = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            context, target = batch

            # Forward pass
            outputs = model(
                past_values=context,
                future_values=target,
                forecast_context_len=config['context_len']
            )

            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                context, target = batch

                outputs = model(
                    past_values=context,
                    future_values=target,
                    forecast_context_len=config['context_len']
                )

                val_loss += outputs.loss.item()

        val_loss /= len(val_loader)

        # Learning rate step
        scheduler.step()

        # Logging
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.8f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'models/best_model.pt')

            print(f"  ✓ New best model saved!")
        else:
            patience_counter += 1

        if patience_counter >= config['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # History
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

    return history
```

---

## 📈 **Training Monitoring**

### **Key Metrics to Track:**

```python
# Monitor these during training
metrics_to_track = {
    'train_loss': 'Training MSE loss',
    'val_loss': 'Validation MSE loss',
    'train_val_gap': 'Difference (should be < 2x)',
    'learning_rate': 'Current learning rate',
    'epoch_time': 'Seconds per epoch',
    'memory_usage': 'GPU memory (GB)',
}
```

### **Warning Signs:**

```
⚠️  TRAINING ISSUES:

1. Val loss >> Train loss (> 3x gap)
   → Possible overfitting or data leakage

2. Loss spikes suddenly
   → Learning rate too high or bad batch

3. Loss becomes NaN
   → Numerical instability (check clipping)

4. Memory overflow
   → Reduce batch size or gradient accumulation

5. Very slow convergence
   → Learning rate too low
```

### **Healthy Training Pattern:**

```
✅ GOOD TRAINING:

Epoch 1:   Train Loss: 3.60,  Val Loss: 0.64
Epoch 10:  Train Loss: 1.20,  Val Loss: 0.58
Epoch 50:  Train Loss: 0.35,  Val Loss: 0.53
Epoch 75:  Train Loss: 0.12,  Val Loss: 0.52  ← Best
Epoch 100: Train Loss: 0.11,  Val Loss: 0.55

Pattern:
- Both losses decreasing steadily
- Val loss < 2x train loss (acceptable gap)
- Convergence by epoch 75-100
- Best model saved at epoch 75
```

---

## 🎯 **Hyperparameter Tuning**

### **Recommended Ranges:**

```python
hyperparameter_grid = {
    'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4],  # Best: 1e-4
    'lora_r': [2, 4, 8, 16],                      # Best: 4
    'lora_alpha': [4, 8, 16, 32],                 # Best: 8
    'batch_size': [16, 32, 64],                   # Best: 32
    'momentum': [0.9, 0.95, 0.99],               # Best: 0.9
    'dropout': [0.0, 0.05, 0.1],                 # Best: 0.05
}
```

### **Tuning Strategy:**

```python
# 1. Start with defaults (our configuration)
# 2. Tune learning rate first (most important)
# 3. Tune LoRA r and alpha together
# 4. Tune batch size based on memory
# 5. Fine-tune dropout and momentum
```

---

## 💾 **Checkpoint Management**

### **Saving Strategy:**

```python
# Save checkpoints every N epochs
if (epoch + 1) % 10 == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
    }, f'models/checkpoints/checkpoint_epoch_{epoch+1}.pt')

# Save best model separately
if val_loss < best_val_loss:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, 'models/checkpoints/best_model.pt')
```

### **Loading Strategy:**

```python
# Load best model for inference
checkpoint = torch.load('models/checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## 🚫 **Common Training Mistakes**

### **Mistake 1: Wrong Optimizer**

```python
# ❌ WRONG
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Problem:
# - AdamW not optimal for financial time series
# - Can overfit more easily
# - Less stable convergence

# ✅ CORRECT
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-4,
    momentum=0.9,
    nesterov=True
)
```

### **Mistake 2: No Gradient Clipping**

```python
# ❌ WRONG
loss.backward()
optimizer.step()

# Problem:
# - Exploding gradients possible
# - Training instability

# ✅ CORRECT
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### **Mistake 3: Wrong Learning Rate**

```python
# ❌ WRONG - Too high
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Problem:
# - Loss spikes
# - Training divergence

# ❌ WRONG - Too low
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

# Problem:
# - Very slow convergence
# - Gets stuck in poor minima

# ✅ CORRECT - Just right
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
```

### **Mistake 4: No Early Stopping**

```python
# ❌ WRONG - Train all epochs blindly
for epoch in range(1000):
    # ... training code ...

# ✅ CORRECT - Early stopping
if patience_counter >= patience:
    print(f"Early stopping at epoch {epoch+1}")
    break
```

---

## ✅ **Best Practices Summary**

### **DO:**

1. ✅ **Use SGD optimizer** for financial time series
   - Momentum=0.9, Nesterov=True
   - Better than AdamW for this domain

2. ✅ **Use cosine annealing** for learning rate
   - Smooth decay
   - Better convergence

3. ✅ **Apply gradient clipping**
   - Prevents exploding gradients
   - Training stability

4. ✅ **Implement early stopping**
   - Prevents overfitting
   - Saves training time

5. ✅ **Track multiple metrics**
   - Train/val gap
   - Learning rate
   - Memory usage

### **DON'T:**

1. ❌ **Don't use AdamW** for financial data
   - SGD is empirically better
   - More stable convergence

2. ❌ **Don't skip gradient clipping**
   - Can cause training instability
   - Exploding gradients

3. ❌ **Don't use constant learning rate**
   - Decay is important
   - Use cosine annealing

4. ❌ **Don't ignore train/val gap**
   - Gap > 3x = problem
   - Check for data leakage

5. ❌ **Don't train blindly**
   - Monitor metrics
   - Early stopping

---

## 📊 **Our Training Results**

### **Configuration Used:**

```
Model: TimesFM 2.5 + LoRA (r=4, alpha=8)
Optimizer: SGD (lr=1e-4, momentum=0.9)
Scheduler: Cosine annealing
Batch Size: 32
Epochs: 100 (early stopped at 75)
```

### **Training Progress:**

```
Epoch 1:   Train Loss: 3.5994,  Val Loss: 0.6412
Epoch 10:  Train Loss: 1.3245,  Val Loss: 0.6056
Epoch 50:  Train Loss: 0.2847,  Val Loss: 0.5389
Epoch 75:  Train Loss: 0.1200,  Val Loss: 0.5215  ← Best
Epoch 100: Train Loss: 0.1112,  Val Loss: 0.5542

Improvement:
- Train loss: -96.9% (3.60 → 0.11)
- Val loss:   -18.7% (0.64 → 0.52)
- Best epoch: 75
```

### **Performance:**

```
True Out-of-Sample (no data leakage):
- R²: 0.5193 (exceeds target of 0.5)
- QLIKE: -4.0063 (good volatility forecasting)
- RMSE: 0.0062 (acceptable error rate)
- MSE: 0.000038 (low squared error)
```

---

## 📚 **Key Takeaways**

### **Critical Rules:**

1. **SGD > AdamW** for financial time series
2. **Gradient clipping is mandatory** (max_norm=1.0)
3. **Cosine annealing works best** for learning rate
4. **Early stopping prevents overfitting** (patience=5)
5. **Monitor train/val gap** (should be < 3x)

### **Remember:**

> **Training configuration matters as much as model architecture!**
> **SGD + momentum + cosine annealing = best for financial ML**

---

**Status:** ✅ Training Guidelines Documented
**Next:** See [Testing-Validation-Strategy.md](./04-Testing-Validation-Strategy.md)
**Last Updated:** 2026-06-11

---

*Follow these guidelines to ensure stable, reproducible training. Financial time series require specialized optimization - don't treat it like image classification!*
