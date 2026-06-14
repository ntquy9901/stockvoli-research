# Learning Curve Implementation - TimesFM Training

**Date:** 2026-06-10
**Status:** ✅ **REAL-TIME LEARNING CURVES IMPLEMENTED**

---

## 🎨 **Features Added:**

### **1. Real-Time Learning Curves**
- ✅ **Auto-updates after each epoch** (no need to wait for completion)
- ✅ **Saves to:** `experiments/learning_curves.png`
- ✅ **Shows:**
  - Training & Validation loss curves
  - Learning rate schedule
  - Best epoch marker
  - Best val loss reference line

### **2. Training Summary Plot**
- ✅ **Generated at training completion**
- ✅ **Saves to:** `experiments/training_summary.png`
- ✅ **Shows:**
  - 4-panel comprehensive summary
  - Loss curves with best epoch
  - Overfitting indicator (Val/Train ratio)
  - Learning rate schedule
  - Training statistics table

### **3. Real-Time Monitoring Script**
- ✅ **Script:** `monitor_training.py`
- ✅ **Features:**
  - Auto-refresh every 30 seconds
  - Show current epoch & losses
  - Display recent performance trend
  - Alert for early stopping
  - Show plot file location

---

## 📊 **Usage:**

### **Option 1: Training (Auto-generates curves)**

```bash
# Start training (learning curves update automatically)
python src/model_training_google_research.py

# In another terminal, monitor in real-time
python monitor_training.py
```

**What happens:**
1. After each epoch → `learning_curves.png` is updated
2. Monitor script shows progress every 30 seconds
3. Plots include best epoch markers and statistics

---

### **Option 2: View Existing Curves**

```bash
# Open the latest learning curves
start experiments/learning_curves.png

# Or view in Python
python -c "
from PIL import Image
img = Image.open('experiments/learning_curves.png')
img.show()
"
```

---

## 📈 **Learning Curve Features:**

### **Real-Time Plot (after each epoch):**

**Panel 1: Loss Curves**
- Blue line: Training loss
- Orange line: Validation loss
- Green dashed: Best epoch
- Red dashed: Best val loss reference
- Green star: Best epoch marker

**Panel 2: Learning Rate**
- Red line: Learning rate over epochs
- Log scale (to see cosine annealing decay)

---

### **Final Summary Plot (at completion):**

**Panel 1: Loss Curves**
- Full training history
- Best epoch highlighted
- Best val loss marked

**Panel 2: Overfitting Indicator**
- Purple line: Val loss / Train loss ratio
- Gray dashed: Ratio = 1.0 baseline
- Red shaded area: Overfitting zone (ratio > 1)
- **Interpretation:** Ratio > 1 indicates overfitting

**Panel 3: Learning Rate Schedule**
- Full LR schedule (log scale)
- Shows cosine annealing pattern

**Panel 4: Statistics Table**
- Total epochs trained
- Best epoch & losses
- Loss improvement percentages
- Configuration summary
- LORA settings

---

## 🔍 **Monitoring Script Features:**

```bash
python monitor_training.py
```

**Output every 30 seconds:**
```
======================================================================
               TIMESFM TRAINING MONITOR
======================================================================

[STATUS] Training Progress - 2026-06-10 12:35:00
----------------------------------------------------------------------
Current Epoch:  28/100
Train Loss:     0.2027 (-0.0050)
Val Loss:       0.5449
Learning Rate:  0.00009500
Best Epoch:     24 (Val Loss: 0.5261)
----------------------------------------------------------------------

[TREND] Recent Performance (Last 5 epochs):
Epoch | Train Loss | Val Loss | Train Δ | Val Δ
--------------------------------------------------
   24 |    0.2783 |   0.5261 |   -0.0108 |  -0.0509
   25 |    0.2302 |   0.6113 |   -0.0481 |  +0.0852
   26 |    0.2163 |   0.5943 |   -0.0139 |  -0.0170
   27 |    0.2077 |   0.5641 |   -0.0086 |  -0.0302
   28 |    0.2027 |   0.5449 |   -0.0050 |  -0.0192

[PLOT] Learning Curves:
  Path: experiments/learning_curves.png
  Size: 91.0 KB
  Last Updated: 12:31:05

  Open this file to see training curves:
  → D:\bmad-projects\stockvoli-research\experiments\learning_curves.png

[ALERT] Early stopping likely soon (4/5 epochs without improvement)
```

---

## 📁 **Generated Files:**

### **During Training:**
```
experiments/
├── learning_curves.png          # Updated after each epoch
├── training_history.json         # Training data
└── model_training.log           # Detailed logs
```

### **After Training:**
```
experiments/
├── learning_curves.png          # Final curves
├── training_summary.png         # Comprehensive summary
├── training_summary.txt         # Text summary
└── training_history.json         # Complete history
```

---

## 🎯 **Interpreting Learning Curves:**

### **Good Training:**
```
✅ Both losses decreasing steadily
✅ Val loss < Train loss (ratio < 1)
✅ Gap between train/val not widening
✅ Learning rate decays smoothly
```

### **Warning Signs:**
```
⚠️ Train loss decreasing but val loss increasing (overfitting)
⚠️ Val/Train ratio > 1.5 (severe overfitting)
⚠️ Loss spikes (data quality or learning rate issues)
⚠️ Plateau in val loss (consider early stopping)
```

### **Current Training Analysis (Epoch 28):**
```
Train Loss: 0.20 → Still decreasing (good)
Val Loss:   0.54 → Stable (not improving much)
Ratio:      2.69x → Overfitting indicator
Verdict:    Approaching convergence, consider early stopping
```

---

## 🚀 **Quick Start:**

### **Step 1: Start Training**
```bash
python src/model_training_google_research.py
```

### **Step 2: Monitor in Another Terminal**
```bash
python monitor_training.py
```

### **Step 3: View Curves**
```bash
# Auto-opens every 30 seconds
# Or manually open:
start experiments/learning_curves.png
```

---

## 🔧 **Implementation Details:**

### **Auto-Update After Each Epoch:**
```python
# In train_model() method, after each epoch:
self.plot_learning_curves()  # Updates learning_curves.png

# At training completion:
self.create_training_summary()  # Creates training_summary.png
```

### **Plotting Library:**
- Uses `matplotlib` with `Agg` backend (non-interactive)
- Saves as PNG (150 DPI)
- Works in server environments

### **Data Tracked:**
```python
epoch_log = {
    'epoch': epoch + 1,
    'train_metrics': train_metrics,
    'val_metrics': val_metrics,
    'learning_rate': optimizer.param_groups[0]['lr']  # NEW
}
```

---

## 📊 **Example Output:**

### **Learning Curves Plot:**
- **Left panel:** Loss curves (train & val)
- **Right panel:** Learning rate schedule
- **Markers:** Best epoch, best val loss
- **Grid:** For easier reading

### **Training Summary Plot:**
- **4 panels:** Loss, Overfitting, LR, Stats
- **Comprehensive:** All training info in one plot
- **Publication-ready:** High quality (150 DPI)

---

## 💡 **Tips:**

1. **Monitor during training:** Use `monitor_training.py`
2. **Check for overfitting:** Look at Val/Train ratio
3. **Early stopping:** Watch for "No improvement for X epochs" in logs
4. **Best model:** Automatically saved when val loss improves
5. **Final analysis:** Check `training_summary.png` after completion

---

## ✅ **Summary:**

**Features Added:**
- ✅ Real-time learning curves (update each epoch)
- ✅ Training summary plot (at completion)
- ✅ Monitoring script (auto-refresh)
- ✅ Overfitting indicator (Val/Train ratio)
- ✅ Best epoch markers
- ✅ Learning rate tracking

**Files Created:**
- ✅ `experiments/learning_curves.png` (real-time)
- ✅ `experiments/training_summary.png` (final)
- ✅ `experiments/training_summary.txt` (text summary)
- ✅ `monitor_training.py` (monitoring script)

**Status: ✅ READY FOR REAL-TIME MONITORING**

---

**Next:** Run `python monitor_training.py` to monitor current training!
