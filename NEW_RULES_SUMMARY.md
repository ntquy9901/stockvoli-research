# ✅ Added New Rules: File Archiving & Learning Curves

**Date:** 2026-06-13  
**Repositories:** ml-ds-common-rules & stockvoli-research

---

## 🎯 Two Important New Rules Added

### 1. File Archiving Rule

**Problem:** Projects accumulate many old files, making it hard to know which files are current.

**Solution:** When fixing or creating new files, move old files to `archived/` folder.

---

### 2. Learning Curves Rule (MANDATORY)

**Problem:** Training without visualization can't detect overfitting until it's too late.

**Solution:** MANDATORY learning curve visualization for ALL training runs.

---

## 📁 File Archiving Rule

### When to Apply
- Creating new version of existing file
- Fixing bugs in file
- Refactoring code
- Improving implementation

### How to Apply

**✅ CORRECT:**
```bash
# Directory structure
project/
├── src/                    # Active code only
│   ├── model_training.py
│   ├── data_processing.py
│   └── model_evaluation.py
└── archived/               # Old code for reference
    ├── train_old_2025-06-13_refactored.py
    ├── processing_v1.py
    └── evaluate_old_2025-06-13_bugfix.py

# Naming convention: filename_old_DATE_REASON.py
# Example: model_training_old_2025-06-13_refactored.py
# Example: data_processing_old_2025-06-13_bugfix_v1.py
```

**❌ WRONG:**
```bash
# Leave old files in src/
src/train.py
src/train_new.py
src/train_fixed.py  # Confusing! Which one is current?
```

### Benefits
- ✅ Clean directory structure
- ✅ Easy to identify current files
- ✅ Reference old implementations
- ✅ Track code evolution
- ✅ No file clutter

---

## 📊 Learning Curves Rule (MANDATORY)

### When to Apply
- ALL training runs (no exceptions!)
- Every model training
- Every hyperparameter experiment

### How to Apply

**✅ CORRECT - With Learning Curves:**
```python
# During training
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss = validate_epoch(model, val_loader)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # Plot every N epochs (MANDATORY!)
    if (epoch + 1) % 10 == 0:
        plot_learning_curves(
            train_losses, 
            val_losses, 
            f'experiments/learning_curves_epoch_{epoch+1}.png'
        )
    
    # Check for overfitting
    # Val loss increases while train loss decreases = OVERFITTING
    if len(val_losses) > 10:
        recent_val = val_losses[-5:]
        if all(recent_val[i] > recent_val[i-1] for i in range(1, len(recent_val))):
            print("⚠️ WARNING: Overfitting detected!")
            print(f"   Recent val losses: {recent_val}")
            break  # Early stopping

# Final learning curve (MANDATORY!)
plot_learning_curves(
    train_losses, 
    val_losses, 
    'experiments/learning_curves_final.png'
)
```

**❌ WRONG - No Learning Curves:**
```python
# No learning curves at all
for epoch in range(num_epochs):
    train(model, data)
# How do you know if it's overfitting?

# Can't detect issues without visualization!
```

### What Learning Curves Show

**Normal Training:**
```
Train Loss:    ╲
Val Loss:      ╲  (Both decrease, good!)
```

**Overfitting:**
```
Train Loss:    ╲
Val Loss:      ╱  (Val increases = overfitting!)
```

**Underfitting:**
```
Train Loss:    ╲ (but still high)
Val Loss:      ╲ (both high, need more capacity)
```

### Benefits
- ✅ Early overfitting detection
- ✅ Visual training monitoring
- ✅ Better hyperparameter tuning
- ✅ Improved experiment documentation
- ✅ Mandatory quality check

---

## 🔄 Integration into Projects

### For ml-ds-common-rules Repository

**Files Updated:**
1. `COMMON_RULES.md`
   - Added Section: "File Management and Archiving"
   - Added Section: "Learning Curves and Overfitting Detection"

2. `QUICK_REFERENCE.md`
   - Added summary of both rules
   - Included in Research Best Practices

**Location:** https://github.com/ntquy9901/ml-ds-common-rules  
**Commit:** ff1e2e8

### For stockvoli-research Repository

**Files Updated:**
1. `CLAUDE.md`
   - Added Section 7.4: "File Management and Archiving"
   - Added Section 7.5: "Learning Curves (MANDATORY)"
   - Included full implementation examples

**Location:** https://github.com/ntquy9901/stockvoli-research  
**Commit:** dc9a687

---

## 📋 Implementation Checklist

### File Archiving
- [ ] Create `archived/` directory in project root
- [ ] Move old/deprecated files to `archived/`
- [ ] Use naming convention: `filename_old_DATE_REASON.py`
- [ ] Keep `src/` clean (only active code)
- [ ] Update `archived/README.md` if needed

### Learning Curves
- [ ] Add `plot_learning_curves()` function to training script
- [ ] Plot every N epochs (e.g., every 10)
- [ ] Save final learning curve
- [ ] Check for overfitting automatically
- [ ] Include in experiment reports
- [ ] Apply to ALL training runs (no exceptions!)

---

## 🎓 Training and Onboarding

### For New Team Members

**Explain File Archiving:**
1. "When you fix or improve a file, move the old version to `archived/`"
2. "Use descriptive names so we know why it was archived"
3. "Keep `src/` clean - only current working code"

**Explain Learning Curves:**
1. "Learning curves are MANDATORY for all training"
2. "Plot them every N epochs to catch issues early"
3. "If val loss increases while train loss decreases = overfitting"
4. "Always save final learning curve with experiment results"

### Code Review Checklist

Add to your code review process:

```markdown
## New Rules Checklist

### File Management
- [ ] Old files moved to archived/?
- [ ] Descriptive archived naming (DATE_REASON)?
- [ ] src/ directory clean (no clutter)?

### Learning Curves
- [ ] Learning curves plotted during training?
- [ ] Saved final learning curve?
- [ ] No overfitting detected?
- [ ] Included in experiment report?
```

---

## 📚 Documentation References

### In ml-ds-common-rules

**COMMON_RULES.md:**
- Section: "File Management and Archiving"
- Section: "Learning Curves and Overfitting Detection"

**QUICK_REFERENCE.md:**
- Section: "File Management (Archiving)"
- Section: "Learning Curves (Mandatory)"

### In stockvoli-research

**CLAUDE.md:**
- Section 7.4: "File Management and Archiving"
- Section 7.5: "Learning Curves (MANDATORY)"

---

## 🔗 Related Resources

**GitHub Repositories:**
- ml-ds-common-rules: https://github.com/ntquy9901/ml-ds-common-rules
- stockvoli-research: https://github.com/ntquy9901/stockvoli-research

**Key Commits:**
- ml-ds-common-rules: ff1e2e8
- stockvoli-research: dc9a687

---

## 🚀 Next Steps

### For Existing Projects

1. **Clean up directories:**
   ```bash
   # Create archived/ directory
   mkdir -p archived
   
   # Move old files
   mv src/train_old.py archived/train_old_2025-06-13_refactored.py
   mv src/process_v1.py archived/process_v1.py
   ```

2. **Add learning curves to training:**
   ```python
   # Add to training script
   from plotting import plot_learning_curves
   
   # Plot during training
   plot_learning_curves(train_losses, val_losses, 'curves.png')
   ```

### For New Projects

1. **Use updated ml-ds-common-rules:**
   ```bash
   git submodule update --remote docs/common-rules
   ```

2. **Follow new rules from day 1:**
   - Archive old files immediately
   - Plot learning curves for all training

---

## 📊 Impact Summary

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **File clutter** | Many old files in src/ | Clean src/ | Better organization |
| **Current file confusion** | Which file to use? | Only files in src/ | Clear |
| **Overfitting detection** | After training (too late) | During training (early) | Much faster |
| **Training visibility** | Numbers only | Visual curves | Better insight |
| **Experiment documentation** | Metrics only | Metrics + curves | Complete |

---

## ✅ Summary

**Two New Rules Added:**

1. **File Archiving Rule:**
   - Move old files to `archived/` folder
   - Use descriptive naming with date and reason
   - Keep `src/` clean and organized

2. **Learning Curves Rule (MANDATORY):**
   - Plot learning curves for ALL training runs
   - Plot every N epochs for early detection
   - Detect overfitting: val loss increases while train loss decreases
   - Save final learning curve with results

**Repositories Updated:**
- ✅ ml-ds-common-rules (commit ff1e2e8)
- ✅ stockvoli-research (commit dc9a687)

**Ready to use immediately!** 🚀

---

*Last Updated: 2026-06-13*
*New Rules Added: File Archiving & Learning Curves*
