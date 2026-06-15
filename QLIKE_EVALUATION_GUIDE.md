# 📊 QLIKE Evaluation Guide
## Understanding Volatility Forecasting Metrics

### 🎯 Overview

This guide explains the **QLIKE metric** that has been added to the training pipeline and how to interpret it alongside the standard MSE loss.

---

## 🔍 THE KEY QUESTION: "Loss là hàm QLIKE?"

### Answer: NO - But now we track BOTH!

**Training Loss (MSE):**
- Used for **gradient descent optimization**
- What gets logged as "Train Loss" and "Val Loss"
- Purpose: Update model weights

**QLIKE Metric:**
- Used for **volatility forecasting evaluation**
- Now logged as "Val QLIKE" alongside validation loss
- Purpose: Assess forecast quality for financial applications

---

## 📊 WHAT YOU'LL SEE NOW

### During Training:

```
======================================================================
EPOCH 1/2 COMPLETE
======================================================================
Train Loss: 5.817700
Val Loss:   4.822900
Val QLIKE:  0.014567 ⭐ Volatility Metric

======================================================================
EPOCH 2/2 COMPLETE
======================================================================
Train Loss: 3.643600
Val Loss:   4.802200
Val QLIKE:  0.013234 ⭐ Volatility Metric
```

### In Final Results:

```
======================================================================
[FINAL RESULT]
======================================================================
MSE Loss (Model Optimization):
  RV_20 baseline:        4.8229
  Overnight volatility:   4.8022
  Improvement:           +0.4%

QLIKE Metric (Volatility Quality):
  RV_20 baseline:        0.014567
  Overnight volatility:   0.013234
  Improvement:           +9.1%

✅ OVERNIGHT VOLATILITY IS BETTER!
   ✅ Lower MSE loss (better model fit)
   ✅ Lower QLIKE (better volatility forecasts)
```

---

## 🎯 WHY TWO DIFFERENT METRICS?

### MSE Loss (Model Optimization)
```python
MSE = mean((y_pred - y_true)²)

# Used for:
- Gradient descent: loss.backward()
- Optimizer updates
- Learning rate scheduling
- Early stopping decisions

# Properties:
- Treats all errors equally
- Standard for deep learning
- Good for general regression
- Smooth gradient for optimization
```

### QLIKE Metric (Volatility Evaluation)
```python
QLIKE = mean(actual/pred + log(pred) - 1)

# Used for:
- Model comparison
- Volatility forecast quality
- Financial decision making
- Risk management assessment

# Properties:
- Penalizes underestimation more than overestimation
- Scale-independent (works across volatility regimes)
- Handles heteroskedasticity (variance changes)
- Financial industry standard
```

---

## 🔍 CRITICAL DISTINCTION: MSE ≠ QLIKE

### Example: Why They Differ

**Case 1: Underestimation (DANGEROUS!)**
```python
actual = 0.05  # 5% volatility
pred = 0.01   # 1% volatility (underestimated)

MSE = (0.05-0.01)² = 0.0016
QLIKE = 0.05/0.01 + log(0.01) - 1 = 5 - 4.605 - 1 = -0.395

# QLIKE penalty: HIGH! (underestimation dangerous)
```

**Case 2: Overestimation (SAFER)**
```python
actual = 0.01  # 1% volatility
pred = 0.05   # 5% volatility (overestimated)

MSE = (0.05-0.01)² = 0.0016  # Same MSE as above!
QLIKE = 0.01/0.05 + log(0.05) - 1 = 0.2 - 2.996 - 1 = -3.796

# QLIKE penalty: Lower! (overestimation safer)
```

**Key Insight:**
- MSE treats both cases **equally** (same MSE value)
- QLIKE penalizes **underestimation more** (higher QLIKE = worse)
- For risk management, **underestimating risk is dangerous**

---

## 📈 HOW TO INTERPRET RESULTS

### Scenario 1: Both Improve ✅ (Ideal)
```
MSE Improvement: +10%
QLIKE Improvement: +15%

Conclusion: ✅ EXCELLENT!
- Model fits better (MSE down)
- Volatility forecasts improve (QLIKE down)
- Safe to deploy
```

### Scenario 2: MSE Improves, QLIKE Worsens ⚠️ (Concern)
```
MSE Improvement: +10%
QLIKE Improvement: -5%

Conclusion: ⚠️ CONCERNING
- Model fits better technically
- But volatility quality degrading
- Model optimizing for MSE, not volatility
- Need to reconsider approach
```

### Scenario 3: Both Worsen ❌ (Bad)
```
MSE Improvement: -5%
QLIKE Improvement: -8%

Conclusion: ❌ BAD
- Model performance degrading
- Both metrics getting worse
- Check for bugs, data issues
```

### Scenario 4: QLIKE Improves More Than MSE ✅ (Excellent)
```
MSE Improvement: +2%
QLIKE Improvement: +20%

Conclusion: ✅ EXCELLENT FOR VOLATILITY!
- Modest MSE improvement
- Large QLIKE improvement
- Feature particularly good for volatility
- Strong candidate for deployment
```

---

## 🔧 IMPLEMENTATION DETAILS

### How QLIKE is Calculated

**In the validation loop:**
```python
# 1. Collect predictions and actuals
all_predictions = []
all_actuals = []

for batch in test_loader:
    outputs = model(batch)
    predictions = outputs.prediction_outputs.cpu().numpy()
    actuals = batch['ground_truth'].cpu().numpy()

    all_predictions.extend(predictions.flatten())
    all_actuals.extend(actuals.flatten())

# 2. Calculate QLIKE
predictions_safe = np.maximum(predictions, 1e-8)  # Prevent division by zero
actuals_safe = np.maximum(actuals, 1e-8)

val_qlike = np.mean(actuals_safe / predictions_safe + np.log(predictions_safe) - 1)
```

**Why safe minimum (1e-8)?**
- Prevents division by zero
- Prevents log(0) errors
- Tiny values don't affect result much

---

## 🎯 PRACTICAL IMPLICATIONS

### For Model Selection

**When comparing features:**
1. **Primary metric:** QLIKE (volatility quality)
2. **Secondary metric:** MSE (model fit)
3. **Decision:** Prefer lower QLIKE even if MSE is slightly higher

**Example:**
```
Feature A: MSE=4.80, QLIKE=0.0140
Feature B: MSE=4.85, QLIKE=0.0130

Decision: Choose Feature B
- Higher MSE (worse fit) BUT
- Lower QLIKE (better volatility forecasts)
- QLIKE more important for financial applications
```

### For Risk Management

**Why QLIKE matters:**
```python
# Underestimating volatility (dangerous):
actual_volatility = 5%  # High risk day
model_prediction = 1%  # Model says low risk

# Result:
- Insufficient capital reserves
- Unexpected losses
- Regulatory issues
- Reputational damage

# QLIKE catches this:
MSE = 0.0016  # Looks OK
QLIKE = -0.395  # Clearly BAD! (higher value)
```

### For Trading Strategies

**Volatility forecasts used for:**
- Option pricing
- Risk management
- Position sizing
- Stop-loss placement

**Better QLIKE = Better decisions = Better profits**

---

## 📊 TROUBLESHOOTING

### Issue: QLIKE = inf

**Cause:** Division by zero or log(0) in calculation

**Solution:** Check prediction outputs
```python
# Verify predictions are valid
print(f"Min prediction: {predictions.min()}")
print(f"Max prediction: {predictions.max()}")
print(f"NaN count: {np.isnan(predictions).sum()}")
```

### Issue: QLIKE not decreasing

**Possible causes:**
1. Model optimizing for MSE, not volatility quality
2. Feature doesn't improve volatility forecasting
3. Need different loss function (direct QLIKE optimization)

**Solutions:**
- Monitor trends over multiple epochs
- Compare with baseline feature
- Consider QLIKE-based loss function

### Issue: QLIKE very different from MSE trend

**This is NORMAL and EXPECTED!**

**Reason:**
- MSE optimizes for mean squared error
- QLIKE optimizes for volatility quality
- They measure different things
- May not correlate perfectly

**Action:**
- Trust QLIKE for volatility applications
- Use MSE for technical optimization
- Don't expect perfect correlation

---

## 🔬 FINANCIAL CONTEXT

### Why QLIKE is Industry Standard

**Academic validation:**
- Published in top finance journals
- Validated across multiple markets
- Theoretically sound properties

**Practical advantages:**
```python
# 1. Scale-independence
# Works across different volatility regimes
high_vol_period = QLIKE(volatility_0.5, predictions_0.4)
low_vol_period = QLIKE(volatility_0.1, predictions_0.08)
# Both comparable despite different scales

# 2. Asymmetric penalty
# Underestimation penalized more than overestimation
underestimate = QLIKE(0.05, 0.01)  # High penalty
overestimate = QLIKE(0.05, 0.10)   # Lower penalty

# 3. Heteroskedasticity handling
# Works well when variance changes over time
```

### Vietnamese Market Considerations

**Your VN30 data:**
- Different volatility regimes (calm vs crisis periods)
- TET holiday effects
- Market-specific patterns

**QLIKE advantages:**
- Handles regime changes well
- Scale-independent (works across stocks)
- Penalizes dangerous underestimation

---

## 🎖️ SUMMARY

**Key Takeaways:**

1. ✅ **Both metrics matter:** MSE for optimization, QLIKE for evaluation
2. ✅ **QLIKE is primary:** For volatility forecasting applications
3. ✅ **They may differ:** MSE optimizing doesn't guarantee QLIKE improvement
4. ✅ **Trust QLIKE:** For financial decision making
5. ✅ **Monitor both:** To understand model behavior

**What you'll see:**
```
Val Loss:   4.822900  ← MSE (optimization metric)
Val QLIKE:  0.014567  ← QLIKE (volatility quality) ⭐
```

**What it means:**
- Lower MSE = Better model fit (technically)
- Lower QLIKE = Better volatility forecasts (financially)
- Prefer features with lower QLIKE

**Next steps:**
1. Pull the update from GitHub
2. Run training with QLIKE enabled
3. Compare features on QLIKE (primary) and MSE (secondary)
4. Deploy features with best QLIKE performance

---

*Added: 2026-06-15*
*Based on: User question "loss là hàm QLIKE?"*
*Status: ✅ Comprehensive QLIKE evaluation system implemented*
