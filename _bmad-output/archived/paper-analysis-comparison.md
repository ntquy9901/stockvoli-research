# Research Papers Analysis & Implementation Comparison

## Your Research Papers Overview

### Paper 1: TimesFM for Volatility Forecasting (arXiv:2505.11163)
**Title:** "Foundation Time-Series AI Model for Realized Volatility Forecasting"  
**Authors:** Anubha Goel, Puneet Pasricha, Martin Magris, Juho Kanniainen  
**Date:** May 16, 2025

**Key Contributions:**
- Evaluates **TimesFM** (Time Series Foundation Model) for volatility forecasting
- Introduces **incremental fine-tuning** methodology
- Shows that fine-tuned models **statistically outperform** traditional econometric models
- Uses **Diebold-Mariano** and **Giacomini-White** tests for statistical validation

**Critical Finding:** "incremental fine-tuning, which allows the model to adapt to new financial return data over time, is essential for learning volatility patterns effectively"

### Paper 2: Moirai 2.0 (arXiv:2511.11698)
**Title:** "Moirai 2.0: When Less Is More for Time Series Forecasting"  
**Authors:** Chenghao Liu, Taha Aksu, et al.  
**Date:** November 12, 2025 (v3: February 3, 2026)

**Key Contributions:**
- **Decoder-only architecture** (simpler than masked-encoder)
- Trained on **36M time series** corpus
- **Quantile forecasting** for probabilistic predictions
- **Recursive multi-quantile decoding**
- **2x faster, 30x smaller** than Moirai 1.0-Large

## Comparison: My Generic Approach vs. Paper-Based Approach

| Aspect | My Previous Generic Guide | Paper-Based Approach |
|--------|---------------------------|----------------------|
| **Fine-Tuning Method** | Standard LoRA fine-tuning | **Incremental learning** (TimesFM) |
| **Model Architecture** | Generic transformer | **Decoder-only** (Moirai 2.0) |
| **Training Strategy** | Epoch-based training | **Continuous incremental adaptation** |
| **Prediction Output** | Point predictions | **Quantile forecasts** (Moirai 2.0) |
| **Validation** | Standard metrics (MAE, RMSE) | **Statistical tests** (DM, GW) |
| **Model Selection** | General foundation models | **TimesFM** or **Moirai 2.0** specifically |
| **Training Data** | Vietnamese market data | **36M series** (Moirai) + incremental adaptation |

## Key Differences & Corrections

### 1. Fine-Tuning Methodology
**❌ My Wrong Approach:** Standard LoRA fine-tuning with fixed epochs  
**✅ Paper Approach:** **Incremental learning** - continuous adaptation as new financial data arrives

**TimesFM Paper Quote:** "incremental fine-tuning, which allows the model to adapt to new financial return data over time, is essential for learning volatility patterns effectively"

### 2. Model Architecture  
**❌ My Wrong Approach:** Generic transformer with attention mechanisms  
**✅ Paper Approach:** **Decoder-only** (Moirai 2.0) - simpler, more efficient

**Moirai 2.0 Paper:** "Replaces masked-encoder training, multi-patch inputs, and mixture-distribution outputs with a simpler decoder-only architecture"

### 3. Validation Methodology
**❌ My Wrong Approach:** Standard ML metrics (MAE, RMSE)  
**✅ Paper Approach:** **Statistical significance testing** (Diebold-Mariano, Giacomini-White)

**TimesFM Paper:** "Fine-tuned variants not only improve forecast accuracy but also statistically outperform traditional models, as demonstrated through Diebold-Mariano and Giacomini-White tests"

### 4. Prediction Style
**❌ My Wrong Approach:** Single point predictions  
**✅ Paper Approach:** **Quantile forecasting** with confidence intervals

**Moirai 2.0 Paper:** "The model adopts quantile forecasting and multi-token prediction, improving both probabilistic accuracy and inference efficiency"

## Corrected Implementation Approach

### Step 1: Choose Right Base Model
```python
# CORRECT: Use models from your papers
timesfm_model = "google/timesfm"  # From paper 2505.11163
moirai_model = "Salesforce/moirai-2.0-R-small"  # From paper 2511.11698

# WRONG: Generic models (my previous suggestion)
generic_model = "facebook/chronos-5-t5-small"  # Not from your papers
```

### Step 2: Apply Incremental Fine-Tuning
```python
# CORRECT: Incremental learning (TimesFM approach)
class IncrementalFineTuner:
    def update_on_new_data(self, new_data_window):
        """Continuously adapt as new financial data arrives"""
        # Single epoch on new data, then move to next window
        self.train_single_epoch(new_data_window)
        
# WRONG: Fixed epochs (my previous approach)
class StandardFineTuner:
    def train_fixed_epochs(self, all_data, epochs=10):
        """Train all data for fixed epochs"""
        for epoch in range(epochs):
            self.train_epoch(all_data)
```

### Step 3: Use Quantile Forecasting
```python
# CORRECT: Quantile forecasting (Moirai 2.0 approach)
def predict_volatility_quantiles(context, quantiles=[0.1, 0.5, 0.9]):
    """Generate probabilistic forecasts"""
    predictions = {}
    for q in quantiles:
        predictions[f'q_{q}'] = model.predict(context, quantile=q)
    return predictions

# WRONG: Point predictions (my previous approach)  
def predict_volatility_point(context):
    """Generate single point forecast"""
    return model.predict(context)
```

### Step 4: Statistical Validation
```python
# CORRECT: Statistical significance testing (TimesFM approach)
def validate_with_statistical_tests(model_preds, benchmark_preds):
    """Test if improvements are statistically significant"""
    dm_statistic = diebold_mariano_test(model_preds, benchmark_preds)
    gw_statistic = giacomini_white_test(model_preds, benchmark_preds)
    return dm_statistic, gw_statistic

# WRONG: Simple metric comparison (my previous approach)
def validate_with_metrics(model_preds, actual):
    """Compare error metrics"""
    mae = mean_absolute_error(model_preds, actual)
    return mae
```

## Implementation Recommendations

### For Vietnamese Stock Volatility Prediction:

**Primary Recommendation: TimesFM with Incremental Learning**
- Use TimesFM as base model (from paper 2505.11163)
- Implement **incremental fine-tuning** - adapt model continuously as new Vietnamese market data arrives
- Validate with **Diebold-Mariano tests** to prove statistical superiority over GARCH/ARIMA
- Focus on **realized volatility (RV)** as target variable

**Secondary Recommendation: Moirai 2.0 for Probabilistic Forecasts**
- Use Moirai 2.0 for **quantile forecasting** (from paper 2511.11698)
- Generate **prediction intervals** (10th, 50th, 90th percentiles)
- Leverage **computational efficiency** (2x faster, 30x smaller)
- Use for **probabilistic risk management**

### Implementation Priority:
1. **Phase 1:** TimesFM incremental learning setup
2. **Phase 2:** Statistical validation framework  
3. **Phase 3:** Moirai 2.0 quantile forecasting
4. **Phase 4:** Hybrid system combining both approaches

## Code Correction Examples

### Before (My Generic Approach):
```python
# Generic LoRA fine-tuning
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj"])
lora_model = get_peft_model(base_model, lora_config)

# Train for fixed epochs
trainer = Trainer(model=lora_model, args=training_args)
trainer.train()
```

### After (Paper-Based Approach):
```python
# TimesFM incremental learning (Paper 2505.11163)
class IncrementalFineTuner:
    def incremental_update(self, new_data_window):
        """Adapt to new financial data over time"""
        # Single epoch on new data only
        self.train_single_epoch(new_data_window)
        self.incremental_step += 1

# Apply incremental learning as new Vietnamese data arrives
for time_window in vietnamese_data_windows:
    finetuner.incremental_update(time_window)
```

### Before (My Generic Approach):
```python
# Point predictions
predictions = model.predict(context)
mae = mean_absolute_error(predictions, actual)
```

### After (Paper-Based Approach):
```python
# Quantile forecasting (Paper 2511.11698)
quantile_predictions = model.predict_quantiles(context, [0.1, 0.5, 0.9])
prediction_intervals = quantile_predictions['q_0.9'] - quantile_predictions['q_0.1']

# Statistical validation (Paper 2505.11163)
dm_result = diebold_mariano_test(predictions, benchmark_predictions)
print(f"Statistical significance: p-value = {dm_result['p_value']}")
```

## Summary

Your research papers provide **specific, validated methodologies** for foundation model fine-tuning in financial volatility forecasting:

1. **TimesFM** uses **incremental learning** for continuous adaptation
2. **Moirai 2.0** uses **decoder-only architecture** and **quantile forecasting**  
3. **Statistical validation** using Diebold-Mariano and Giacomini-White tests
4. **Proven results**: Fine-tuned models statistically outperform traditional methods

The corrected implementation guide now follows these specific methodologies rather than generic ML approaches.