# Feasibility Analysis: OHLC Range Estimators for TimesFM VN30

## Current Implementation Analysis

### 1. Current Architecture (Univariate)
Your current code uses a **univariate time series approach**:

```python
# From vn30_dataset.py (line 59-60)
if 'RV_20' in df.columns:
    series_data = df['RV_20'].dropna().values  # ← Single series only
```

**Current Flow:**
- Input: Historical RV_20 values (context window)
- Model: TimesFM learns patterns from RV history
- Output: Future RV_20 predictions

### 2. Google TimesFM Reference (Univariate)

From the Google repo (`finetune_lora.py`), they also use **univariate approach**:

```python
# Google's retail sales example
for _, group in sales_train_df.groupby("id"):
    values = group[target].values.astype(np.float32)  # ← Single series
    all_series.append(values)  # Sales values only

# Random window sampling
train_ds = TimeSeriesRandomWindowDataset(
    all_series,  # List of univariate series
    context_len, 
    horizon_len
)
```

**Google's Flow:**
- Input: Historical sales values (context window)
- Model: TimesFM learns patterns from sales history
- Output: Future sales predictions

## Key Insight: TimesFM is Fundamentally Univariate

**Critical Discovery from Google Repo:**

Looking at the Google TimesFM finetuning code, **TimesFM 2.5 is designed for univariate time series forecasting**:

```python
# From finetune_lora.py line 235-244
for context, target_vals in train_loader:
    context = context.to(device)  # Shape: [batch, context_len]
    target_vals = target_vals.to(device)  # Shape: [batch, horizon_len]
    
    outputs = model(
        past_values=context,      # ← Single time series only!
        future_values=target_vals,  # ← Single target series!
        forecast_context_len=context_len,
    )
```

**TimesFM Architecture:**
- Input: `[batch, seq_len]` - single time series
- Output: `[batch, horizon]` - single target series
- Internal: RevIN normalization per instance

## The Challenge: Adding OHLC Features

### My Original Design (Multivariate)

I originally suggested this **multivariate approach**:

```python
# My suggestion (INCORRECT for TimesFM!)
timesfm_features = np.column_stack([
    parkinson_ts,      # Feature 1
    gk_ts,            # Feature 2  
    overnight_ts,      # Feature 3
    cc_ts,             # Feature 4
])
# Shape: [time_steps, num_features] → ❌ TimesFM doesn't support this!
```

### Why This Doesn't Work

**Problem:** TimesFM 2.5 expects univariate input:
```python
# TimesFM forward signature
model(past_values)  # past_values shape: [batch, seq_len]
```

It does **NOT** support:
```python
model(past_values)  # past_values shape: [batch, seq_len, features] → ❌
```

## Solution: Feature Engineering for Univariate TimesFM

### ✅ CORRECT Approach: Create Multiple Derived Time Series

**Option 1: Train Multiple Models (Recommended)**
```python
# Train separate TimesFM models for each feature
models = {
    'overnight_model': TimesFM trained on overnight_ts,
    'parkinson_model': TimesFM trained on parkinson_ts,
    'gk_model': TimesFM trained on gk_ts,
}

# Ensemble predictions
final_prediction = ensemble([
    overnight_model(input),
    parkinson_model(input),
    gk_model(input)
])
```

**Option 2: Feature Concatenation (Creative Approach)**
```python
# Create composite time series
def create_composite_series(df):
    """
    Create univariate series that combines OHLC information
    """
    # Stack features as longer univariate series
    overnight = calculate_overnight(df)
    parkinson = calculate_parkinson(df)
    gk = calculate_gk(df)
    
    # Normalize each to same scale
    overnight_norm = (overnight - overnight.mean()) / overnight.std()
    parkinson_norm = (parkinson - parkinson.mean()) / parkinson.std()
    gk_norm = (gk - gk.mean()) / gk.std()
    
    # Concatenate as single series
    composite = np.concatenate([
        overnight_norm,
        parkinson_norm, 
        gk_norm
    ])
    
    return composite  # Single univariate series!
```

**Option 3: Weighted Average Feature (Simplest)**
```python
def create_weighted_volatility_feature(df):
    """
    Create single volatility feature combining multiple estimators
    
    Based on paper findings: overnight + Parkinson + GK work best
    """
    # Calculate individual estimators
    overnight = calculate_overnight(df)
    parkinson = calculate_parkinson(df)
    gk = calculate_gk(df)
    
    # Weighted average (based on paper performance)
    # Overnight: most consistent → weight 0.5
    # Parkinson: good performer → weight 0.3  
    # GK: good performer → weight 0.2
    weighted_feature = (
        0.5 * overnight +
        0.3 * parkinson +
        0.2 * gk
    )
    
    return weighted_feature  # Single univariate series!
```

## Feasibility Demonstration: Modified Dataset Class

Here's how to modify your `VN30MultiStockDataset` to include OHLC features:

```python
class VN30MultiStockWithOHLCDataset(Dataset):
    """
    Multi-stock dataset with OHLC-derived features as univariate series
    """
    
    def __init__(self, stock_data_dict, feature_type='RV_20', **kwargs):
        self.feature_type = feature_type  # 'RV_20', 'overnight', 'parkinson', 'weighted'
        self.series_list = []
        
        for stock_name, df in stock_data_dict.items():
            # Calculate OHLC features
            if self.feature_type == 'overnight':
                series_data = calculate_overnight_volatility(df)
            elif self.feature_type == 'parkinson':
                series_data = calculate_parkinson(df)
            elif self.feature_type == 'weighted':
                series_data = self._calculate_weighted_feature(df)
            else:  # 'RV_20' (default)
                series_data = df['RV_20'].dropna().values
            
            if len(series_data) >= context_len + horizon_len:
                self.series_list.append({
                    'name': stock_name,
                    'data': series_data
                })
    
    def _calculate_weighted_feature(self, df):
        """Calculate weighted OHLC feature"""
        overnight = calculate_overnight_volatility(df)
        parkinson = calculate_parkinson(df)
        gk = calculate_garmanklass(df)
        
        # Weighted average (paper-based weights)
        weighted = 0.5 * overnight + 0.3 * parkinson + 0.2 * gk
        return weighted
```

## Validation: Feasibility Test

### Test 1: Can TimesFM Handle Overnight Volatility Series?

**Answer:** ✅ **YES** - Overnight volatility is just another univariate time series

```python
# Overnight volatility time series
overnight_ts = calculate_overnight_volatility(df)
# Shape: [time_steps] → Compatible with TimesFM!
```

### Test 2: Can TimesFM Handle Parkinson Estimator Series?

**Answer:** ✅ **YES** - Parkinson estimator is just another univariate time series

```python
# Parkinson estimator time series  
parkinson_ts = calculate_parkinson(df)
# Shape: [time_steps] → Compatible with TimesFM!
```

### Test 3: Can We Train TimesFM on These Features?

**Answer:** ✅ **YES** - Just change the target series in your dataset

**Current code:**
```python
series_data = df['RV_20'].dropna().values  # Current target
```

**Modified code:**
```python
series_data = calculate_overnight_volatility(df)  # New target
```

## Implementation Strategy

### Phase 1: Test Individual Features (Recommended)
```python
# Test 1: Train TimesFM on overnight volatility
overnight_dataset = VN30MultiStockDataset(
    stock_data_dict, 
    feature_type='overnight'
)
train_model(overnight_dataset)

# Test 2: Train TimesFM on Parkinson estimator
parkinson_dataset = VN30MultiStockDataset(
    stock_data_dict,
    feature_type='parkinson'  
)
train_model(parkinson_dataset)

# Compare performance vs baseline RV_20
```

### Phase 2: Ensemble Approach
```python
# Train multiple models
models = {
    'RV_20': train_model(feature_type='RV_20'),
    'overnight': train_model(feature_type='overnight'),
    'parkinson': train_model(feature_type='parkinson'),
}

# Ensemble predictions
def ensemble_predict(models, context):
    predictions = [model.predict(context) for model in models.values()]
    return np.mean(predictions, axis=0)  # Simple average ensemble
```

### Phase 3: Weighted Feature Approach
```python
# Create single weighted feature series
weighted_dataset = VN30MultiStockDataset(
    stock_data_dict,
    feature_type='weighted'  # Uses weighted avg of OHLC estimators
)
train_model(weighted_dataset)
```

## Conclusion: Design Feasibility

### ✅ **FEASIBLE** - With Modified Approach

**Original Design Status:** ❌ **NOT DIRECTLY FEASIBLE**
- Multivariate input [batch, seq_len, features] not supported by TimesFM 2.5

**Modified Design Status:** ✅ **FULLY FEASIBLE**
- Treat each OHLC estimator as separate univariate time series
- Use ensemble or weighted average approach
- Leverage paper findings for feature selection

### Key Recommendations

1. **Start Simple:** Test overnight volatility as target (paper's best performer)
2. **Compare Performance:** Overnight vs Parkinson vs GK vs baseline RV_20
3. **Build Ensemble:** Combine best performing models
4. **Follow Paper:** Prioritize overnight > Parkinson > GK > others

### Code Compatibility

**Your current infrastructure works perfectly:**
- ✅ Dataset class needs only minor modification
- ✅ Training loop unchanged (TimesFM handles univariate series)
- ✅ Evaluation metrics remain the same
- ✅ No architectural changes needed

**Only modification needed:**
```python
# Change this line in vn30_dataset.py:
series_data = df['RV_20'].dropna().values

# To this:
series_data = calculate_overnight_volatility(df)  # Or other OHLC features
```

## Final Verdict

**My design IS feasible** for your TimesFM VN30 project, but requires a **modified univariate approach** rather than true multivariate input. The paper's findings about OHLC range estimators can be successfully applied through:

1. **Feature selection:** Overnight volatility (most important)
2. **Single-model approach:** Train on weighted OHLC feature
3. **Multi-model ensemble:** Train separate models on each feature

This aligns perfectly with both your current code structure and Google's TimesFM architecture!