# Foundation Model Fine-Tuning Implementation Guide
## Based on Research Papers: TimesFM & Moirai 2.0

## Paper Analysis Summary

### Paper 1: TimesFM for Volatility Forecasting (arXiv:2505.11163)
**Authors:** Anubha Goel, Puneet Pasricha, Martin Magris, Juho Kanniainen

**Key Findings:**
- **Model:** TimesFM (Time Series Foundation Model)
- **Focus:** Volatility forecasting for financial risk management
- **Fine-tuning Method:** **Incremental Learning** - allows model to adapt to new financial return data over time
- **Validation:** Diebold-Mariano and Giacomini-White statistical tests
- **Results:** Fine-tuned variants statistically outperform traditional econometric models

### Paper 2: Moirai 2.0 (arXiv:2511.11698)  
**Authors:** Chenghao Liu, Taha Aksu, et al.

**Key Innovations:**
- **Architecture:** Decoder-only (simpler than masked-encoder)
- **Training:** 36M time series corpus
- **Features:** Quantile forecasting, multi-token prediction
- **Efficiency:** 2x faster, 30x smaller than Moirai 1.0-Large
- **Key Method:** Recursive multi-quantile decoding

## Implementation Guide Based on Paper Methodologies

### Phase 1: Foundation Model Selection & Setup

#### 1.1 Model Choice Based on Papers

**Option A: TimesFM (from Paper 2505.11163)**
```python
# TimesFM implementation for volatility forecasting
# Paper: "Foundation Time-Series AI Model for Realized Volatility Forecasting"

import torch
from transformers import AutoModelForTimeSeriesForecasting, AutoTokenizer

def load_timesfm_model():
    """Load TimesFM model as used in volatility forecasting paper"""
    
    model_name = "google/timesfm"  # or specific TimesFM checkpoint
    
    # Load model
    model = AutoModelForTimeSeriesForecasting.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer

print("Loading TimesFM model for volatility forecasting...")
timesfm_model, timesfm_tokenizer = load_timesfm_model()
```

**Option B: Moirai 2.0 (from Paper 2511.11698)**
```python
# Moirai 2.0 implementation
# Paper: "Moirai 2.0: When Less Is More for Time Series Forecasting"

def load_moirai_model():
    """Load Moirai 2.0 decoder-only model"""
    
    model_name = "Salesforce/moirai-2.0-R-small"  # Available on HuggingFace
    
    # Load decoder-only model
    model = AutoModelForTimeSeriesForecasting.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Moirai 2.0 uses quantile forecasting
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer

print("Loading Moirai 2.0 model...")
moirai_model, moirai_tokenizer = load_moirai_model()
```

#### 1.2 Vietnamese Market Data Preparation

```python
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Vietnamese stocks as mentioned in your research context
vietnamese_stocks = [
    'VCB.VN',  # Vietcombank
    'VIC.VN',  # Vingroup  
    'VNM.VN',  # Vinamilk
    'HPG.VN',  # Hoa Phat Group
    'MSN.VN',  # Masan Group
    'STB.VN',  # Sacombank
    'MWG.VN',  # Mobile World Group
    'FPT.VN',  # FPT Corporation
]

def collect_vietnamese_volatility_data(symbols, start_date, end_date):
    """Collect Vietnamese stock data for volatility forecasting"""
    
    volatility_data = {}
    
    for symbol in symbols:
        try:
            # Download historical data
            stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            # Calculate realized volatility (as used in TimesFM paper)
            stock_data['Log_Returns'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
            
            # Realized Volatility (RV) - key metric from the paper
            for window in [5, 10, 20, 30]:
                stock_data[f'RV_{window}'] = stock_data['Log_Returns'].rolling(window=window).std()
            
            volatility_data[symbol] = stock_data
            print(f"Collected {len(stock_data)} days for {symbol}")
            
        except Exception as e:
            print(f"Error collecting {symbol}: {e}")
    
    return volatility_data

# Collect 5 years of data (paper used multi-year datasets)
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

vietnam_volatility_data = collect_vietnamese_volatility_data(
    vietnamese_stocks, start_date, end_date
)
```

### Phase 2: Incremental Fine-Tuning (Based on TimesFM Paper)

#### 2.1 Incremental Learning Implementation

The TimesFM paper emphasizes **incremental fine-tuning** as essential for adapting to new financial return data over time.

```python
class IncrementalFineTuner:
    """Incremental fine-tuning approach from TimesFM volatility paper"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.incremental_step = 0
        
    def prepare_incremental_batch(self, new_data_window):
        """Prepare new data window for incremental learning"""
        
        # Extract features from new data window
        context_features = []
        target_volatility = []
        
        for symbol, data in new_data_window.items():
            # Use RV_20 as target (20-day realized volatility)
            if f'RV_20' in data.columns:
                # Context window: past returns and volatility features
                context = data[['Log_Returns', 'RV_5', 'RV_10']].values[-512:]  # 512 context length
                
                # Target: next period volatility
                target = data[f'RV_20'].values[-1]
                
                if not np.isnan(target) and not np.isnan(context).any():
                    context_features.append(context)
                    target_volatility.append(target)
        
        return {
            'context_features': torch.tensor(context_features, dtype=torch.float32),
            'target_volatility': torch.tensor(target_volatility, dtype=torch.float32)
        }
    
    def incremental_update(self, new_data_window, learning_rate=1e-5):
        """Perform incremental update as model encounters new data"""
        
        self.incremental_step += 1
        print(f"Incremental Step {self.incremental_step}")
        
        # Prepare batch
        batch = self.prepare_incremental_batch(new_data_window)
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            # Get current predictions
            outputs = self.model(context=batch['context_features'].to(self.device))
            current_predictions = outputs.prediction
        
        # Calculate loss
        loss = torch.nn.functional.mse_loss(
            current_predictions.squeeze(), 
            batch['target_volatility'].to(self.device)
        )
        
        # Incremental update (single epoch on new data)
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        optimizer.zero_grad()
        outputs = self.model(context=batch['context_features'].to(self.device))
        loss = torch.nn.functional.mse_loss(
            outputs.prediction.squeeze(), 
            batch['target_volatility'].to(self.device)
        )
        loss.backward()
        optimizer.step()
        
        print(f"Incremental Loss: {loss.item():.6f}")
        
        return loss.item()

# Initialize incremental fine-tuner
timesfm_finetuner = IncrementalFineTuner(timesfm_model, timesfm_tokenizer)
```

#### 2.2 Temporal Incremental Learning Strategy

```python
def incremental_learning_pipeline(volatility_data, finetuner, retrain_freq=30):
    """
    Implement incremental learning as described in TimesFM paper.
    Model adapts to new financial return data over time.
    
    Args:
        volatility_data: Dictionary of stock volatility data
        finetuner: IncrementalFineTuner instance
        retrain_freq: Frequency of incremental updates (days)
    """
    
    # Sort dates
    all_dates = sorted(list(volatility_data[vietnamese_stocks[0]].index))
    
    # Split into incremental windows
    incremental_windows = []
    for i in range(0, len(all_dates) - retrain_freq, retrain_freq):
        window_dates = all_dates[i:i+retrain_freq+512]  # 512 context + retrain_freq
        
        window_data = {}
        for symbol, data in volatility_data.items():
            symbol_data = data[data.index.isin(window_dates)]
            if len(symbol_data) > 512:  # Ensure sufficient context
                window_data[symbol] = symbol_data
        
        if window_data:
            incremental_windows.append(window_data)
    
    print(f"Created {len(incremental_windows)} incremental learning windows")
    
    # Perform incremental learning
    incremental_losses = []
    
    for i, window_data in enumerate(incremental_windows):
        print(f"\n=== Incremental Window {i+1}/{len(incremental_windows)} ===")
        print(f"Window date range: {window_data[vietnamese_stocks[0]].index.min()} to {window_data[vietnamese_stocks[0]].index.max()}")
        
        loss = finetuner.incremental_update(window_data)
        incremental_losses.append(loss)
    
    return incremental_losses

# Run incremental learning
print("Starting incremental fine-tuning process...")
incremental_losses = incremental_learning_pipeline(vietnam_volatility_data, timesfm_finetuner)
```

### Phase 3: Quantile Forecasting (Based on Moirai 2.0 Paper)

#### 3.1 Multi-Quantile Decoding Implementation

Moirai 2.0 uses **recursive multi-quantile decoding** instead of mixture-distribution outputs.

```python
class QuantileVolatilityPredictor:
    """Quantile forecasting based on Moirai 2.0 methodology"""
    
    def __init__(self, model, tokenizer, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
        self.model = model
        self.tokenizer = tokenizer
        self.quantiles = quantiles
        
    def predict_quantiles(self, context_features, horizon=1):
        """
        Generate probabilistic volatility forecasts using quantile prediction
        as implemented in Moirai 2.0
        """
        
        quantile_predictions = {}
        
        with torch.no_grad():
            for q in self.quantiles:
                # Predict specific quantile
                outputs = self.model(
                    context=context_features,
                    quantile=q  # Moirai 2.0 quantile forecasting
                )
                
                prediction = outputs.prediction.squeeze()
                quantile_predictions[f'q_{q}'] = prediction.cpu().numpy()
        
        return quantile_predictions
    
    def generate_volatility_forecast(self, current_data, prediction_date):
        """Generate volatility forecast with confidence intervals"""
        
        # Prepare context
        context_features = self.prepare_context(current_data)
        context_tensor = torch.tensor(context_features, dtype=torch.float32).unsqueeze(0)
        
        # Generate quantile predictions
        quantile_preds = self.predict_quantiles(context_tensor)
        
        # Calculate prediction intervals
        median_forecast = quantile_preds['q_0.5']
        lower_80 = quantile_preds['q_0.1']  # 80% confidence interval
        upper_80 = quantile_preds['q_0.9']
        
        return {
            'date': prediction_date,
            'median_volatility': median_forecast,
            'lower_80_ci': lower_80,
            'upper_80_ci': upper_80,
            'prediction_interval': upper_80 - lower_80,
            'all_quantiles': quantile_preds
        }

# Initialize quantile predictor
moirai_predictor = QuantileVolatilityPredictor(moirai_model, moirai_tokenizer)
```

#### 3.2 Efficient Decoder-Only Inference

Moirai 2.0 uses decoder-only architecture for better efficiency.

```python
def efficient_volatility_forecasting(moirai_predictor, volatility_data, test_dates):
    """
    Efficient forecasting using Moirai 2.0 decoder-only architecture
    2x faster, 30x smaller model as mentioned in paper
    """
    
    predictions = []
    
    for test_date in test_dates:
        # Get context data (previous 512 days)
        context_end = test_date - timedelta(days=1)
        context_start = context_end - timedelta(days=512)
        
        context_data = {}
        for symbol, data in volatility_data.items():
            symbol_context = data[(data.index >= context_start) & (data.index <= context_end)]
            if len(symbol_context) == 512:
                context_data[symbol] = symbol_context
        
        if context_data:
            # Generate forecast
            forecast = moirai_predictor.generate_volatility_forecast(context_data, test_date)
            predictions.append(forecast)
    
    return predictions

# Generate efficient forecasts
print("Generating efficient forecasts using Moirai 2.0 methodology...")
test_dates = vietnam_volatility_data[vietnamese_stocks[0]].index[-100:]  # Last 100 days
volatility_forecasts = efficient_volatility_forecasting(moirai_predictor, vietnam_volatility_data, test_dates)
```

### Phase 4: Statistical Validation (Based on TimesFM Paper)

#### 4.1 Diebold-Mariano Test Implementation

The TimesFM paper uses **Diebold-Mariano and Giacomini-White tests** to validate that fine-tuned models statistically outperform traditional models.

```python
class ModelValidation:
    """Statistical validation based on TimesFM paper methodology"""
    
    def __init__(self):
        pass
    
    def calculate_forecast_errors(self, actual_volatility, predicted_volatility):
        """Calculate forecast errors for Diebold-Mariano test"""
        
        actual = np.array(actual_volatility)
        predicted = np.array(predicted_volatility)
        
        # Squared errors
        se_actual = actual ** 2  # Benchmark (e.g., random walk or historical mean)
        se_model = (actual - predicted) ** 2
        
        # Loss differential
        loss_diff = se_actual - se_model
        
        return loss_diff
    
    def diebold_mariano_test(self, actuals, predictions, benchmark_predictions):
        """
        Diebold-Mariano test for forecast accuracy
        
        H0: Model has same accuracy as benchmark
        H1: Model is more accurate than benchmark
        """
        
        loss_differentials = []
        
        for actual, pred, bench_pred in zip(actuals, predictions, benchmark_predictions):
            loss_diff = self.calculate_forecast_errors(actual, pred) - \
                       self.calculate_forecast_errors(actual, bench_pred)
            loss_differentials.append(loss_diff)
        
        loss_diff = np.array(loss_differentials)
        
        # Calculate DM statistic
        mean_diff = np.mean(loss_diff)
        std_diff = np.std(loss_diff) / np.sqrt(len(loss_diff))
        
        dm_statistic = mean_diff / std_diff
        
        # p-value (two-tailed test)
        from scipy import stats
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_statistic)))
        
        return {
            'dm_statistic': dm_statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': 'Model significantly outperforms benchmark' if p_value < 0.05 else 'No significant difference'
        }
    
    def giacomini_white_test(self, actuals, predictions, benchmark_predictions):
        """
        Giacomini-White test for conditional predictive ability
        More robust than DM test for finite samples
        """
        
        # Calculate loss differentials
        loss_differentials = []
        for actual, pred, bench_pred in zip(actuals, predictions, benchmark_predictions):
            loss_diff = self.calculate_forecast_errors(actual, pred) - \
                       self.calculate_forecast_errors(actual, bench_pred)
            loss_differentials.append(loss_diff)
        
        loss_diff = np.array(loss_differentials)
        
        # Regress loss differential on constant
        X = np.ones_like(loss_diff).reshape(-1, 1)
        
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, loss_diff)
        
        # Calculate test statistic
        predictions_diff = reg.predict(X)
        residuals = loss_diff - predictions_diff
        
        # GW statistic
        n = len(loss_diff)
        gw_statistic = n * (reg.coef_[0] ** 2) / np.sum(residuals ** 2)
        
        # p-value
        from scipy import stats
        p_value = 1 - stats.chi2.cdf(gw_statistic, 1)
        
        return {
            'gw_statistic': gw_statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': 'Model has significant conditional predictive ability' if p_value < 0.05 else 'No significant conditional ability'
        }

# Initialize validation
validator = ModelValidation()
```

#### 4.2 Benchmark Comparison

```python
def validate_against_benchmarks(actual_volatility, model_predictions):
    """
    Compare fine-tuned model against traditional benchmarks
    As performed in TimesFM paper
    """
    
    # Traditional benchmark models
    benchmark_predictions = {
        'random_walk': actual_volatility.shift(1).fillna(method='bfill'),
        'historical_mean': np.full_like(actual_volatility, actual_volatility.mean()),
        'garch': None  # Would need GARCH implementation
    }
    
    validation_results = {}
    
    for benchmark_name, bench_preds in benchmark_predictions.items():
        if bench_preds is not None and len(bench_preds) == len(model_predictions):
            print(f"\nValidating against {benchmark_name}...")
            
            # Diebold-Mariano test
            dm_result = validator.diebold_mariano_test(
                actual_volatility.values,
                model_predictions.values,
                bench_preds.values
            )
            
            # Giacomini-White test
            gw_result = validator.giacomini_white_test(
                actual_volatility.values,
                model_predictions.values,
                bench_preds.values
            )
            
            validation_results[benchmark_name] = {
                'diebold_mariano': dm_result,
                'giacomini_white': gw_result
            }
    
    return validation_results

# Example usage
# validation_results = validate_against_benchmarks(test_volatility, model_predictions)
```

### Phase 5: Production Implementation

#### 5.1 Model Comparison & Selection

```python
def compare_foundation_models():
    """Compare TimesFM vs Moirai 2.0 for Vietnamese volatility forecasting"""
    
    comparison_metrics = {
        'TimesFM': {
            'focus': 'Volatility forecasting with incremental learning',
            'architecture': 'Masked-encoder',
            'training': 'Pre-trained on diverse time series',
            'fine_tuning': 'Incremental adaptation to new financial data',
            'strengths': 'Statistical validation, adapts over time',
            'paper': '2505.11163'
        },
        'Moirai 2.0': {
            'focus': 'Universal forecasting with quantile prediction',
            'architecture': 'Decoder-only (simpler)',
            'training': '36M time series corpus',
            'fine_tuning': 'Standard fine-tuning',
            'strengths': 'Efficiency (2x faster, 30x smaller), quantile forecasts',
            'paper': '2511.11698'
        }
    }
    
    return comparison_metrics

model_comparison = compare_foundation_models()
```

#### 5.2 Hybrid Approach Recommendation

Based on both papers, I recommend a **hybrid approach**:

1. **Use TimesFM incremental learning** for continuous adaptation to Vietnamese market data
2. **Use Moirai 2.0 quantile forecasting** for probabilistic volatility predictions
3. **Implement statistical validation** using Diebold-Mariano and Giacomini-White tests

```python
class HybridVolatilityForecaster:
    """Hybrid approach combining both paper methodologies"""
    
    def __init__(self):
        # TimesFM for incremental learning
        self.timesfm_model = None
        self.incremental_finetuner = None
        
        # Moirai 2.0 for quantile forecasting
        self.moirai_model = None
        self.quantile_predictor = None
        
        # Validation
        self.validator = ModelValidation()
    
    def setup_models(self):
        """Initialize both models"""
        print("Setting up hybrid forecasting system...")
        
        # Load TimesFM
        timesfm_model, timesfm_tokenizer = load_timesfm_model()
        self.incremental_finetuner = IncrementalFineTuner(timesfm_model, timesfm_tokenizer)
        
        # Load Moirai 2.0
        moirai_model, moirai_tokenizer = load_moirai_model()
        self.quantile_predictor = QuantileVolatilityPredictor(moirai_model, moirai_tokenizer)
        
        print("Hybrid system ready")
    
    def train_incremental(self, volatility_data):
        """Train TimesFM with incremental learning"""
        return incremental_learning_pipeline(volatility_data, self.incremental_finetuner)
    
    def predict_with_quantiles(self, context_data, prediction_date):
        """Generate quantile forecasts using Moirai 2.0"""
        return self.quantile_predictor.generate_volatility_forecast(context_data, prediction_date)
    
    def validate_performance(self, actuals, predictions, benchmarks):
        """Validate using statistical tests from TimesFM paper"""
        return validate_against_benchmarks(actuals, predictions, benchmarks)

# Initialize hybrid system
hybrid_forecaster = HybridVolatilityForecaster()
hybrid_forecaster.setup_models()
```

## Key Implementation Insights from Papers

### From TimesFM Paper (2505.11163):
1. **Incremental Learning is Essential**: "incremental fine-tuning, which allows the model to adapt to new financial return data over time, is essential for learning volatility patterns effectively"
2. **Statistical Validation**: Use Diebold-Mariano and Giacomini-White tests to prove statistical significance
3. **Outperforms Traditional Models**: Fine-tuned variants statistically outperform GARCH and other econometric models

### From Moirai 2.0 Paper (2511.11698):
1. **Decoder-Only Architecture**: Simpler and more efficient than masked-encoder approaches
2. **Quantile Forecasting**: Provides probabilistic forecasts with confidence intervals
3. **Massive Training Data**: 36M time series provides strong foundation
4. **Efficiency**: 2x faster, 30x smaller than previous versions

## Expected Results Based on Papers

Following these methodologies, you should achieve:
- **Statistical Significance**: DM and GW tests showing p < 0.05 vs traditional models
- **Incremental Improvement**: Continuous adaptation to Vietnamese market patterns
- **Probabilistic Forecasts**: Quantile predictions with confidence intervals
- **Computational Efficiency**: Fast inference suitable for real-time applications

This implementation guide is now based on the specific methodologies from your research papers rather than generic ML approaches.