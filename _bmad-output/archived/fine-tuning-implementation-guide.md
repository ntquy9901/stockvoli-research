# Foundation Model Fine-Tuning Implementation Guide
## Vietnamese Stock Volatility Prediction

## Overview

This guide provides a practical, step-by-step approach to fine-tuning foundation models for predicting stock volatility in Vietnamese markets. The implementation combines state-of-the-art ML techniques with financial domain expertise.

## Prerequisites

### Hardware Requirements
- **GPU:** NVIDIA A100 (40GB) or H100 (80GB) recommended
- **RAM:** 128GB+ system memory
- **Storage:** 2TB+ NVMe SSD for data and models
- **CPU:** 32+ cores for data processing

### Software Requirements
```bash
# Core ML Framework
pip install torch>=2.0.0 transformers>=4.35.0

# Fine-tuning Libraries
pip install peft>=0.6.0 bitsandbytes>=0.41.0

# Data Processing
pip install pandas numpy scikit-learn

# Financial Data
pip install yfinance pandas-ta

# Experiment Tracking
pip install mlflow wandb

# Utilities
pip install tqdm wandb
```

## Step-by-Step Fine-Tuning Process

### Phase 1: Data Preparation (Weeks 1-2)

#### 1.1 Data Collection
```python
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Vietnamese Stock Symbols
vietnamese_stocks = [
    'VCB.VN',  # Vietcombank
    'VIC.VN',  # Vingroup
    'VNM.VN',  # Vinamilk
    'HPG.VN',  # Hoa Phat Group
    'MSN.VN',  # Masan Group
    # Add more Vietnamese stocks
]

def collect_vietnamese_stock_data(symbols, start_date, end_date):
    """Collect historical data for Vietnamese stocks"""
    data = {}
    
    for symbol in symbols:
        try:
            # Download historical data
            stock_data = yf.download(symbol, start=start_date, end=end_date)
            
            # Calculate basic features
            stock_data['Returns'] = stock_data['Adj Close'].pct_change()
            stock_data['Log_Returns'] = np.log(stock_data['Adj Close']/stock_data['Adj Close'].shift(1))
            stock_data['Volatility'] = stock_data['Returns'].rolling(window=20).std()
            
            data[symbol] = stock_data
            print(f"Collected {len(stock_data)} days of data for {symbol}")
            
        except Exception as e:
            print(f"Error collecting {symbol}: {e}")
    
    return data

# Collect 5 years of historical data
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)
stock_data = collect_vietnamese_stock_data(vietnamese_stocks, start_date, end_date)
```

#### 1.2 Feature Engineering
```python
import pandas_ta as ta
import numpy as np

def create_technical_features(df):
    """Create technical analysis features"""
    df = df.copy()
    
    # Moving Averages
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['EMA_12'] = ta.ema(df['Close'], length=12)
    df['EMA_26'] = ta.ema(df['Close'], length=26)
    
    # MACD
    df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
    df['MACD_Signal'] = ta.macd(df['Close'])['MACDs_12_26_9']
    
    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # Bollinger Bands
    bb = ta.bbands(df['Close'], length=20)
    df['BB_Upper'] = bb['BBU_20_2.0']
    df['BB_Middle'] = bb['BBM_20_2.0']
    df['BB_Lower'] = bb['BBL_20_2.0']
    
    # ATR (Average True Range) - volatility indicator
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    
    # Price momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(5)
    
    return df

def create_volatility_features(df):
    """Create volatility-specific features"""
    df = df.copy()
    
    # Realized volatility (different time windows)
    for window in [5, 10, 20, 40]:
        df[f'RV_{window}'] = df['Returns'].rolling(window=window).std()
        df[f'RV_{window}_MA'] = df[f'RV_{window}'].rolling(window=10).mean()
    
    # GARCH-like features
    df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
    df['Vol_Spike'] = df['Returns'].abs() > df['RV_20'].mean() + 2*df['RV_20'].std()
    
    # Trend indicators
    df['Trend'] = np.where(df['SMA_20'] > df['SMA_50'], 1, -1) if 'SMA_50' in df.columns else 0
    
    return df

# Apply feature engineering
for symbol, data in stock_data.items():
    stock_data[symbol] = create_technical_features(data)
    stock_data[symbol] = create_volatility_features(stock_data[symbol])
```

#### 1.3 Train/Validation/Test Split
```python
def create_time_series_splits(data_dict, test_size=0.2, val_size=0.1):
    """Create temporal train/validation/test splits"""
    
    # Calculate split points
    all_dates = set()
    for data in data_dict.values():
        all_dates.update(data.index)
    
    all_dates = sorted(list(all_dates))
    n_total = len(all_dates)
    
    # Calculate split indices
    test_start = int(n_total * (1 - test_size))
    val_start = int(n_total * (1 - test_size - val_size))
    
    train_dates = all_dates[:val_start]
    val_dates = all_dates[val_start:test_start]
    test_dates = all_dates[test_start:]
    
    print(f"Train period: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
    print(f"Val period: {val_dates[0]} to {val_dates[-1]} ({len(val_dates)} days)")
    print(f"Test period: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")
    
    return train_dates, val_dates, test_dates

train_dates, val_dates, test_dates = create_time_series_splits(stock_data)
```

### Phase 2: Model Preparation (Week 3)

#### 2.1 Select Base Foundation Model
```python
from transformers import (
    AutoModelForTimeSeriesForecasting,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
import torch
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
from bitsandbytes import BitsAndBytesConfig

# Model options for time series:
# 1. Chronos (Amazon) - Pre-trained on time series
# 2. TimeGPT (Nixtla) - Foundation model for time series
# 3. Lag-Llama - Open-source time series foundation model
# 4. Custom Transformer trained on financial data

MODEL_NAME = "amazon/chronos-t5-small"  # Start with smaller model

def load_base_model(model_name):
    """Load base foundation model with quantization"""
    
    # Configure 4-bit quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load model
    model = AutoModelForTimeSeriesForecasting.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

print("Loading base foundation model...")
base_model, tokenizer = load_base_model(MODEL_NAME)
print(f"Model loaded with {sum(p.numel() for p in base_model.parameters()):,} parameters")
```

#### 2.2 Configure LoRA Adapters
```python
def setup_lora_model(model, target_modules=None):
    """Setup LoRA adapters for efficient fine-tuning"""
    
    if target_modules is None:
        # Default target modules for transformer models
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    lora_config = LoraConfig(
        task_type=TaskType.TIME_SERIES_FORECASTING,
        inference_mode=False,
        r=16,  # LoRA rank (higher = more parameters to train)
        lora_alpha=32,  # LoRA scaling factor
        lora_dropout=0.1,
        target_modules=target_modules,
        modules_to_save=["decoder"],  # Additional modules to train
    )
    
    # Get LoRA model
    lora_model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in lora_model.parameters())
    
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    return lora_model

lora_model = setup_lora_model(base_model)
```

### Phase 3: Data Preparation for Fine-Tuning (Week 4)

#### 3.1 Create Custom Dataset
```python
import torch
from torch.utils.data import Dataset

class VolatilityPredictionDataset(Dataset):
    """Custom dataset for volatility prediction"""
    
    def __init__(self, data_dict, dates, feature_columns, context_length=512, prediction_length=1):
        self.data_dict = data_dict
        self.dates = dates
        self.feature_columns = feature_columns
        self.context_length = context_length
        self.prediction_length = prediction_length
        
        # Prepare samples
        self.samples = self._prepare_samples()
    
    def _prepare_samples(self):
        """Prepare training samples from data"""
        samples = []
        
        for symbol, data in self.data_dict.items():
            symbol_dates = [d for d in self.dates if d in data.index]
            
            for i in range(len(symbol_dates) - self.context_length - self.prediction_length + 1):
                context_dates = symbol_dates[i:i+self.context_length]
                target_date = symbol_dates[i+self.context_length]
                
                if target_date in data.index and not data.loc[target_date, 'Volatility'] != data.loc[target_date, 'Volatility']:
                    sample = {
                        'symbol': symbol,
                        'context_dates': context_dates,
                        'target_date': target_date,
                        'features': data.loc[context_dates, self.feature_columns].values,
                        'target': data.loc[target_date, 'Volatility']
                    }
                    samples.append(sample)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get features and target
        features = torch.tensor(sample['features'], dtype=torch.float32)
        target = torch.tensor(sample['target'], dtype=torch.float32)
        
        return {
            'context': features,
            'target': target,
            'symbol': sample['symbol'],
            'dates': sample['context_dates']
        }

# Define feature columns
feature_columns = [
    'Returns', 'Log_Returns', 'Volatility',
    'SMA_20', 'EMA_12', 'EMA_26', 'MACD', 'RSI',
    'ATR', 'Momentum', 'RV_5', 'RV_10', 'RV_20'
]

# Filter available columns
available_features = [col for col in feature_columns if col in stock_data[vietnamese_stocks[0]].columns]

# Create datasets
train_dataset = VolatilityPredictionDataset(
    stock_data, train_dates, available_features, context_length=256, prediction_length=1
)
val_dataset = VolatilityPredictionDataset(
    stock_data, val_dates, available_features, context_length=256, prediction_length=1
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
```

#### 3.2 Custom Training Functions
```python
def compute_volatility_metrics(predictions, targets):
    """Compute volatility prediction metrics"""
    
    # Mean Absolute Error
    mae = torch.abs(predictions - targets).mean()
    
    # Root Mean Squared Error
    rmse = torch.sqrt(((predictions - targets) ** 2).mean())
    
    # Mean Absolute Percentage Error
    mape = torch.abs((targets - predictions) / (targets + 1e-8)).mean() * 100
    
    return {
        'mae': mae.item(),
        'rmse': rmse.item(),
        'mape': mape.item()
    }

def custom_trainer(model, train_dataset, val_dataset, output_dir, num_epochs=10):
    """Custom training loop with MLflow tracking"""
    
    import mlflow
    from torch.utils.data import DataLoader
    from transformers import AdamW, get_linear_schedule_with_warmup
    
    # Start MLflow run
    mlflow.start_run()
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Setup learning rate scheduler
    num_training_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_metrics = {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0}
        
        # Training
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                context=batch['context'].to(model.device),
                target=batch['target'].to(model.device)
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            # Calculate metrics
            with torch.no_grad():
                predictions = outputs.prediction.squeeze()
                metrics = compute_volatility_metrics(predictions, batch['target'].to(model.device))
                for key, value in metrics.items():
                    train_metrics[key] += value
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Average training metrics
        train_loss /= len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_metrics = {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0}
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    context=batch['context'].to(model.device),
                    target=batch['target'].to(model.device)
                )
                
                loss = outputs.loss
                val_loss += loss.item()
                
                predictions = outputs.prediction.squeeze()
                metrics = compute_volatility_metrics(predictions, batch['target'].to(model.device))
                for key, value in metrics.items():
                    val_metrics[key] += value
        
        val_loss /= len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)
        
        # Log metrics
        mlflow.log_metrics({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_mae': train_metrics['mae'],
            'val_mae': val_metrics['mae'],
            'train_rmse': train_metrics['rmse'],
            'val_rmse': val_metrics['rmse']
        }, step=epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, MAE: {train_metrics['mae']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, MAE: {val_metrics['mae']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(f"{output_dir}/best_model")
            tokenizer.save_pretrained(f"{output_dir}/best_model")
            print(f"Saved best model with val_loss: {val_loss:.4f}")
    
    mlflow.end_run()
    return model
```

### Phase 4: Fine-Tuning Execution (Weeks 5-6)

#### 4.1 Execute Fine-Tuning
```python
import os
from transformers import Trainer, TrainingArguments

# Setup training arguments
training_args = TrainingArguments(
    output_dir='./volatility_finetuning',
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    learning_rate=1e-4,
    fp16=True,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
)

# Create output directory
os.makedirs(training_args.output_dir, exist_ok=True)

# Start fine-tuning
print("Starting fine-tuning process...")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Context length: 256, Prediction length: 1")

fine_tuned_model = custom_trainer(
    lora_model,
    train_dataset,
    val_dataset,
    training_args.output_dir,
    num_epochs=training_args.num_train_epochs
)

print("Fine-tuning completed!")
```

#### 4.2 Model Evaluation
```python
def evaluate_model(model, test_dataset):
    """Evaluate fine-tuned model on test set"""
    from torch.utils.data import DataLoader
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_symbols = []
    
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                context=batch['context'].to(model.device),
                target=batch['target'].to(model.device)
            )
            
            predictions = outputs.prediction.squeeze()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch['target'].cpu().numpy())
            all_symbols.extend(batch['symbol'])
    
    # Calculate metrics
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
    
    # Correlation
    correlation = np.corrcoef(predictions, targets)[0, 1]
    
    print(f"Test Set Evaluation:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Correlation: {correlation:.4f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'correlation': correlation,
        'predictions': predictions,
        'targets': targets,
        'symbols': all_symbols
    }

# Create test dataset
test_dataset = VolatilityPredictionDataset(
    stock_data, test_dates, available_features, context_length=256, prediction_length=1
)

# Evaluate model
test_results = evaluate_model(fine_tuned_model, test_dataset)
```

### Phase 5: Model Deployment (Weeks 7-8)

#### 5.1 Model Packaging
```python
def package_model_for_serving(model, tokenizer, output_dir):
    """Package model for production serving"""
    
    # Save fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Create serving configuration
    config = {
        'model_name': 'vietnam_volatility_predictor',
        'model_version': '1.0.0',
        'context_length': 256,
        'prediction_length': 1,
        'features': available_features,
        'target': 'Volatility',
        'threshold': test_results['mae'] * 1.5  # Alert threshold
    }
    
    import json
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Model packaged for serving: {output_dir}")
    
    return output_dir

# Package model
serving_dir = package_model_for_serving(fine_tuned_model, tokenizer, "./serving_model")
```

#### 5.2 Docker Container Creation
```dockerfile
# Dockerfile for model serving
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving_model ./model
COPY model_server.py .

EXPOSE 8080

CMD ["python", "model_server.py", "--model-path", "./model", "--port", "8080"]
```

#### 5.3 Model Server Implementation
```python
# model_server.py
import torch
from transformers import AutoModelForTimeSeriesForecasting, AutoTokenizer
from flask import Flask, request, jsonify
import numpy as np
import json

app = Flask(__name__)

class VolatilityPredictionServer:
    def __init__(self, model_path):
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = AutoModelForTimeSeriesForecasting.from_pretrained(
            model_path,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load configuration
        with open(f"{model_path}/config.json", 'r') as f:
            self.config = json.load(f)
        
        self.model.eval()
    
    def predict(self, features):
        """Make prediction on input features"""
        with torch.no_grad():
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            outputs = self.model(context=features_tensor)
            prediction = outputs.prediction.squeeze()
            
            return prediction.cpu().numpy()
    
    def health_check(self):
        """Health check endpoint"""
        return {
            'status': 'healthy',
            'model': self.config['model_name'],
            'version': self.config['model_version']
        }

# Initialize server
server = VolatilityPredictionServer("./model")

@app.route('/health', methods=['GET'])
def health():
    return jsonify(server.health_check())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data['features'])
        
        # Validate input
        if features.shape[1] != len(server.config['features']):
            return jsonify({'error': f'Expected {len(server.config["features"])} features'}), 400
        
        # Make prediction
        prediction = server.predict(features)
        
        response = {
            'prediction': float(prediction),
            'timestamp': data.get('timestamp'),
            'symbol': data.get('symbol'),
            'confidence': 'high' if prediction < server.config['threshold'] else 'low'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

### Phase 6: Monitoring & Maintenance (Ongoing)

#### 6.1 Performance Monitoring
```python
def setup_monitoring():
    """Setup monitoring for prediction service"""
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    
    # Metrics
    prediction_counter = Counter('volatility_predictions_total', 'Total predictions made')
    prediction_latency = Histogram('volatility_prediction_latency_seconds', 'Prediction latency')
    prediction_error = Gauge('volatility_prediction_error', 'Prediction error (MAE)')
    
    # Start Prometheus server
    start_http_server(8000)
    
    return {
        'counter': prediction_counter,
        'latency': prediction_latency,
        'error': prediction_error
    }

def log_predictions_to_mlflow(predictions, actuals, symbols):
    """Log prediction performance to MLflow"""
    import mlflow
    
    with mlflow.start_run():
        # Calculate metrics
        mae = np.mean(np.abs(predictions - actuals))
        
        mlflow.log_metric('daily_mae', mae)
        mlflow.log_text("symbols", str(symbols))
```

## Best Practices & Tips

### 1. Data Quality
- Always validate data quality before training
- Handle missing values appropriately
- Normalize features to prevent scale issues
- Monitor for data drift in production

### 2. Training Efficiency
- Use mixed precision training (fp16) for speed
- Enable gradient checkpointing for memory efficiency
- Use appropriate batch sizes for your GPU
- Monitor GPU utilization

### 3. Model Selection
- Start with smaller models for faster iteration
- Gradually increase model size if needed
- Consider ensemble methods for final production
- Always maintain baseline comparisons

### 4. Production Readiness
- Implement comprehensive monitoring
- Setup automated alerting
- Create rollback procedures
- Document model behavior and limitations

### 5. Vietnamese Market Specifics
- Consider local market hours and holidays
- Account for currency fluctuations
- Monitor local regulatory changes
- Include market-specific features

## Expected Results

With proper implementation, you should achieve:

- **MAE:** < 0.02 (2% volatility error)
- **RMSE:** < 0.03 (3% volatility error)
- **Correlation:** > 0.85 with actual volatility
- **Inference Time:** < 100ms per prediction
- **Model Size:** < 500MB with LoRA

## Next Steps

1. **Expand Coverage:** Add more Vietnamese stocks and indices
2. **Multi-Horizon:** Extend predictions to multiple time horizons
3. **Explainability:** Add SHAP values for model interpretability
4. **Ensemble:** Combine multiple models for improved accuracy
5. **Real-Time:** Implement streaming predictions for real-time updates

## Troubleshooting

### Common Issues

**Issue:** GPU memory errors during training
**Solution:** Reduce batch size, enable gradient checkpointing, use 8-bit training

**Issue:** Poor model convergence
**Solution:** Check data quality, adjust learning rate, increase training epochs

**Issue:** Overfitting on training data
**Solution:** Add regularization, increase dropout, use more training data

**Issue:** Slow inference in production
**Solution:** Use model quantization, optimize serving infrastructure, implement caching

This implementation guide provides a complete roadmap for fine-tuning foundation models for Vietnamese stock volatility prediction. Adjust parameters and approaches based on your specific requirements and data availability.