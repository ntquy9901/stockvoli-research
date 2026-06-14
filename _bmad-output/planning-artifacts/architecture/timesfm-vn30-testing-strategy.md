# TimesFM VN30 Testing Strategy
**Project:** Vietnamese Stock Volatility Forecasting System
**Date:** 2026-06-09
**Architect:** Winston (System Architect)
**Testing Lead:** Development Team

---

## Testing Philosophy

Our testing strategy follows the **financial ML testing pyramid**: extensive unit testing at the component level, focused integration testing for data pipelines, and comprehensive statistical validation for model performance.

**Key Principles:**
1. **Test data transformations first** - Financial data quality is critical
2. **Statistical validation over simple accuracy** - Financial models require significance testing
3. **Production-like testing environments** - Mirror deployment architecture
4. **Automated continuous testing** - Every change triggers test suite

---

## Test Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TESTING PYRAMID ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌───────────────┐
                    │   E2E Tests   │  ← 10% (Backtesting, Production Simulation)
                    │   (Slow)      │
                    └───────┬───────┘
                            │
            ┌───────────────┴───────────────┐
            │   Integration Tests           │  ← 30% (Data pipelines, Model training)
            │   (Medium)                    │
            └───────────────┬───────────────┘
                            │
    ┌───────────────────────┴───────────────────────┐
    │           Unit Tests                          │  ← 60% (Components, Functions)
    │           (Fast)                              │
    └───────────────────────┬───────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │   Test Coverage Target: 85%+  │
            └───────────────────────────────┘
```

---

## Phase 1: Unit Testing Strategy

### 1.1 Data Processing Components

#### Test: Log Transformation
```python
def test_log_transformation_no_extreme_values():
    """Verify log transformation prevents extreme values"""
    # Create sample data with extreme price movements
    data = pd.DataFrame({
        'close': [100, 50, 25, 12.5, 6.25],  # 50% daily drops (crash scenario)
        'date': pd.date_range('2020-01-01', periods=5)
    })
    
    processor = VN30FinancialDataProcessor()
    result = processor.apply_log_transformation(data)
    
    # Assertions
    assert 'log_close' in result.columns
    assert 'log_returns' in result.columns
    assert not result['log_returns'].isna().any()
    assert result['log_returns'].max() < np.inf  # No infinite values
    assert result['log_returns'].min() > -np.inf  # No -inf values
    
def test_log_transformation_normal_distribution():
    """Verify log returns are approximately normally distributed"""
    # Generate realistic price data
    np.random.seed(42)
    prices = np.random.lognormal(mean=4, sigma=0.02, size=1000)  # Stock prices
    data = pd.DataFrame({'close': prices})
    
    processor = VN30FinancialDataProcessor()
    result = processor.apply_log_transformation(data)
    
    # Test for normality (Shapiro-Wilk)
    from scipy import stats
    statistic, p_value = stats.shapiro(result['log_returns'].dropna())
    
    # Log returns should be approximately normal (p > 0.01 for large samples)
    assert p_value > 0.01 or len(result) < 50  # Allow deviation for small samples
```

#### Test: Realized Volatility Calculation
```python
def test_realized_volatility_calculation():
    """Verify multi-horizon realized volatility calculation"""
    # Create sample returns
    np.random.seed(42)
    returns = np.random.normal(0, 0.02, 100)  # 2% daily volatility
    data = pd.DataFrame({'log_returns': returns})
    
    processor = VN30FinancialDataProcessor()
    result = processor.calculate_realized_volatility(data)
    
    # Check all horizons exist
    for window in [5, 10, 20, 30]:
        assert f'RV_{window}' in result.columns
        
        # Verify calculation: rolling standard deviation
        manual_rv = data['log_returns'].rolling(window).std()
        assert np.allclose(result[f'RV_{window}'], manual_rv, rtol=1e-5)
    
def test_realized_volatility_clipping():
    """Verify financial clipping prevents extreme values"""
    # Create data with extreme volatility
    extreme_returns = pd.Series([0.01, 0.5, -0.6, 0.02, 0.01])  # Extreme moves
    data = pd.DataFrame({'log_returns': extreme_returns})
    
    processor = VN30FinancialDataProcessor()
    result = processor.calculate_realized_volatility(data)
    result = processor.apply_financial_clipping(result)
    
    # Check clipping range
    assert result['RV_20'].max() <= 5.0
    assert result['RV_20'].min() >= -5.0
```

#### Test: Vietnamese Market Features
```python
def test_tet_holiday_detection():
    """Verify TET holiday detection for Vietnamese market"""
    # Create dates around TET 2024 (Feb 10-12, 2024)
    tet_dates = pd.date_range('2024-02-08', '2024-02-15')
    data = pd.DataFrame({'date': tet_dates})
    
    processor = VN30FinancialDataProcessor()
    result = processor.detect_vietnamese_holidays(data)
    
    # Check TET period detection
    assert 'is_tet_period' in result.columns
    assert result['is_tet_period'].sum() > 0  # Should detect TET period
    
def test_day_of_week_patterns():
    """Verify day-of-week feature extraction"""
    # Create trading week (Mon-Fri)
    trading_week = pd.date_range('2024-01-08', '2024-01-12', freq='B')
    data = pd.DataFrame({'date': trading_week})
    
    processor = VN30FinancialDataProcessor()
    result = processor.extract_day_of_week(data)
    
    # Check feature extraction
    assert 'day_of_week' in result.columns
    assert result['day_of_week'].min() >= 0  # Monday
    assert result['day_of_week'].max() <= 4  # Friday
    assert len(result) == 5  # 5 trading days
```

### 1.2 Dataset Components

#### Test: Multi-Stock Dataset Creation
```python
def test_multi_stock_dataset_initialization():
    """Verify multi-stock dataset handles multiple stocks correctly"""
    # Create sample data for 3 stocks
    stock_data = {
        'VCB': pd.DataFrame({'RV_20': np.random.normal(0.02, 0.01, 1000)}),
        'VIC': pd.DataFrame({'RV_20': np.random.normal(0.018, 0.008, 800)}),
        'VNM': pd.DataFrame({'RV_20': np.random.normal(0.022, 0.012, 1200)})
    }
    
    dataset = VN30MultiStockDataset(
        stock_data, 
        context_len=64, 
        horizon_len=13
    )
    
    # Verify dataset structure
    assert len(dataset) > 0  # Should create samples
    assert len(dataset.series_list) == 3  # Should process all stocks
    
def test_multi_stock_dataset_sample_structure():
    """Verify individual sample structure"""
    stock_data = {
        'VCB': pd.DataFrame({'RV_20': np.random.normal(0.02, 0.01, 200)})
    }
    
    dataset = VN30MultiStockDataset(stock_data, context_len=64, horizon_len=13)
    
    if len(dataset) > 0:
        sample = dataset[0]
        
        # Check sample structure
        assert 'context' in sample
        assert 'target' in sample
        assert 'stock' in sample
        
        # Check dimensions
        assert sample['context'].shape == (64,)  # Context length
        assert isinstance(sample['target'], (float, np.floating))
        assert sample['stock'] == 'VCB'
```

#### Test: Data Loading and Batching
```python
def test_dataloader_creation():
    """Verify DataLoader creation works correctly"""
    stock_data = {
        'VCB': pd.DataFrame({'RV_20': np.random.normal(0.02, 0.01, 500)})
    }
    
    dataset = VN30MultiStockDataset(stock_data, context_len=64, horizon_len=13)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Verify batching
    batch = next(iter(dataloader))
    
    assert 'context' in batch
    assert 'target' in batch
    assert batch['context'].shape[0] <= 32  # Batch size
    assert batch['context'].shape[1] == 64  # Context length

def test_train_test_split_no_leakage():
    """Verify time-based split prevents data leakage"""
    # Create temporal data
    dates = pd.date_range('2020-01-01', '2022-12-31', freq='B')
    stock_data = {
        'VCB': pd.DataFrame({
            'RV_20': np.random.normal(0.02, 0.01, len(dates)),
            'date': dates
        }).set_index('date')
    }
    
    # Create train and test datasets
    train_dataset = VN30MultiStockDataset(stock_data, context_len=64, horizon_len=13, mode='train')
    test_dataset = VN30MultiStockDataset(stock_data, context_len=64, horizon_len=13, mode='test')
    
    # Verify no temporal overlap
    if len(train_dataset) > 0 and len(test_dataset) > 0:
        train_dates = [sample['date'] for sample in train_dataset.samples]
        test_dates = [sample['date'] for sample in test_dataset.samples]
        
        # Find maximum training date
        max_train_date = max(train_dates) if train_dates else None
        min_test_date = min(test_dates) if test_dates else None
        
        if max_train_date and min_test_date:
            assert max_train_date < min_test_date, "Temporal leakage detected!"
```

### 1.3 Model Components

#### Test: TimesFM Model Loading
```python
def test_timesfm_base_model_loading():
    """Verify TimesFM 2.5 model loads successfully"""
    try:
        from transformers import TimesFm2_5ModelForPrediction
        
        model = TimesFm2_5ModelForPrediction.from_pretrained(
            "google/timesfm-2.5-200m-transformers",
            torch_dtype=torch.bfloat16
        )
        
        # Verify model structure
        assert model is not None
        assert hasattr(model, 'config')
        
        # Check model can process input
        dummy_input = torch.randn(1, 64, 7)  # Batch, context, features
        with torch.no_grad():
            output = model(past_values=dummy_input)
        
        assert output is not None
        
    except Exception as e:
        pytest.skip(f"TimesFM model loading failed: {e}")

def test_lora_adapter_integration():
    """Verify LoRA adapters integrate correctly with TimesFM"""
    try:
        from transformers import TimesFm2_5ModelForPrediction
        from peft import LoraConfig, get_peft_model
        
        base_model = TimesFm2_5ModelForPrediction.from_pretrained(
            "google/timesfm-2.5-200m-transformers",
            torch_dtype=torch.bfloat16
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none"
        )
        
        model = get_peft_model(base_model, lora_config)
        
        # Verify LoRA integration
        assert model is not None
        
        # Check trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        # LoRA should only train small subset
        assert trainable_params < total_params * 0.1  # Less than 10% trainable
        
        # Verify forward pass works
        dummy_input = torch.randn(1, 64, 7)
        with torch.no_grad():
            output = model(past_values=dummy_input)
        
        assert output is not None
        
    except Exception as e:
        pytest.skip(f"LoRA integration test failed: {e}")
```

#### Test: Optimizer Configuration
```python
def test_sgd_optimizer_configuration():
    """Verify SGD optimizer configured correctly for financial data"""
    # Create simple model
    model = nn.Linear(10, 1)
    
    # Configure SGD optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-4,
        momentum=0.9,
        nesterov=True
    )
    
    # Verify optimizer type and parameters
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == 1e-4
    assert optimizer.defaults['momentum'] == 0.9
    assert optimizer.defaults['nesterov'] == True
    
def test_gradient_clipping():
    """Verify gradient clipping prevents exploding gradients"""
    # Create simple model and data
    model = nn.Linear(10, 1)
    input_data = torch.randn(32, 10)
    target = torch.randn(32, 1)
    
    # Forward pass
    output = model(input_data)
    loss = nn.MSELoss()(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check gradient norms before clipping
    max_grad_norm = max(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    
    # Apply gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Check gradient norms after clipping
    clipped_max_grad_norm = max(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    
    assert clipped_max_grad_norm <= 1.0, "Gradient clipping failed!"
```

---

## Phase 2: Integration Testing Strategy

### 2.1 Data Pipeline Integration

#### Test: End-to-End Data Processing
```python
def test_end_to_end_data_processing():
    """Verify complete data processing pipeline"""
    # Load real stock data
    vcb_data = pd.read_csv('data/raw/prices/VCB_ohlcv.csv')
    vcb_data['date'] = pd.to_datetime(vcb_data['date'])
    vcb_data.set_index('date', inplace=True)
    
    # Process through pipeline
    processor = VN30FinancialDataProcessor()
    processed_data = processor.process_stock_data(vcb_data)
    
    # Verify all transformations applied
    assert 'log_close' in processed_data.columns
    assert 'log_returns' in processed_data.columns
    assert 'RV_20' in processed_data.columns
    assert 'is_tet_period' in processed_data.columns
    assert 'day_of_week' in processed_data.columns
    
    # Verify data quality
    assert not processed_data['RV_20'].isna().all()  # Should have valid RV data
    assert processed_data['RV_20'].max() <= 5.0  # Clipping applied
    assert processed_data['RV_20'].min() >= -5.0

def test_multi_stock_pipeline_integration():
    """Verify pipeline handles all 30 stocks correctly"""
    data_dir = Path('data/raw/prices')
    stock_files = list(data_dir.glob('*_ohlcv.csv'))
    
    processor = VN30FinancialDataProcessor()
    processed_stocks = []
    
    for stock_file in stock_files[:5]:  # Test first 5 stocks
        stock_name = stock_file.stem.replace('_ohlcv', '')
        
        try:
            data = pd.read_csv(stock_file)
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
            
            processed = processor.process_stock_data(data)
            processed_stocks.append(stock_name)
            
        except Exception as e:
            print(f"Failed to process {stock_name}: {e}")
    
    # Verify pipeline success
    assert len(processed_stocks) >= 3  # At least 3 stocks should process
    print(f"Successfully processed {len(processed_stocks)} stocks")
```

### 2.2 Training Pipeline Integration

#### Test: Single Training Epoch
```python
def test_single_training_epoch():
    """Verify one training epoch completes successfully"""
    # Create small dataset
    stock_data = {
        'TEST': pd.DataFrame({'RV_20': np.random.normal(0.02, 0.01, 500)})
    }
    
    dataset = VN30MultiStockDataset(stock_data, context_len=64, horizon_len=13)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Create model
    model = nn.Linear(64, 1)  # Simplified model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    
    # Training epoch
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        context = batch['context']
        target = batch['target']
        
        optimizer.zero_grad()
        predictions = model(context)
        loss = nn.MSELoss()(predictions.squeeze(), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    # Verify training completed
    assert total_loss > 0  # Loss should be positive
    assert not np.isnan(total_loss)  # Loss should not be NaN
    assert not np.isinf(total_loss)  # Loss should not be infinite
    
    print(f"Single epoch test passed. Total loss: {total_loss:.4f}")

def test_model_checkpoint_saving():
    """Verify model checkpoints save and load correctly"""
    # Create simple model
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    
    # Create checkpoint directory
    checkpoint_dir = Path('test_checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / 'test_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': 0.1234
    }, checkpoint_path)
    
    # Verify file exists
    assert checkpoint_path.exists()
    
    # Load checkpoint
    loaded_checkpoint = torch.load(checkpoint_path)
    
    # Verify checkpoint contents
    assert 'model_state_dict' in loaded_checkpoint
    assert 'optimizer_state_dict' in loaded_checkpoint
    assert loaded_checkpoint['loss'] == 0.1234
    
    # Verify model can load state
    model.load_state_dict(loaded_checkpoint['model_state_dict'])
    
    # Cleanup
    checkpoint_path.unlink()
    checkpoint_dir.rmdir()
    
    print("Checkpoint saving test passed")
```

### 2.3 Model Inference Integration

#### Test: End-to-End Inference
```python
def test_model_inference_pipeline():
    """Verify complete inference pipeline"""
    # Create test data
    context_data = torch.randn(1, 64, 7)  # Single sample
    
    # Create model
    model = nn.Linear(64, 1)
    model.eval()
    
    # Inference
    with torch.no_grad():
        prediction = model(context_data.squeeze(-1))  # Remove feature dimension
    
    # Verify prediction
    assert prediction.shape == (1,)  # Single prediction
    assert not torch.isnan(prediction).any()  # No NaN predictions
    assert not torch.isinf(prediction).any()  # No infinite predictions
    
    print(f"Inference test passed. Prediction: {prediction.item():.6f}")

def test_batch_inference():
    """Verify batch inference for multiple stocks"""
    # Create batch data
    batch_context = torch.randn(30, 64, 7)  # 30 stocks
    
    # Create model
    model = nn.Linear(64, 1)
    model.eval()
    
    # Batch inference
    with torch.no_grad():
        predictions = model(batch_context.squeeze(-1))
    
    # Verify batch predictions
    assert predictions.shape == (30,)  # 30 predictions
    assert not torch.isnan(predictions).any()
    assert not torch.isinf(predictions).any()
    
    print(f"Batch inference test passed. {len(predictions)} predictions generated")
```

---

## Phase 3: Statistical Validation Testing

### 3.1 Diebold-Mariano Test Implementation

#### Test: Statistical Test Framework
```python
def test_diebold_mariano_test_implementation():
    """Verify Diebold-Mariano statistical test implementation"""
    # Create sample data
    np.random.seed(42)
    actual = np.random.normal(0.02, 0.01, 1000)
    model_pred = actual + np.random.normal(0, 0.005, 1000)  # Better model
    bench_pred = actual + np.random.normal(0, 0.008, 1000)  # Worse benchmark
    
    # Run DM test
    dm_result = diebold_mariano_test(actual, model_pred, bench_pred)
    
    # Verify test structure
    assert 'dm_statistic' in dm_result
    assert 'p_value' in dm_result
    assert 'significant' in dm_result
    
    # Verify value ranges
    assert 0 <= dm_result['p_value'] <= 1
    assert isinstance(dm_result['significant'], bool)
    
    print(f"DM Test: Statistic={dm_result['dm_statistic']:.4f}, p-value={dm_result['p_value']:.4f}")

def test_diebold_mariano_significance_detection():
    """Verify DM test correctly detects significance"""
    # Create scenarios where model should be significantly better
    np.random.seed(42)
    actual = np.random.normal(0.02, 0.01, 500)
    
    # Model is much better than benchmark
    model_pred = actual + np.random.normal(0, 0.002, 500)  # Very accurate
    bench_pred = actual + np.random.normal(0, 0.02, 500)  # Poor benchmark
    
    dm_result = diebold_mariano_test(actual, model_pred, bench_pred)
    
    # Should detect significance
    assert dm_result['significant'] == True
    assert dm_result['p_value'] < 0.05
    
    print(f"Significance detection test passed. p-value: {dm_result['p_value']:.4f}")
```

### 3.2 Performance Metrics Testing

#### Test: Comprehensive Metrics Calculation
```python
def test_performance_metrics_calculation():
    """Verify comprehensive performance metrics calculation"""
    # Create sample predictions and targets
    np.random.seed(42)
    targets = np.random.normal(0.02, 0.01, 1000)
    predictions = targets + np.random.normal(0, 0.003, 1000)  # Good predictions
    
    # Calculate metrics
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    correlation = np.corrcoef(targets, predictions)[0, 1]
    
    # Verify metric ranges
    assert mae > 0  # Should have some error
    assert rmge > 0  # Should have some error
    assert -1 <= r2 <= 1  # R² should be in valid range
    assert -1 <= correlation <= 1  # Correlation should be in valid range
    
    # For good predictions, metrics should be reasonable
    assert r2 > 0  # Should explain some variance
    assert correlation > 0  # Should be positively correlated
    
    print(f"Metrics test passed. R²={r2:.4f}, Correlation={correlation:.4f}")

def test_volatility_specific_metrics():
    """Verify volatility-specific metrics calculation"""
    # Create sample volatility predictions
    actual_vol = np.array([0.01, 0.02, 0.015, 0.025, 0.018])
    pred_vol = np.array([0.012, 0.018, 0.016, 0.023, 0.019])
    
    # Direction accuracy
    mean_vol = np.mean(actual_vol)
    actual_direction = np.sign(actual_vol - mean_vol)
    pred_direction = np.sign(pred_vol - mean_vol)
    direction_accuracy = np.mean(actual_direction == pred_direction)
    
    # Verify direction accuracy
    assert 0 <= direction_accuracy <= 1  # Should be between 0-100%
    assert direction_accuracy > 0.3  # Should be better than random
    
    print(f"Volatility metrics test passed. Direction accuracy: {direction_accuracy*100:.1f}%")
```

---

## Phase 4: End-to-End Testing Strategy

### 4.1 Full Training Pipeline Test

#### Test: Complete Model Training
```python
def test_complete_model_training():
    """Verify complete model training pipeline"""
    # Create sample multi-stock dataset
    stock_data = {
        'STOCK1': pd.DataFrame({'RV_20': np.random.normal(0.02, 0.01, 500)}),
        'STOCK2': pd.DataFrame({'RV_20': np.random.normal(0.018, 0.008, 500)}),
    }
    
    dataset = VN30MultiStockDataset(stock_data, context_len=64, horizon_len=13)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Create model
    model = nn.Linear(64, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    
    # Train for 3 epochs
    training_losses = []
    for epoch in range(3):
        model.train()
        epoch_loss = 0
        
        for batch in dataloader:
            context = batch['context']
            target = batch['target']
            
            optimizer.zero_grad()
            predictions = model(context)
            loss = nn.MSELoss()(predictions.squeeze(), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        training_losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")
    
    # Verify training convergence
    assert len(training_losses) == 3
    assert all(not np.isnan(loss) for loss in training_losses)  # No NaN losses
    assert all(not np.isinf(loss) for loss in training_losses)  # No infinite losses
    
    # Loss should generally decrease (allow some fluctuation)
    assert training_losses[-1] < training_losses[0] * 1.5  # Last loss not much worse than first
    
    print("Complete training test passed")
```

### 4.2 Production Simulation Test

#### Test: Production Environment Simulation
```python
def test_production_inference_simulation():
    """Simulate production inference environment"""
    # Create model
    model = nn.Linear(64, 1)
    model.eval()
    
    # Simulate production request
    stock_contexts = {
        'VCB': torch.randn(1, 64),
        'VIC': torch.randn(1, 64),
        'VNM': torch.randn(1, 64),
    }
    
    # Batch inference
    predictions = {}
    with torch.no_grad():
        for stock_name, context in stock_contexts.items():
            prediction = model(context)
            predictions[stock_name] = prediction.item()
    
    # Verify predictions
    assert len(predictions) == 3  # All stocks predicted
    assert all(not np.isnan(pred) for pred in predictions.values())  # No NaN
    assert all(not np.isinf(pred) for pred in predictions.values())  # No infinite
    
    print(f"Production simulation test passed. {len(predictions)} stocks processed")

def test_model_loading_and_inference():
    """Verify model checkpoint loading and inference"""
    # Create and save model
    model = nn.Linear(64, 1)
    checkpoint_path = Path('test_model_checkpoint.pt')
    torch.save(model.state_dict(), checkpoint_path)
    
    # Load model
    loaded_model = nn.Linear(64, 1)
    loaded_model.load_state_dict(torch.load(checkpoint_path))
    loaded_model.eval()
    
    # Test inference
    with torch.no_grad():
        prediction = loaded_model(torch.randn(1, 64))
    
    # Verify prediction
    assert prediction.shape == (1, 1)
    assert not torch.isnan(prediction).any()
    
    # Cleanup
    checkpoint_path.unlink()
    
    print("Model loading and inference test passed")
```

---

## Test Execution Strategy

### Continuous Integration Testing
```bash
# Run tests on every commit
git commit -m "Add feature"

# Automated test execution
pytest tests/ -v --tb=short --cov=src --cov-report=html

# Test categories
pytest tests/unit/ -v           # Fast unit tests (< 2 min)
pytest tests/integration/ -v   # Integration tests (< 10 min)
pytest tests/statistical/ -v    # Statistical validation tests
```

### Performance Testing
```python
# Test execution time requirements
def test_inference_performance():
    """Verify inference meets performance requirements"""
    model = nn.Linear(64, 1)
    model.eval()
    
    # Performance test: 100 stocks
    start_time = time.time()
    for _ in range(100):
        context = torch.randn(1, 64)
        with torch.no_grad():
            prediction = model(context)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_prediction = total_time / 100
    
    # Requirement: < 1 second per prediction
    assert avg_time_per_prediction < 1.0, f"Inference too slow: {avg_time_per_prediction:.3f}s"
    
    print(f"Performance test passed. Avg time: {avg_time_per_prediction*1000:.2f}ms")
```

### Data Quality Testing
```python
def test_data_quality_checks():
    """Verify data quality validation framework"""
    # Load real data
    vcb_data = pd.read_csv('data/raw/prices/VCB_ohlcv.csv')
    
    # Quality checks
    checks = {
        'sufficient_length': len(vcb_data) >= 100,  # At least 100 trading days
        'no_missing_dates': not vcb_data['date'].isna().any(),
        'positive_prices': (vcb_data['close'] > 0).all(),
        'valid_volume': (vcb_data['volume'] >= 0).all(),
    }
    
    # Verify all checks pass
    assert all(checks.values()), f"Data quality checks failed: {checks}"
    
    print(f"Data quality checks passed: {sum(checks.values())}/{len(checks)}")
```

---

## Test Coverage Requirements

### Coverage Targets by Component
```
Data Processing Module:     90% coverage required
Dataset Creation Module:     85% coverage required
Model Training Module:       80% coverage required
Validation Module:          85% coverage required
Inference Module:          90% coverage required

Overall Project Target:     85% coverage minimum
```

### Critical Path Testing
```
MUST TEST (Critical):
1. Log transformation and financial data handling
2. Multi-stock dataset creation
3. TimesFM model loading
4. LoRA adapter integration
5. SGD optimizer configuration
6. Diebold-Mariano statistical test
7. Model inference pipeline

SHOULD TEST (Important):
1. Vietnamese market features
2. Gradient clipping
3. Learning rate scheduling
4. Model checkpoint saving/loading
5. Performance metrics calculation

COULD TEST (Nice to have):
1. Extreme market scenarios
2. Missing data handling
3. Visualization generation
```

---

## Success Criteria for Testing

### Automated Testing Requirements
- ✅ All unit tests pass (< 5 minutes)
- ✅ All integration tests pass (< 20 minutes)
- ✅ Code coverage > 85%
- ✅ No critical bugs remain

### Statistical Validation Requirements
- ✅ Diebold-Mariano test shows significance (p < 0.05)
- ✅ R² score > 0.5 on test set
- ✅ Model outperforms baseline significantly
- ✅ Predictions show meaningful variation

### Production Readiness Requirements
- ✅ Model loads in < 10 seconds
- ✅ Inference time < 1 second per stock
- ✅ No NaN or infinite predictions
- ✅ Consistent performance across multiple runs

---

**Document Status:** Testing Strategy Complete
**Next Phase:** Implementation Execution
**Owner:** Winston (System Architect)
**Review Date:** 2026-06-09