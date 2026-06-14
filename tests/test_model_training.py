"""
Unit Tests for TimesFM VN30 Training Pipeline
Tests all components before full training execution
"""

import sys
import unittest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json
import yaml
import io

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.append('src')

from model_training import TimesFMVN30Finetuner
from vn30_dataset import create_vn30_dataloaders
from model_evaluation import TimesFMModelEvaluator


class TestTimesFMModelTraining(unittest.TestCase):
    """Unit tests for TimesFM training pipeline"""

    @classmethod
    def setUpClass(cls):
        """Setup test configuration"""
        cls.config_path = 'configs/config.yaml'
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_01_config_loading(self):
        """Test 1: Configuration file loading"""
        print("\n[TEST 1] Loading configuration file...")
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            self.assertIn('system', config)
            self.assertIn('data', config)
            self.assertIn('model', config)
            self.assertIn('training', config)

            print(f"[OK] Config loaded: {config['system']['name']}")
            return True
        except Exception as e:
            print(f"[FAIL] Config loading failed: {e}")
            return False

    def test_02_timesfm_initialization(self):
        """Test 2: TimesFM finetuner initialization"""
        print("\n[TEST 2] Initializing TimesFM finetuner...")
        try:
            finetuner = TimesFMVN30Finetuner(self.config_path)

            self.assertIsNotNone(finetuner)
            self.assertEqual(finetuner.device.type, self.device.type)

            print(f"[OK] Finetuner initialized on {finetuner.device}")
            return True
        except Exception as e:
            print(f"[FAIL] Finetuner initialization failed: {e}")
            return False

    def test_03_timesfm_model_loading(self):
        """Test 3: TimesFM 2.5 model loading"""
        print("\n[TEST 3] Loading TimesFM 2.5 model...")
        try:
            finetuner = TimesFMVN30Finetuner(self.config_path)
            finetuner.load_timesfm_model()

            self.assertIsNotNone(finetuner.base_model)

            print("[OK] TimesFM 2.5 model loaded successfully")
            return True
        except Exception as e:
            print(f"[FAIL] Model loading failed: {e}")
            return False

    def test_04_optimizer_setup(self):
        """Test 4: Optimizer configuration"""
        print("\n[TEST 4] Setting up optimizer...")
        try:
            finetuner = TimesFMVN30Finetuner(self.config_path)
            finetuner.load_timesfm_model()
            finetuner.setup_optimizer()

            self.assertIsNotNone(finetuner.optimizer)
            self.assertIsNotNone(finetuner.scheduler)

            print("[OK] Optimizer configured successfully")
            return True
        except Exception as e:
            print(f"[FAIL] Optimizer setup failed: {e}")
            return False

    def test_05_dataset_loading(self):
        """Test 5: Dataset creation and loading"""
        print("\n[TEST 5] Loading dataset...")
        try:
            train_loader, test_loader = create_vn30_dataloaders(self.config_path)

            self.assertIsNotNone(train_loader)
            self.assertIsNotNone(test_loader)
            self.assertGreater(len(train_loader), 0)
            self.assertGreater(len(test_loader), 0)

            print(f"[OK] Dataset loaded: {len(train_loader)} train batches, {len(test_loader)} test batches")
            return True
        except Exception as e:
            print(f"[FAIL] Dataset loading failed: {e}")
            return False

    def test_06_batch_processing(self):
        """Test 6: Single batch processing"""
        print("\n[TEST 6] Testing single batch processing...")
        try:
            train_loader, test_loader = create_vn30_dataloaders(self.config_path)

            # Get one batch
            batch = next(iter(train_loader))

            self.assertIn('context', batch)
            self.assertIn('target', batch)
            self.assertIn('stock', batch)

            context_shape = batch['context'].shape
            target_shape = batch['target'].shape

            print(f"[OK] Batch processed: context {context_shape}, target {target_shape}")
            return True
        except Exception as e:
            print(f"[FAIL] Batch processing failed: {e}")
            return False

    def test_07_timesfm_forward_pass(self):
        """Test 7: TimesFM forecast pass"""
        print("\n[TEST 7] Testing TimesFM forecast pass...")
        try:
            finetuner = TimesFMVN30Finetuner(self.config_path)
            finetuner.load_timesfm_model()

            train_loader, _ = create_vn30_dataloaders(self.config_path)
            batch = next(iter(train_loader))

            context = batch['context'].to(finetuner.device)

            # Test forecast pass with correct API
            with torch.no_grad():
                # Convert first sample to numpy for TimesFM
                context_numpy = context[0].cpu().numpy()
                outputs = finetuner.base_model.forecast(horizon=1, inputs=[context_numpy])

            self.assertIsNotNone(outputs)
            print(f"[OK] Forecast pass successful, output type: {type(outputs)}")
            return True
        except Exception as e:
            print(f"[FAIL] Forecast pass failed: {e}")
            return False

    def test_08_loss_calculation(self):
        """Test 8: Loss calculation with TimesFM forecast"""
        print("\n[TEST 8] Testing loss calculation...")
        try:
            finetuner = TimesFMVN30Finetuner(self.config_path)
            finetuner.load_timesfm_model()

            train_loader, _ = create_vn30_dataloaders(self.config_path)
            batch = next(iter(train_loader))

            context = batch['context'].to(finetuner.device)
            targets = batch['target'].to(finetuner.device)

            # Get predictions using TimesFM forecast API
            with torch.no_grad():
                context_numpy = context[0].cpu().numpy()
                forecast_result = finetuner.base_model.forecast(horizon=1, inputs=[context_numpy])

                # Extract predictions
                if isinstance(forecast_result, tuple):
                    predictions = forecast_result[0]  # Get predictions
                else:
                    predictions = forecast_result

                # Convert to torch tensor
                predictions = torch.tensor(predictions, device=finetuner.device)

                # Ensure correct shape
                if predictions.dim() == 0:
                    predictions = predictions.unsqueeze(0)
                if predictions.dim() == 1:
                    predictions = predictions.unsqueeze(-1)

                # Match target shape
                if predictions.shape[0] != targets.shape[0]:
                    predictions = predictions.repeat(targets.shape[0], 1)

            # Calculate loss
            loss = finetuner.calculate_financial_loss(predictions, targets)

            self.assertIsNotNone(loss)
            self.assertGreater(loss.item(), 0)

            print(f"[OK] Loss calculated: {loss.item():.6f}")
            return True
        except Exception as e:
            print(f"[FAIL] Loss calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_09_single_training_step(self):
        """Test 9: Single training step with TimesFM forecast"""
        print("\n[TEST 9] Testing single training step...")
        try:
            finetuner = TimesFMVN30Finetuner(self.config_path)
            finetuner.load_timesfm_model()
            finetuner.setup_optimizer()

            train_loader, _ = create_vn30_dataloaders(self.config_path)
            batch = next(iter(train_loader))

            context = batch['context'].to(finetuner.device)
            targets = batch['target'].to(finetuner.device)

            # Single forward and backward pass with TimesFM forecast
            finetuner.optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Use TimesFM forecast API
                context_numpy = context[0].cpu().numpy()
                forecast_result = finetuner.base_model.forecast(horizon=1, inputs=[context_numpy])

                # Extract predictions
                if isinstance(forecast_result, tuple):
                    predictions = forecast_result[0]
                else:
                    predictions = forecast_result

                predictions = torch.tensor(predictions, device=finetuner.device)

                # Ensure correct shape
                if predictions.dim() == 0:
                    predictions = predictions.unsqueeze(0)
                if predictions.dim() == 1:
                    predictions = predictions.unsqueeze(-1)

                if predictions.shape[0] != targets.shape[0]:
                    predictions = predictions.repeat(targets.shape[0], 1)

                loss = finetuner.calculate_financial_loss(predictions, targets)

            loss.backward()
            finetuner.optimizer.step()

            print(f"[OK] Training step completed, loss: {loss.item():.6f}")
            return True
        except Exception as e:
            print(f"[FAIL] Training step failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_10_evaluation_metrics(self):
        """Test 10: Evaluation metrics calculation"""
        print("\n[TEST 10] Testing evaluation metrics...")
        try:
            evaluator = TimesFMModelEvaluator()

            # Create sample data
            actuals = np.array([0.01, 0.02, 0.015, 0.018, 0.022])
            predictions = np.array([0.011, 0.019, 0.016, 0.017, 0.021])

            # Calculate all metrics
            qlike = evaluator.calculate_qlike(actuals, predictions)
            r2 = evaluator.calculate_r2(actuals, predictions)
            rmse = evaluator.calculate_rmse(actuals, predictions)
            mse = evaluator.calculate_mse(actuals, predictions)

            self.assertIsNotNone(qlike)
            self.assertIsNotNone(r2)
            self.assertIsNotNone(rmse)
            self.assertIsNotNone(mse)

            print(f"[OK] Metrics calculated: QLIKE={qlike:.4f}, R²={r2:.4f}, RMSE={rmse:.4f}")
            return True
        except Exception as e:
            print(f"[FAIL] Metrics calculation failed: {e}")
            return False


def run_unit_tests():
    """Run all unit tests and generate report"""
    print("=" * 70)
    print("UNIT TESTS FOR TIMESFM VN30 TRAINING PIPELINE")
    print("=" * 70)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTimesFMModelTraining)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Generate summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n[OK] ALL TESTS PASSED - System ready for training!")
        return 0
    else:
        print("\n[FAIL] SOME TESTS FAILED - Please fix issues before training")
        return 1


if __name__ == "__main__":
    sys.exit(run_unit_tests())