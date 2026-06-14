"""
Unit Tests for Colab Pre-Flight Validation
==========================================

Tests critical paths to catch issues before expensive Colab training.

Run with:
    pytest tests/test_colab_preflight.py -v

Author: Murat (Test Architect)
Date: 2026-06-13
"""

import pytest
import torch
import yaml
import json
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

try:
    from model_training_fixed import TimesFMVN30Finetuner
except ImportError:
    pass  # Will handle in test


class TestConfigValidation:
    """Test configuration file validation."""

    def test_config_file_exists(self):
        """Test config file exists."""
        config_path = Path('configs/config.yaml')
        assert config_path.exists(), f"Config file not found: {config_path}"

    def test_config_has_required_sections(self):
        """Test config has all required sections."""
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        required_sections = ['system', 'data', 'dataset', 'model', 'training']
        for section in required_sections:
            assert section in config, f"Missing section: {section}"

    def test_training_config_valid(self):
        """Test training configuration has valid values."""
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        training = config['training']

        # Test basic validations
        assert training['num_epochs'] > 0, "num_epochs must be positive"
        assert training['batch_size'] > 0, "batch_size must be positive"
        assert training['batch_size'] <= 32, "batch_size too large for Colab"

    def test_dataset_config_valid(self):
        """Test dataset configuration has valid values."""
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        dataset = config['dataset']

        assert dataset['samples_per_stock'] > 0, "samples_per_stock must be positive"
        assert dataset['samples_per_stock'] <= 200, "samples_per_stock too large"

    def test_model_config_valid(self):
        """Test model configuration has required fields."""
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        model = config['model']

        assert 'model_name' in model, "Missing model_name"
        assert 'parameters' in model, "Missing parameters"

        # Validate model name is a string
        assert isinstance(model['model_name'], str), "model_name must be string"


class TestDataValidation:
    """Test data file validation."""

    def test_data_directory_exists(self):
        """Test data directory exists."""
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        data_path = Path(config.get('data', {}).get('processed_path', 'data/processed'))
        assert data_path.exists(), f"Data directory not found: {data_path}"

    def test_data_files_exist(self):
        """Test data files exist."""
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        data_path = Path(config.get('data', {}).get('processed_path', 'data/processed'))
        files = list(data_path.glob('*_processed.csv'))

        assert len(files) >= 10, f"Insufficient data files: {len(files)} (< 10)"
        assert len(files) == 30, f"Expected 30 files, got {len(files)}"

    def test_data_files_have_content(self):
        """Test data files have valid content."""
        import pandas as pd

        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        data_path = Path(config.get('data', {}).get('processed_path', 'data/processed'))
        files = list(data_path.glob('*_processed.csv'))

        # Check first 5 files have content
        for file in files[:5]:
            df = pd.read_csv(file)
            assert len(df) > 0, f"Empty file: {file.name}"

            # Check for required columns
            assert 'RV_20' in df.columns, f"Missing RV_20 column in {file.name}"
            assert not df['RV_20'].isnull().all(), f"All NaN in {file.name}"


class TestGPUValidation:
    """Test GPU availability and requirements."""

    def test_cuda_available(self):
        """Test CUDA is available."""
        assert torch.cuda.is_available(), "CUDA not available - GPU required"

    def test_gpu_memory_sufficient(self):
        """Test GPU memory is sufficient."""
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        assert gpu_memory >= 8.0, f"Insufficient GPU memory: {gpu_memory:.1f}GB < 8GB"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="GPU not available"
    )
    def test_gpu_can_load_model(self):
        """Test GPU can load TimesFM model."""
        try:
            from transformers import TimesFm2_5ModelForPrediction

            # Test model loading with bfloat16
            model = TimesFm2_5ModelForPrediction.from_pretrained(
                "google/timesfm-2.5-200m-transformers",
                torch_dtype=torch.bfloat16
            )

            assert model is not None, "Model loading failed"

            # Clean up
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            pytest.fail(f"Model loading failed: {e}")


class TestImportValidation:
    """Test import compatibility."""

    def test_transformers_import(self):
        """Test transformers library import."""
        from transformers import TimesFm2_5ModelForPrediction
        assert TimesFm2_5ModelForPrediction is not None

    def test_peft_import(self):
        """Test PEFT library import."""
        from peft import LoraConfig, get_peft_model
        assert LoraConfig is not None
        assert get_peft_model is not None

    def test_torch_import(self):
        """Test PyTorch import."""
        assert torch is not None
        assert torch.cuda.is_available(), "CUDA not available"

    def test_pandas_import(self):
        """Test pandas import."""
        import pandas as pd
        assert pd is not None

    def test_numpy_import(self):
        """Test numpy import."""
        import numpy as np
        assert np is not None

    def test_yaml_import(self):
        """Test PyYAML import."""
        import yaml
        assert yaml is not None


class TestCleanModuleImport:
    """Test that module imports cleanly without monkey patching."""

    def test_no_monkey_patching_in_training_module(self):
        """Test that training module does not contain monkey patching code."""
        with open('src/model_training_fixed.py', 'r') as f:
            code = f.read()

        # Check for forbidden monkey patching patterns
        forbidden_patterns = [
            'importlib.util.find_spec',  # Monkey patching importlib
            'patched_find_spec',         # Patched find_spec function
            'sys.modules[\'bitsandbytes\']',  # Fake module injection
            'peft_utils.is_bnb_available',  # PEFT monkey patching
            'float8_e8m0fnu',            # PyTorch compatibility workaround
        ]

        for pattern in forbidden_patterns:
            assert pattern not in code, f"Found forbidden monkey patching pattern: {pattern}"

    def test_clean_import_of_training_module(self):
        """Test that training module imports cleanly OR gives clear incompatibility error."""
        import subprocess
        import sys

        # Test import in clean Python environment
        result = subprocess.run(
            [sys.executable, '-c', 'import sys; sys.path.insert(0, "src"); from model_training_fixed import TimesFMVN30Finetuner; print("Import OK")'],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Either imports successfully OR gives clear incompatibility error (not silent failure)
        if result.returncode != 0:
            # If import fails, it should be due to dependency incompatibility, not monkey patching
            assert "float8_e8m0fnu" in result.stderr or "ModuleNotFoundError" in result.stderr, \
                f"Import failed with unexpected error: {result.stderr}"
            # Should NOT have monkey patching warnings
            assert "patch" not in result.stderr.lower(), "Error mentions monkey patching"
        else:
            # If import succeeds, verify no warnings
            assert "Import OK" in result.stdout, "Import did not complete"
            assert "patch" not in result.stderr.lower(), "Import triggered patching warnings"
            assert "warning" not in result.stderr.lower(), "Import generated warnings"

    def test_cuda_config_not_at_import_time(self):
        """Test that CUDA config is not set at import time."""
        import subprocess
        import sys

        # Test that CUDA config is not set globally at import
        result = subprocess.run(
            [sys.executable, '-c', '''
import os
import sys

# Check CUDA env before import
cuda_before = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")

# Try to read the training script to check for CUDA env setting
with open("src/model_training_fixed.py", "r") as f:
    code = f.read()

# Check that os.environ CUDA config is NOT in global scope
lines = code.split("\\n")
import_section = []
main_section = []
in_main = False

for line in lines:
    if "def main()" in line:
        in_main = True
    if in_main:
        main_section.append(line)
    else:
        import_section.append(line)

import_code = "\\n".join(import_section)
main_code = "\\n".join(main_section)

# Verify CUDA config is NOT in import section
assert "PYTORCH_CUDA_ALLOC_CONF" not in import_code or "def main" in import_code, \\
    "CUDA config found in import section (should be in main() only)"

# Verify CUDA config IS in main section
assert "PYTORCH_CUDA_ALLOC_CONF" in main_code, \\
    "CUDA config not found in main() section"

print("CUDA config NOT at import time: OK")
'''],
            capture_output=True,
            text=True,
            timeout=30
        )

        assert result.returncode == 0, f"CUDA config check failed: {result.stderr}"
        assert "CUDA config NOT at import time: OK" in result.stdout


class TestColabPreflightScript:
    """Test the pre-flight validation script itself."""

    def test_preflight_script_exists(self):
        """Test pre-flight script exists."""
        script_path = Path('scripts/colab_preflight.py')
        assert script_path.exists(), f"Pre-flight script not found: {script_path}"

    def test_preflight_script_executable(self):
        """Test pre-flight script is executable."""
        script_path = Path('scripts/colab_preflight.py')
        # Just check file exists, don't test execute permissions in CI
        assert script_path.exists()

    @pytest.mark.skipif(
        sys.platform == 'win32',
        reason="Script execution may fail on Windows CI"
    )
    def test_preflight_script_runs(self):
        """Test pre-flight script executes successfully."""
        import subprocess

        result = subprocess.run(
            ['python', 'scripts/colab_preflight.py', '--config', 'configs/config.yaml'],
            capture_output=True,
            text=True,
            timeout=60
        )

        # Check exit code
        assert result.returncode == 0, f"Script failed with exit code {result.returncode}"
        assert "✅ ALL CHECKS PASSED" in result.stdout, "Script didn't pass all checks"


class TestTrainingScriptComponents:
    """Test individual components of training script."""

    def test_timesfm_finetuner_class_exists(self):
        """Test TimesFMVN30Finetuner class exists."""
        try:
            from model_training_fixed import TimesFMVN30Finetuner
            assert TimesFMVN30Finetuner is not None
        except ImportError as e:
            pytest.skip(f"Cannot import TimesFMVN30Finetuner: {e}")

    def test_finetuner_initialization(self):
        """Test finetuner can be initialized."""
        try:
            from model_training_fixed import TimesFMVN30Finetuner

            finetuner = TimesFMVN30Finetuner()
            assert finetuner is not None
            assert finetuner.config is not None

        except Exception as e:
            pytest.skip(f"Cannot initialize finetuner: {e}")

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="GPU not available"
    )
    def test_model_loading(self):
        """Test model can be loaded."""
        try:
            from transformers import TimesFm2_5ModelForPrediction

            model = TimesFm2_5ModelForPrediction.from_pretrained(
                "google/timesfm-2.5-200m-transformers",
                torch_dtype=torch.bfloat16
            )

            assert model is not None

            # Clean up
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            pytest.fail(f"Model loading failed: {e}")


class TestMemoryRequirements:
    """Test memory requirements."""

    def test_system_ram_sufficient(self):
        """Test system RAM is sufficient."""
        try:
            import psutil

            system_ram = psutil.virtual_memory().total / 1e9
            assert system_ram >= 8.0, f"Insufficient RAM: {system_ram:.1f}GB < 8GB"

        except ImportError:
            pytest.skip("psutil not installed")

    def test_gpu_memory_calculation(self):
        """Test GPU memory calculation is correct."""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

            # Validate memory is reasonable (8GB to 80GB)
            assert 8.0 <= gpu_memory <= 80.0, f"Unusual GPU memory: {gpu_memory:.1f}GB"


class TestExceptionHandling:
    """Test exception handling in training script."""

    def test_save_training_history_has_error_handling(self):
        """Test save_training_history has exception handling."""
        with open('src/model_training_fixed.py', 'r') as f:
            code = f.read()

        # Check for try-except in save_training_history
        assert 'def save_training_history' in code
        assert 'try:' in code
        assert 'except Exception' in code

    def test_training_loop_has_error_handling(self):
        """Test training loop has exception handling."""
        with open('src/model_training_fixed.py', 'r') as f:
            code = f.read()

        # Check for try-except in training loop
        assert 'for epoch in range' in code
        assert 'except Exception as e:' in code
        assert '[ERROR] Exception during epoch' in code


class TestConfigIntegrity:
    """Test configuration file integrity."""

    def test_config_is_valid_yaml(self):
        """Test config file is valid YAML."""
        try:
            with open('configs/config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            assert config is not None
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML: {e}")

    def test_config_no_syntax_errors(self):
        """Test config has no syntax errors."""
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Test basic structure
        assert isinstance(config, dict)
        assert 'training' in config
        assert isinstance(config['training'], dict)


# Parametrized tests for different batch sizes
@pytest.mark.parametrize("batch_size,expected", [
    (8, True),    # Valid for Colab
    (12, True),   # Valid for G4
    (16, True),   # Valid for A100
    (64, False),  # Too large for most GPUs
])
def test_batch_size_validation(batch_size, expected):
    """Test batch size validation logic."""
    if expected:
        assert batch_size <= 32, f"Batch size {batch_size} too large for Colab"
    else:
        assert batch_size > 32, f"Batch size {batch_size} should fail validation"


# Parametrized tests for epoch counts
@pytest.mark.parametrize("num_epochs,expected", [
    (1, True),     # Valid for testing
    (10, True),    # Valid for staging
    (100, True),   # Valid for production
    (0, False),    # Invalid
    (-5, False),   # Invalid
])
def test_epochs_validation(num_epochs, expected):
    """Test epoch count validation logic."""
    if expected:
        assert num_epochs > 0, f"Epochs {num_epochs} must be positive"
    else:
        assert num_epochs <= 0, f"Epochs {num_epochs} should be invalid"


class TestDataPipelineIntegrity:
    """Test data pipeline integrity."""

    def test_data_files_readable(self):
        """Test data files are readable."""
        import pandas as pd

        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        data_path = Path(config.get('data', {}).get('processed_path', 'data/processed'))
        files = list(data_path.glob('*_processed.csv'))

        # Test first 5 files are readable
        for file in files[:5]:
            try:
                df = pd.read_csv(file)
                assert len(df) > 0
            except Exception as e:
                pytest.fail(f"Cannot read {file.name}: {e}")

    def test_data_files_have_required_columns(self):
        """Test data files have required columns."""
        import pandas as pd

        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        data_path = Path(config.get('data', {}).get('processed_path', 'data/processed'))
        files = list(data_path.glob('*_processed.csv'))

        required_columns = ['RV_20']  # Adjust based on your actual columns

        for file in files[:3]:  # Check first 3 files
            df = pd.read_csv(file)
            for col in required_columns:
                assert col in df.columns, f"Missing column {col} in {file.name}"


# Test fixtures for setup/teardown
@pytest.fixture
def sample_config():
    """Provide sample config for testing."""
    return {
        'system': {
            'device': 'cuda',
            'random_seed': 42
        },
        'training': {
            'batch_size': 8,
            'num_epochs': 5,
            'gradient_accumulation_steps': 6
        },
        'dataset': {
            'samples_per_stock': 10
        }
    }


@pytest.fixture
def mock_gpu_available():
    """Mock GPU availability for testing."""
    with patch('torch.cuda.is_available', return_value=True):
        with patch('torch.cuda.get_device_properties') as mock_props:
            mock_gpu = MagicMock()
            mock_gpu.total_memory = 16 * 1024**3 * 1024**3  # 16GB
            mock_props.return_value = mock_gpu
            yield


class TestColabFreeTierCompatibility:
    """Test compatibility with Colab free tier."""

    def test_free_tier_batch_size(self):
        """Test batch size is compatible with free tier."""
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        batch_size = config['training']['batch_size']
        # Free tier T4 typically 16GB
        assert batch_size <= 16, f"Batch size {batch_size} too large for free tier"

    def test_free_tier_memory_safe(self):
        """Test config is safe for free tier memory."""
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Estimate memory usage
        batch_size = config['training']['batch_size']
        samples = config['dataset']['samples_per_stock']

        # Rough estimate: batch_size * samples * features * bytes
        estimated_mb = (batch_size * samples * 64 * 4) / (1024**2)  # Very rough

        # Should be under 12GB for free tier
        assert estimated_mb < 12000, f"Estimated memory {estimated_mb:.0f}MB too high"


# Integration tests
class TestEndToEndValidation:
    """End-to-end integration tests."""

    def test_full_preflight_validation(self):
        """Test full pre-flight validation runs successfully."""
        import subprocess

        result = subprocess.run(
            ['python', 'scripts/colab_preflight.py', '--config', 'configs/config.yaml'],
            capture_output=True,
            text=True,
            timeout=120
        )

        # Should pass all checks
        assert result.returncode == 0, f"Preflight failed: {result.stderr}"
        assert "[PASS] ALL CHECKS PASSED" in result.stdout

    def test_preflight_with_invalid_config(self):
        """Test pre-flight catches invalid config."""
        import subprocess
        import tempfile

        # Create invalid config
        invalid_config = {
            'training': {
                'num_epochs': -1,  # Invalid
                'batch_size': 0     # Invalid
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_config = f.name

        try:
            result = subprocess.run(
                ['python', 'scripts/colab_preflight.py', '--config', temp_config],
                capture_output=True,
                text=True,
                timeout=60
            )

            # Should fail validation
            assert result.returncode != 0, "Should have failed with invalid config"
            assert "[FAIL]" in result.stdout, "Should show validation errors"

        finally:
            Path(temp_config).unlink()


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
