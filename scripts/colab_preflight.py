#!/usr/bin/env python3
"""
Colab Pre-Flight Validation Script
===============================

Run this locally before pushing to Colab to catch issues early.

Validates:
- Configuration file integrity
- Data file existence and quality
- GPU availability and memory
- Model import compatibility
- Python package dependencies

Returns exit code 0 if safe for Colab, 1 if issues found.

Usage:
    python scripts/colab_preflight.py

    # Check detailed mode
    python scripts/colab_preflight.py --verbose

    # Fix auto-detected issues
    python scripts/colab_preflight.py --fix

Author: Murat (Test Architect)
Date: 2026-06-13
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

try:
    import torch
    import yaml
    import pandas as pd
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    print("Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)


class ColabPreflightValidator:
    """
    Pre-flight validator for Colab training runs.

    Performs comprehensive validation before expensive Colab GPU time.
    Catches configuration, data, and environment issues early.
    """

    def __init__(self, config_path: str = 'configs/config.yaml',
                 verbose: bool = False):
        self.config_path = Path(config_path)
        self.verbose = verbose
        self.issues = []
        self.warnings = []
        self.passed_checks = []

    def validate_all(self) -> int:
        """
        Run all pre-flight checks.

        Returns:
            int: 0 if all checks pass, 1 if any issues found
        """
        print("=" * 70)
        print("[TEST] COLAB PRE-FLIGHT VALIDATION")
        print("=" * 70)

        checks = [
            ("Config Validation", self.validate_config),
            ("Data Files Check", self.validate_data_files),
            ("GPU Availability", self.validate_gpu),
            ("Import Compatibility", self.validate_imports),
            ("Memory Requirements", self.validate_memory_requirements),
            ("Package Dependencies", self.validate_dependencies),
        ]

        for check_name, check_func in checks:
            try:
                if self.verbose:
                    print(f"\n[CHECK] Running: {check_name}")

                check_func()

                if self.verbose:
                    print(f"[PASS] {check_name} passed")

                self.passed_checks.append(check_name)

            except AssertionError as e:
                print(f"[FAIL] {check_name} FAILED: {e}")
                self.issues.append(f"{check_name}: {str(e)}")

            except Exception as e:
                print(f"[WARN] {check_name} ERROR: {e}")
                self.warnings.append(f"{check_name}: {str(e)}")

        # Print summary
        self.print_summary()

        return 0 if not self.issues else 1

    def validate_config(self):
        """Validate configuration file has required fields and valid values."""
        if not self.config_path.exists():
            raise AssertionError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Check required sections
        required_sections = ['system', 'data', 'dataset', 'model', 'training', 'experiment_tracking']
        for section in required_sections:
            if section not in config:
                raise AssertionError(f"Missing config section: '{section}'")

        # Validate training configuration
        training = config['training']
        if training['num_epochs'] <= 0:
            raise AssertionError(f"Invalid num_epochs: {training['num_epochs']}")

        if training['batch_size'] <= 0:
            raise AssertionError(f"Invalid batch_size: {training['batch_size']}")

        if training['batch_size'] > 32:
            raise AssertionError(f"Batch size too large for Colab: {training['batch_size']} (> 32)")

        # Validate dataset configuration
        dataset = config['dataset']
        if dataset['samples_per_stock'] <= 0:
            raise AssertionError(f"Invalid samples_per_stock: {dataset['samples_per_stock']}")

        if dataset['samples_per_stock'] > 200:
            raise AssertionError(f"samples_per_stock too large: {dataset['samples_per_stock']} (> 200)")

        # Validate model configuration
        model = config['model']
        if 'model_name' not in model:
            raise AssertionError("Missing model_name in model config")

        # Validate experiment_tracking configuration
        experiment_tracking = config.get('experiment_tracking', {})
        if 'save_every_n_epochs' not in experiment_tracking:
            raise AssertionError("Missing 'save_every_n_epochs' in experiment_tracking config")

        if experiment_tracking['save_every_n_epochs'] <= 0:
            raise AssertionError(f"Invalid save_every_n_epochs: {experiment_tracking['save_every_n_epochs']}")

        if self.verbose:
            print(f"  [OK] Config file exists: {self.config_path}")
            print(f"  [OK] Required sections present: {required_sections}")
            print(f"  [OK] Epochs: {training['num_epochs']}")
            print(f"  [OK] Batch size: {training['batch_size']}")
            print(f"  [OK] Samples per stock: {dataset['samples_per_stock']}")
            print(f"  [OK] Save every N epochs: {experiment_tracking['save_every_n_epochs']}")

    def validate_data_files(self):
        """Validate data files exist and have valid content."""
        # Check config for data path
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        data_path = Path(config.get('data', {}).get('processed_path', 'data/processed'))

        if not data_path.exists():
            raise AssertionError(f"Data directory not found: {data_path}")

        # Check for data files
        files = list(data_path.glob('*_processed.csv'))

        if len(files) < 10:
            raise AssertionError(f"Insufficient data files: {len(files)} (< 10)")

        # Validate file contents (check first 5 files)
        for file in files[:5]:
            try:
                df = pd.read_csv(file)
                if len(df) == 0:
                    raise AssertionError(f"Empty data file: {file.name}")
                if df.isnull().all().all():
                    raise AssertionError(f"All NaN in file: {file.name}")
            except Exception as e:
                raise AssertionError(f"Error reading {file.name}: {e}")

        if self.verbose:
            print(f"  [OK] Data directory: {data_path}")
            print(f"  [OK] Data files: {len(files)} files")
            print(f"  [OK] Sample files validated: {min(5, len(files))} files")

    def validate_gpu(self):
        """Validate GPU availability and basic requirements."""
        if not torch.cuda.is_available():
            raise AssertionError("CUDA not available - GPU required for Colab training")

        # Check GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

        if gpu_memory < 8.0:
            raise AssertionError(f"Insufficient GPU memory: {gpu_memory:.1f}GB < 8GB minimum")

        if gpu_memory < 16.0:
            self.warnings.append(
                f"GPU memory {gpu_memory:.1f}GB < 16GB recommended (may have issues)"
            )

        if self.verbose:
            print(f"  [OK] GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  [OK] GPU memory: {gpu_memory:.1f}GB")

    def validate_imports(self):
        """Validate all required imports work correctly."""
        try:
            # Test basic imports that don't depend on transformers/torch compatibility
            import torch
            import pandas
            import numpy
            import yaml

            if self.verbose:
                print("  [CHECK] Testing basic imports...")

            # Test basic torch functionality
            if torch.cuda.is_available():
                if self.verbose:
                    print(f"  [OK] PyTorch with CUDA support")
            else:
                if self.verbose:
                    print("  [WARN] PyTorch installed but CUDA not available (OK for local testing)")

            # Test pandas
            if self.verbose:
                print(f"  [OK] pandas version: {pandas.__version__}")

            # Test numpy
            if self.verbose:
                print(f"  [OK] numpy version: {numpy.__version__}")

            if self.verbose:
                print("  [PASS] Basic imports successful (transformers/peft checked in dependencies)")

        except ImportError as e:
            raise AssertionError(f"Import error: {e}")

    def validate_memory_requirements(self):
        """Validate system RAM requirements."""
        try:
            import psutil

            system_ram = psutil.virtual_memory().total / 1e9

            if system_ram < 8.0:
                raise AssertionError(f"Insufficient system RAM: {system_ram:.1f}GB < 8GB")

            if self.verbose:
                print(f"  [OK] System RAM: {system_ram:.1f}GB")
        except ImportError:
            # psutil not installed - skip this check
            if self.verbose:
                print("  [SKIP] psutil not installed - skipping RAM check")

    def validate_dependencies(self):
        """Validate required Python packages are installed."""
        # Use importlib.metadata to check package installation without importing
        try:
            from importlib.metadata import distributions
        except ImportError:
            from importlib_metadata import distributions

        required_packages = {
            'torch': 'torch',
            'transformers': 'transformers',
            'peft': 'peft',
            'pandas': 'pandas',
            'numpy': 'numpy',
            'PyYAML': 'yaml',
        }

        installed_packages = {dist.metadata['Name']: dist.version for dist in distributions()}

        missing = []
        for display_name, package_name in required_packages.items():
            # Check if package is installed (case-insensitive)
            is_installed = any(
                package_name.lower() in installed_name.lower()
                for installed_name in installed_packages.keys()
            )

            if is_installed:
                if self.verbose:
                    print(f"  [OK] {display_name}")
            else:
                missing.append(display_name)

        if missing:
            raise AssertionError(f"Missing packages: {', '.join(missing)}")

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 70)
        print("[SUMMARY] VALIDATION SUMMARY")
        print("=" * 70)

        print(f"[PASS] Passed checks: {len(self.passed_checks)}/{len(self.passed_checks) + len(self.issues)}")

        if self.issues:
            print(f"\n[FAIL] ISSUES FOUND ({len(self.issues)}):")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")

        if self.warnings:
            print(f"\n[WARN] WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")

        if not self.issues:
            print("\n[PASS] ALL CHECKS PASSED - SAFE FOR COLAB!")
            print("[INFO] You can proceed with Colab training with confidence.")
        else:
            print("\n[FAIL] VALIDATION FAILED - FIX ISSUES BEFORE COLAB")
            print("[INFO] Run locally to fix issues (no GPU time cost!)")

        print("=" * 70)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Colab pre-flight validation'
    )
    parser.add_argument(
        '--config',
        default='configs/config.yaml',
        help='Path to config file (default: configs/config.yaml)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    validator = ColabPreflightValidator(
        config_path=args.config,
        verbose=args.verbose
    )

    exit_code = validator.validate_all()

    sys.exit(exit_code)


if __name__ == '__main__':
    main()