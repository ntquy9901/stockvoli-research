"""
Environment Validation Script for TimesFM VN30 Fine-tuning
Checks GPU availability, dependencies, and system requirements
"""

import sys
import subprocess
from pathlib import Path
import io
import contextlib

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def check_python_version():
    """Check Python version (requires 3.10+)"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - requires 3.10+")
        return False


def check_pytorch():
    """Check PyTorch installation and GPU support"""
    print("\n🔍 Checking PyTorch installation...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")

        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA Available: {torch.version.cuda}")
            print(f"✅ GPU Count: {torch.cuda.device_count()}")
            print(f"✅ GPU Name: {torch.cuda.get_device_name(0)}")

            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU Memory: {gpu_memory:.1f} GB")

            if gpu_memory >= 16:
                print("✅ Sufficient GPU memory for TimesFM training")
                return True
            else:
                print(f"⚠️  GPU memory {gpu_memory:.1f}GB < 16GB recommended")
                print("   You can still train with reduced batch size")
                return True
        else:
            print("❌ CUDA NOT available - GPU required for TimesFM training")
            print("   Options:")
            print("   1. Install CUDA-enabled PyTorch")
            print("   2. Use cloud GPU (RunPod, Lambda Labs, Colab Pro)")
            print("   3. Continue with CPU-only for data processing setup")
            return False

    except ImportError:
        print("❌ PyTorch not installed - install with: pip install torch")
        return False


def check_transformers():
    """Check Transformers library for TimesFM"""
    print("\n🔍 Checking Transformers library...")
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")

        # Check for TimesFM support
        try:
            from transformers import TimesFm2_5ModelForPrediction
            print("✅ TimesFM 2.5 support available")
            return True
        except ImportError:
            print("⚠️  TimesFM 2.5 not available - install latest transformers")
            print("   pip install transformers>=4.35.0")
            return False

    except ImportError:
        print("❌ Transformers not installed")
        print("   pip install transformers>=4.35.0")
        return False


def check_peft():
    """Check PEFT library for LoRA adapters"""
    print("\n🔍 Checking PEFT library...")
    try:
        import peft
        print(f"✅ PEFT {peft.__version__}")

        try:
            from peft import LoraConfig, get_peft_model
            print("✅ LoRA adapter support available")
            return True
        except ImportError:
            print("⚠️  LoRA functions not available")
            return False

    except ImportError:
        print("❌ PEFT not installed")
        print("   pip install peft>=0.5.0")
        return False


def check_other_dependencies():
    """Check other required dependencies"""
    print("\n🔍 Checking other dependencies...")

    required = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'scikit-learn': 'Scikit-learn',
        'scipy': 'SciPy',
        'accelerate': 'Accelerate'
    }

    all_ok = True
    for package, name in required.items():
        try:
            if package == 'scikit-learn':
                import sklearn
                print(f"✅ {name} {sklearn.__version__}")
            elif package == 'accelerate':
                import accelerate
                print(f"✅ {name} {accelerate.__version__}")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"✅ {name} {version}")
        except ImportError:
            print(f"❌ {name} not installed")
            all_ok = False

    return all_ok


def check_data_availability():
    """Check if Vietnamese stock data is available"""
    print("\n🔍 Checking Vietnamese stock data...")

    data_path = Path("data/raw/prices")
    if data_path.exists():
        stock_files = list(data_path.glob("*_ohlcv.csv"))
        print(f"✅ Found {len(stock_files)} stock data files")

        if len(stock_files) >= 30:
            print("✅ All 30 VN30 stocks data available")
            return True
        elif len(stock_files) >= 10:
            print(f"⚠️  {len(stock_files)} stocks found - aiming for 30")
            return True
        else:
            print(f"❌ Only {len(stock_files)} stocks - insufficient data")
            return False
    else:
        print(f"❌ Data directory not found: {data_path}")
        print("   Create directory and add Vietnamese stock OHLCV data")
        return False


def check_project_structure():
    """Check project directory structure"""
    print("\n🔍 Checking project structure...")

    required_dirs = [
        "data/raw/prices",
        "data/processed",
        "data/features",
        "models/timesfm",
        "models/checkpoints",
        "models/final_models",
        "experiments",
        "configs",
        "src",
        "tests",
        "setup"
    ]

    all_ok = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"⚠️  {dir_path} - will be created")
            path.mkdir(parents=True, exist_ok=True)

    return True


def check_config_file():
    """Check if configuration file exists"""
    print("\n🔍 Checking configuration file...")

    config_path = Path("configs/config.yaml")
    if config_path.exists():
        print(f"✅ Configuration file found: {config_path}")
        return True
    else:
        print(f"❌ Configuration file not found: {config_path}")
        return False


def main():
    """Run all environment checks"""
    print("=" * 70)
    print("🏗️  TimesFM VN30 Environment Validation")
    print("=" * 70)

    checks = [
        ("Python Version", check_python_version),
        ("PyTorch & GPU", check_pytorch),
        ("Transformers Library", check_transformers),
        ("PEFT Library", check_peft),
        ("Other Dependencies", check_other_dependencies),
        ("Data Availability", check_data_availability),
        ("Project Structure", check_project_structure),
        ("Configuration File", check_config_file)
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ Error checking {name}: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 70)
    print("📋 ENVIRONMENT VALIDATION SUMMARY")
    print("=" * 70)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nResult: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 All checks passed! Ready for TimesFM fine-tuning")
        return 0
    elif passed >= total - 2:  # Allow GPU to be missing for setup
        print("\n⚠️  Most checks passed - can proceed with setup")
        print("   Note: GPU required for actual model training")
        return 0
    else:
        print("\n❌ Some critical checks failed - please fix issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())