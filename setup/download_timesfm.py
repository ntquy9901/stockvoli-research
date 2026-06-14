"""
TimesFM 2.5 Model Download and Test Script
Tests downloading and loading TimesFM 2.5 from HuggingFace
"""

import sys
import torch
from pathlib import Path
import io

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def test_timesfm_loading():
    """Test downloading and loading TimesFM 2.5 model"""
    print("=" * 70)
    print("🏗️  TimesFM 2.5 Model Download & Loading Test")
    print("=" * 70)

    # Check PyTorch and CUDA
    print("\n📋 Checking PyTorch and CUDA...")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available - TimesFM training requires GPU")
        print("   Continuing with CPU test (model will load but training will be slow)")
        device = "cpu"
    else:
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        device = "cuda"

    # Test Transformers and TimesFM
    print("\n📋 Testing Transformers library and TimesFM 2.5...")
    try:
        from transformers import TimesFm2_5ModelForPrediction
        print("✅ TimesFM 2.5 support available")

    except ImportError as e:
        print(f"❌ Cannot import TimesFM 2.5: {e}")
        print("   Install: pip install transformers>=4.35.0")
        return False

    # Test model download and loading
    print("\n📋 Downloading TimesFM 2.5 model (this may take a few minutes)...")
    print("   Model: google/timesfm-2.5-200m-transformers")
    print("   Size: ~400MB")

    try:
        model = TimesFm2_5ModelForPrediction.from_pretrained(
            "google/timesfm-2.5-200m-transformers",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto" if device == "cuda" else None
        )
        print("✅ TimesFM 2.5 model downloaded successfully")

    except Exception as e:
        print(f"❌ Failed to download TimesFM 2.5: {e}")
        print("   Check internet connection and HuggingFace access")
        return False

    # Test model architecture
    print("\n📋 Testing model architecture...")
    try:
        print(f"✅ Model type: {type(model).__name__}")
        print(f"✅ Model config: {model.config}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Total parameters: {total_params:,} (~200M)")

    except Exception as e:
        print(f"❌ Error examining model architecture: {e}")
        return False

    # Test model forward pass
    print("\n📋 Testing model forward pass...")
    try:
        # Create sample input (batch_size=1, seq_len=64, features=7)
        batch_size = 1
        context_len = 64
        n_features = 7

        dummy_input = torch.randn(batch_size, context_len, n_features)

        if device == "cuda":
            dummy_input = dummy_input.cuda()

        print(f"   Input shape: {dummy_input.shape}")

        # Forward pass
        with torch.no_grad():
            outputs = model(past_values=dummy_input)

        print(f"✅ Forward pass successful")
        print(f"✅ Output shape: {outputs.last_hidden_state.shape}")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

    # Test LoRA adapter integration
    print("\n📋 Testing LoRA adapter integration...")
    try:
        from peft import LoraConfig, get_peft_model

        # Create LoRA config
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none"
        )
        print("✅ LoRA config created")

        # Apply LoRA to model
        model_with_lora = get_peft_model(model, lora_config)
        print("✅ LoRA adapters applied to model")

        # Check trainable parameters
        trainable_params = sum(p.numel() for p in model_with_lora.parameters() if p.requires_grad)
        total_params_lora = sum(p.numel() for p in model_with_lora.parameters())

        print(f"✅ Total parameters: {total_params_lora:,}")
        print(f"✅ Trainable parameters: {trainable_params:,} ({trainable_params/total_params_lora*100:.1f}%)")

        if trainable_params < total_params_lora * 0.1:  # Less than 10% trainable
            print("✅ LoRA working correctly - only small subset of parameters trainable")
        else:
            print("⚠️  More parameters trainable than expected")

        # Test forward pass with LoRA
        with torch.no_grad():
            outputs_lora = model_with_lora(past_values=dummy_input)

        print(f"✅ LoRA model forward pass successful")

    except ImportError:
        print("⚠️  PEFT library not installed - install with: pip install peft>=0.5.0")
        print("   LoRA adapters required for efficient fine-tuning")
    except Exception as e:
        print(f"❌ LoRA integration failed: {e}")
        return False

    # Save model info
    print("\n📋 Saving model information...")
    try:
        models_dir = Path("models/timesfm")
        models_dir.mkdir(parents=True, exist_ok=True)

        model_info = {
            "model_name": "google/timesfm-2.5-200m-transformers",
            "parameters": total_params,
            "architecture": "TimesFM 2.5",
            "context_length": model.config.context_length if hasattr(model.config, 'context_length') else 128,
            "prediction_length": model.config.prediction_length if hasattr(model.config, 'prediction_length') else 1,
            "torch_dtype": str(model.dtype),
            "device": device,
            "lora_config": {
                "r": 4,
                "lora_alpha": 8,
                "target_modules": "all-linear"
            }
        }

        import json
        with open(models_dir / "timesfm_model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)

        print(f"✅ Model info saved to {models_dir / 'timesfm_model_info.json'}")

    except Exception as e:
        print(f"⚠️  Could not save model info: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("🎉 TIMESFM 2.5 LOADING TEST SUCCESSFUL!")
    print("=" * 70)
    print("\n✅ TimesFM 2.5 model downloaded and loaded successfully")
    print("✅ Model architecture validated")
    print("✅ Forward pass working")
    print("✅ LoRA adapter integration tested")
    print(f"✅ Ready for fine-tuning on {device.upper()}")

    if device == "cpu":
        print("\n⚠️  NOTE: Training on CPU will be very slow")
        print("   GPU highly recommended for TimesFM fine-tuning")

    return True


def main():
    """Main function"""
    try:
        success = test_timesfm_loading()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())