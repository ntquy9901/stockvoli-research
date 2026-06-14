#!/usr/bin/env python3
"""
Test Batch Size Performance Comparison

This script tests different batch sizes to find optimal configuration:
- Test 1: batch_size=48 (4x improvement)
- Test 2: batch_size=96 (8x improvement)

Usage:
    python src/test_batch_performance.py
"""

import sys
from pathlib import Path
import time
import logging
import yaml
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def setup_test_logging(config_name: str):
    """Setup logging for batch size testing"""
    log_dir = Path(f"experiments/batch_performance_tests/{config_name}")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Override any existing loggers
    )
    return logging.getLogger(__name__)


def run_batch_size_test(config_file: str, test_name: str):
    """
    Run training test with specific batch size configuration.

    Args:
        config_file: Path to config YAML file
        test_name: Name of the test (for logging)
    """
    logger = setup_test_logging(test_name)
    logger.info("=" * 70)
    logger.info(f"BATCH SIZE PERFORMANCE TEST - {test_name}")
    logger.info("=" * 70)

    # Load configuration
    logger.info(f"\n[1] Loading configuration from {config_file}")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    batch_size = config['training']['batch_size']
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    logger.info(f"  Effective batch size: {batch_size * config['training']['gradient_accumulation_steps']}")

    # Import training modules
    logger.info("\n[2] Initializing training modules...")
    try:
        from src.model_training_fixed import TimesFMVN30Finetuner
        import torch

        # Check GPU
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"  GPU Memory: {gpu_memory:.2f}GB")
        else:
            logger.warning("  CUDA not available - using CPU")
            return None

        # Initialize finetuner
        logger.info("\n[3] Initializing TimesFM finetuner...")
        finetuner = TimesFMVN30Finetuner(config_file)

        # Prepare dataloaders
        logger.info("\n[4] Preparing dataloaders...")
        from src.model_training_fixed import create_vn30_dataloaders
        train_loader, test_loader = create_vn30_dataloaders(config_file)
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Test batches: {len(test_loader)}")

        # Load model
        logger.info("\n[5] Loading TimesFM model...")
        finetuner.load_timesfm_model()
        finetuner.setup_lora_adapters()
        logger.info(f"  Model loaded successfully")

        # Measure batch processing time
        logger.info("\n[6] Measuring batch processing performance...")
        logger.info("  Running 10 batches to measure average time...")

        batch_times = []

        # Get first batch
        batch_iter = iter(train_loader)

        # Measure batch processing time
        logger.info("\n[5] Measuring batch processing performance...")
        logger.info("  Running 10 batches to measure average time...")

        batch_times = []

        # Get train loader from finetuner
        train_loader = finetuner.train_loader

        # Get first batch
        batch_iter = iter(train_loader)

        for i in range(10):
            try:
                batch = next(batch_iter)

                # Move to GPU
                batch = batch.to(finetuner.device)

                # Measure time
                start_time = time.time()

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    # Forward pass only (no backward for timing test)
                    outputs = finetuner.model(batch)

                torch.cuda.synchronize()  # Wait for GPU to finish
                end_time = time.time()

                batch_time = (end_time - start_time) * 1000  # Convert to ms
                batch_times.append(batch_time)

                logger.info(f"  Batch {i+1}/10: {batch_time:.2f}ms")

            except StopIteration:
                logger.warning(f"  Ran out of batches after {i+1} iterations")
                break
            except Exception as e:
                logger.error(f"  Error processing batch {i+1}: {e}")
                break

        # Calculate statistics
        if batch_times:
            avg_time = sum(batch_times) / len(batch_times)
            min_time = min(batch_times)
            max_time = max(batch_times)

            logger.info("\n[7] Performance Results:")
            logger.info("=" * 70)
            logger.info(f"Average batch time: {avg_time:.2f}ms")
            logger.info(f"Min batch time:     {min_time:.2f}ms")
            logger.info(f"Max batch time:     {max_time:.2f}ms")
            logger.info(f"Samples per second: {batch_size * 1000 / avg_time:.2f}")

            # Estimate epoch time
            batches_per_epoch = len(train_loader)
            estimated_epoch_time = (avg_time / 1000) * batches_per_epoch
            logger.info(f"Estimated epoch time: {estimated_epoch_time / 60:.2f} minutes")

            # GPU Memory
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / 1e9
                memory_reserved = torch.cuda.memory_reserved(0) / 1e9
                logger.info(f"GPU Memory allocated: {memory_allocated:.2f}GB")
                logger.info(f"GPU Memory reserved:  {memory_reserved:.2f}GB")

            logger.info("=" * 70)

            return {
                'test_name': test_name,
                'batch_size': batch_size,
                'avg_batch_time_ms': avg_time,
                'min_batch_time_ms': min_time,
                'max_batch_time_ms': max_time,
                'samples_per_second': batch_size * 1000 / avg_time,
                'estimated_epoch_time_sec': estimated_epoch_time,
                'batches_per_epoch': batches_per_epoch,
                'gpu_memory_gb': memory_allocated if torch.cuda.is_available() else 0
            }
        else:
            logger.error("No batches were successfully processed!")
            return None

    except Exception as e:
        logger.error(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_results(results: list):
    """Compare results from different batch size tests"""
    if not results or None in results:
        print("\n[ERROR] Cannot compare results - some tests failed")
        return

    print("\n" + "=" * 70)
    print("BATCH SIZE PERFORMANCE COMPARISON")
    print("=" * 70)

    print(f"{'Test':<20} {'Batch':<10} {'Avg Time':<12} {'Speedup':<10} {'GPU Mem':<10}")
    print("-" * 70)

    baseline_time = None

    for result in results:
        test_name = result['test_name']
        batch_size = result['batch_size']
        avg_time = result['avg_batch_time_ms']
        gpu_mem = result['gpu_memory_gb']

        if baseline_time is None:
            baseline_time = avg_time
            speedup = "1.0x (baseline)"
        else:
            speedup_ratio = baseline_time / avg_time
            speedup = f"{speedup_ratio:.2f}x"

        print(f"{test_name:<20} {batch_size:<10} {avg_time:<12.2f} {speedup:<10} {gpu_mem:<10.2f}")

    print("=" * 70)

    # Find optimal
    best_result = max(results, key=lambda x: x['samples_per_second'])
    print(f"\n[RECOMMENDED] {best_result['test_name']}")
    print(f"   Batch size: {best_result['batch_size']}")
    print(f"   Throughput: {best_result['samples_per_second']:.2f} samples/sec")
    print(f"   Epoch time: {best_result['estimated_epoch_time_sec']/60:.2f} minutes")


def main():
    """Run batch size performance tests"""
    print("\n")
    print("=" * 70)
    print(" Batch Size Performance Testing - TimesFM VN30")
    print("=" * 70)

    # Test configurations
    test_configs = [
        ('configs/config_batch48_test.yaml', 'batch_size_48'),
        ('configs/config_batch96_test.yaml', 'batch_size_96')
    ]

    results = []

    for config_file, test_name in test_configs:
        print(f"\n[TEST] Running: {test_name}")
        result = run_batch_size_test(config_file, test_name)

        if result:
            results.append(result)
            print(f"[OK] Test completed: {test_name}")
        else:
            print(f"[FAIL] Test failed: {test_name}")

        # Cleanup GPU memory between tests
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compare results
    if results:
        compare_results(results)

        # Save results to file
        from datetime import datetime
        results_file = Path("experiments/batch_performance_tests/results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)

        import json
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': results
            }, f, indent=2)

        print(f"\n[SAVE] Results saved to: {results_file}")
    else:
        print("\n[ERROR] No successful tests to compare")


if __name__ == "__main__":
    main()
