"""
Real-time Training Monitor for TimesFM Fine-tuning

Usage:
    python monitor_training.py

This script:
1. Monitors training progress
2. Displays learning curves
3. Shows current training statistics
4. Auto-refreshes every 30 seconds
"""

import time
import os
from pathlib import Path
from datetime import datetime

def display_banner():
    """Display welcome banner"""
    print("=" * 70)
    print(" " * 15 + "TIMESFM TRAINING MONITOR")
    print("=" * 70)
    print()

def check_training_status():
    """Check current training status"""
    history_path = Path('experiments/training_history.json')
    plot_path = Path('experiments/learning_curves.png')

    if not history_path.exists():
        print("[WAITING] Waiting for training to start...")
        print("  (Run python src/model_training_google_research.py to start training)")
        return None, None

    # Load training history
    import json
    with open(history_path, 'r') as f:
        history = json.load(f)

    if not history:
        print("[WAITING] No training epochs completed yet")
        return None, None

    # Get latest epoch
    latest = history[-1]
    epoch = latest['epoch']
    train_loss = latest['train_metrics']['loss']
    val_loss = latest['val_metrics']['val_loss']
    lr = latest.get('learning_rate', 0)

    # Find best epoch
    val_losses = [entry['val_metrics']['val_loss'] for entry in history]
    best_idx = val_losses.index(min(val_losses))
    best_epoch = history[best_idx]['epoch']
    best_val_loss = val_losses[best_idx]

    # Calculate improvement
    if len(history) > 1:
        prev_train = history[-2]['train_metrics']['loss']
        train_change = train_loss - prev_train
    else:
        train_change = 0

    # Print status
    print(f"[STATUS] Training Progress - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)
    print(f"Current Epoch:  {epoch}/100")
    print(f"Train Loss:     {train_loss:.4f} ({'+' if train_change > 0 else ''}{train_change:+.4f})")
    print(f"Val Loss:       {val_loss:.4f}")
    print(f"Learning Rate:  {lr:.8f}")
    print(f"Best Epoch:     {best_epoch} (Val Loss: {best_val_loss:.4f})")
    print("-" * 70)

    # Check for early stopping
    patience = 5
    recent_losses = val_losses[-patience:] if len(val_losses) >= patience else val_losses
    if len(recent_losses) >= patience:
        min_recent = min(recent_losses)
        if min_recent == best_val_loss:
            epochs_without_improvement = sum(1 for loss in recent_losses if loss <= min_recent + 0.001)
            if epochs_without_improvement >= patience:
                print(f"[ALERT] Early stopping likely soon ({epochs_without_improvement}/{patience} epochs without improvement)")

    return history, plot_path

def show_learning_curves_plot(plot_path):
    """Show information about learning curves plot"""
    if plot_path and plot_path.exists():
        size_kb = plot_path.stat().st_size / 1024
        mtime = datetime.fromtimestamp(plot_path.stat().st_mtime)
        print(f"\n[PLOT] Learning Curves:")
        print(f"  Path: {plot_path}")
        print(f"  Size: {size_kb:.1f} KB")
        print(f"  Last Updated: {mtime.strftime('%H:%M:%S')}")
        print(f"  \n  Open this file to see training curves:")
        print(f"  -> {plot_path.absolute()}")
    else:
        print("\n[PLOT] No learning curves generated yet")

def show_improvement_trend(history):
    """Show improvement trend"""
    if len(history) < 2:
        return

    print("\n[TREND] Recent Performance (Last 5 epochs):")
    print("Epoch | Train Loss | Val Loss | Train Change | Val Change")
    print("-" * 60)

    for entry in history[-5:]:
        epoch = entry['epoch']
        train_loss = entry['train_metrics']['loss']
        val_loss = entry['val_metrics']['val_loss']

        # Calculate change from previous
        idx = history.index(entry)
        if idx > 0:
            prev = history[idx - 1]
            train_delta = train_loss - prev['train_metrics']['loss']
            val_delta = val_loss - prev['val_metrics']['val_loss']
        else:
            train_delta = 0
            val_delta = 0

        print(f"{epoch:5d} | {train_loss:9.4f} | {val_loss:8.4f} | {train_delta:+8.4f} | {val_delta:+7.4f}")

def main():
    """Main monitoring loop"""
    display_banner()

    print("[INFO] Monitoring training progress...")
    print("[INFO] Press Ctrl+C to stop monitoring")
    print("[INFO] Learning curves will auto-refresh every 30 seconds")
    print()

    try:
        while True:
            # Clear screen (optional, commented out for better readability)
            # os.system('cls' if os.name == 'nt' else 'clear')

            display_banner()

            # Check status
            history, plot_path = check_training_status()

            if history:
                show_improvement_trend(history)
                show_learning_curves_plot(plot_path)

            # Wait for next check
            print("\n[INFO] Refreshing in 30 seconds... (Press Ctrl+C to exit)")
            time.sleep(30)

    except KeyboardInterrupt:
        print("\n\n[STOPPED] Monitoring stopped by user")
        print("[INFO] Training continues in background")
        print(f"[INFO] Check logs: tail -f experiments/model_training.log")

if __name__ == "__main__":
    main()
