"""
Resume Training Script
Continue training from a saved checkpoint with modified early stopping settings
"""

import sys
import logging
import yaml
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments/resume_training.log'),
        logging.StreamHandler()
    ]
)

def check_training_status():
    """Check if training was stopped by early stopping"""
    try:
        import json
        history_file = Path('experiments/training_history.json')

        if not history_file.exists():
            logging.warning("No training history found - may be first training run")
            return None

        with open(history_file, 'r') as f:
            history = json.load(f)

        if len(history) == 0:
            logging.warning("Empty training history")
            return None

        # Analyze why training stopped
        completed_epochs = len(history)
        last_epoch = history[-1]

        logging.info("=" * 70)
        logging.info("TRAINING STATUS CHECK")
        logging.info("=" * 70)
        logging.info(f"Completed epochs: {completed_epochs}")
        logging.info(f"Last epoch: {last_epoch['epoch']}")

        # Check if early stopping was triggered
        if completed_epochs < 100:  # Assuming 100 is default num_epochs
            logging.info(f"⚠️  Training stopped early at epoch {completed_epochs}")
            logging.info(f"⚠️  Likely cause: Early stopping triggered")

            # Analyze validation loss pattern
            val_losses = [epoch['val_metrics']['val_loss'] for epoch in history]
            best_idx = min(range(len(val_losses)), key=lambda i: val_losses[i])
            best_loss = val_losses[best_idx]
            best_epoch_num = history[best_idx]['epoch']

            logging.info(f"Best validation loss: {best_loss:.6f} at epoch {best_epoch_num}")
            logging.info(f"Last validation loss: {val_losses[-1]:.6f}")

            # Check if model was still improving
            if completed_epochs - best_epoch_num >= 5:
                logging.info("✓ Confirmed: Early stopping triggered (no improvement for 5+ epochs)")
                logging.info("✓ Recommendation: Increase early_stopping_patience or disable it")
            else:
                logging.info("? Training may have stopped for other reasons")

        else:
            logging.info("✓ Training completed full num_epochs")

        logging.info("=" * 70)
        return history

    except Exception as e:
        logging.error(f"Error checking training status: {e}")
        return None

def resume_training():
    """Resume training from checkpoint with increased patience"""
    logging.info("=" * 70)
    logging.info("RESUME TRAINING WITH INCREASED EARLY STOPPING PATIENCE")
    logging.info("=" * 70)
    logging.info("")
    logging.info("This will:")
    logging.info("1. Load the latest checkpoint")
    logging.info("2. Increase early_stopping_patience to 20 (from 5)")
    logging.info("3. Continue training for more epochs")
    logging.info("")
    logging.info("Expected outcome:")
    logging.info("- Model will continue training for up to 20 epochs without improvement")
    logging.info("- Learning curve will show more epochs")
    logging.info("- Better chance for convergence")
    logging.info("=" * 70)
    logging.info("")

    # Check current status
    history = check_training_status()

    if history and len(history) > 0:
        last_epoch = history[-1]['epoch']
        logging.info(f"Found previous training stopped at epoch {last_epoch}")
        logging.info(f"Can resume from epoch {last_epoch + 1}")
        logging.info("")
        logging.info("To resume training, uncomment and run:")
        logging.info("python -c \"from model_training_fixed import *; finetuner = TimesFMVN30Finetuner(); finetuner.load_checkpoint('best_model'); finetuner.train_model()\"")
    else:
        logging.info("No previous training found or error checking status")
        logging.info("Starting new training with increased patience...")

        # Load modified config
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        patience = config['training'].get('early_stopping_patience', 'Not set')
        logging.info(f"Current early_stopping_patience: {patience}")

if __name__ == "__main__":
    try:
        resume_training()
    except KeyboardInterrupt:
        logging.warning("\n[INTERRUPTED] Resume check stopped by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
