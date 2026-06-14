"""
Restore data files by removing incorrectly formatted June 2026 data

This script removes all June 2026 data that was crawled with wrong format
(raw VND instead of normalized prices in thousands).
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def restore_stock_file(stock_file: Path) -> bool:
    """
    Remove June 2026 data from stock file

    Args:
        stock_file: Path to stock CSV file

    Returns:
        True if file was restored, False if no changes needed
    """
    try:
        # Read data
        df = pd.read_csv(stock_file)
        original_len = len(df)

        if 'date' not in df.columns:
            logger.warning(f"No 'date' column in {stock_file.name}")
            return False

        df['date'] = pd.to_datetime(df['date'])

        # Remove June 2026 data
        df = df[~((df['date'].dt.year == 2026) & (df['date'].dt.month == 6))]

        # Also check for rows with raw VND format (prices > 10000)
        # This indicates the wrong format
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                # Find rows with raw VND format (price > 10000)
                raw_format = df[df[col] > 10000]
                if not raw_format.empty:
                    logger.info(f"Removing {len(raw_format)} rows with raw VND format from {stock_file.name}")
                    df = df[df[col] <= 10000]

        new_len = len(df)

        if new_len < original_len:
            # Save restored data
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            df.to_csv(stock_file, index=False)
            logger.info(f"[OK] Restored {stock_file.name}: {original_len} -> {new_len} rows (removed {original_len - new_len} rows)")
            return True
        else:
            logger.info(f"OK {stock_file.name}: No changes needed")
            return False

    except Exception as e:
        logger.error(f"❌ Error processing {stock_file.name}: {e}")
        return False

def main():
    """Restore all stock files"""
    prices_dir = Path('data/raw/prices')

    if not prices_dir.exists():
        logger.error(f"Directory not found: {prices_dir}")
        return 1

    stock_files = list(prices_dir.glob("*_ohlcv.csv"))

    logger.info(f"Found {len(stock_files)} stock files")
    logger.info("")

    restored_count = 0

    for stock_file in stock_files:
        if restore_stock_file(stock_file):
            restored_count += 1

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"RESTORATION COMPLETE: {restored_count}/{len(stock_files)} files restored")
    logger.info("=" * 70)

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
