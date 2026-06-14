"""
Vietnamese Stock Data Crawler for TimesFM VN30 Project

Crawls OHLCV data for VN30 stocks from Yahoo Finance.
This provides genuinely out-of-sample test data (June 2026) to avoid data leakage.

Usage:
    # Crawl June 2026 data (default)
    python src/crawl_stock_data.py

    # Crawl specific date range
    python src/crawl_stock_data.py --start 2026-06-01 --end 2026-06-30

    # Crawl specific stocks
    python src/crawl_stock_data.py --stocks VCB VIC VNM
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging
import time
import sys
import io
from typing import List, Optional

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments/data_crawling.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# VN30 Stocks with Yahoo Finance tickers
# Vietnamese stocks use .VN suffix on Yahoo Finance
VN30_STOCKS = {
    'ACB': 'ACB.VN',
    'BCM': 'BCM.VN',
    'BID': 'BID.VN',
    'BVH': 'BVH.VN',
    'CTG': 'CTG.VN',
    'FPT': 'FPT.VN',
    'GAS': 'GAS.VN',
    'GVR': 'GVR.VN',
    'HDB': 'HDB.VN',
    'HPG': 'HPG.VN',
    'MBB': 'MBB.VN',
    'MSN': 'MSN.VN',
    'MWG': 'MWG.VN',
    'NVL': 'NVL.VN',
    'PDR': 'PDR.VN',
    'PLX': 'PLX.VN',
    'POW': 'POW.VN',
    'SAB': 'SAB.VN',
    'SHB': 'SHB.VN',
    'SSB': 'SSB.VN',
    'SSI': 'SSI.VN',
    'STB': 'STB.VN',
    'TCB': 'TCB.VN',
    'TPB': 'TPB.VN',
    'VCB': 'VCB.VN',
    'VHM': 'VHM.VN',
    'VIB': 'VIB.VN',
    'VIC': 'VIC.VN',
    'VJC': 'VJC.VN',
    'VNM': 'VNM.VN'
}


class VietnameseStockCrawler:
    """
    Crawler for Vietnamese stock OHLCV data from Yahoo Finance

    Key features:
    - Downloads data for specific date ranges (out-of-sample testing)
    - Appends to existing data files
    - Handles network errors gracefully
    - Maintains exact format for TimesFM compatibility
    """

    def __init__(self, output_dir: str = 'data/raw/prices'):
        """
        Initialize crawler

        Args:
            output_dir: Directory to save crawled data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def crawl_stock(self, stock_symbol: str, yf_ticker: str,
                   start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Crawl single stock data from Yahoo Finance

        Args:
            stock_symbol: Local stock symbol (e.g., 'VCB')
            yf_ticker: Yahoo Finance ticker (e.g., 'VCB.VN')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            self.logger.info(f"Crawling {stock_symbol} from {start_date} to {end_date}")

            # Download data from Yahoo Finance
            ticker = yf.Ticker(yf_ticker)
            data = ticker.history(start=start_date, end=end_date)

            if data.empty:
                self.logger.warning(f"No data found for {stock_symbol} ({yf_ticker})")
                return None

            # Reset index to make date a column
            data.reset_index(inplace=True)

            # Rename columns to match our format
            data = data.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Select required columns
            data = data[['date', 'open', 'high', 'low', 'close', 'volume']]

            # Format date to YYYY-MM-DD
            data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')

            # Convert from raw VND to match existing dataset format
            # Existing data: prices in thousands (e.g., 60.5 = 60,500 VND)
            # Yahoo Finance: raw VND (e.g., 60500.0)
            # Conversion: divide by 1000
            data['open'] = (data['open'] / 1000).round(2)
            data['high'] = (data['high'] / 1000).round(2)
            data['low'] = (data['low'] / 1000).round(2)
            data['close'] = (data['close'] / 1000).round(2)

            # Convert volume to integer
            data['volume'] = data['volume'].astype(int)

            self.logger.info(f"✅ Downloaded {len(data)} rows for {stock_symbol}")

            # Rate limiting (respect Yahoo Finance)
            time.sleep(0.5)

            return data

        except Exception as e:
            self.logger.error(f"❌ Error crawling {stock_symbol}: {e}")
            return None

    def load_existing_data(self, stock_symbol: str) -> pd.DataFrame:
        """
        Load existing data for a stock

        Args:
            stock_symbol: Stock symbol (e.g., 'VCB')

        Returns:
            Existing data or empty DataFrame if not found
        """
        existing_file = self.output_dir / f"{stock_symbol}_ohlcv.csv"

        if existing_file.exists():
            try:
                existing_data = pd.read_csv(existing_file)
                existing_data['date'] = pd.to_datetime(existing_data['date'])
                self.logger.info(f"Loaded {len(existing_data)} existing rows for {stock_symbol}")
                return existing_data
            except Exception as e:
                self.logger.warning(f"Could not load existing data for {stock_symbol}: {e}")

        return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])

    def merge_and_deduplicate(self, existing_data: pd.DataFrame,
                            new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge existing and new data, removing duplicates

        Args:
            existing_data: Existing stock data
            new_data: Newly crawled data

        Returns:
            Merged data with duplicates removed
        """
        if existing_data.empty:
            return new_data

        # Convert dates
        new_data['date'] = pd.to_datetime(new_data['date'])

        # Concatenate
        merged = pd.concat([existing_data, new_data], ignore_index=True)

        # Remove duplicates (keep newest)
        merged = merged.drop_duplicates(subset=['date'], keep='last')

        # Sort by date
        merged = merged.sort_values('date').reset_index(drop=True)

        return merged

    def save_stock_data(self, stock_symbol: str, data: pd.DataFrame):
        """
        Save stock data to CSV

        Args:
            stock_symbol: Stock symbol (e.g., 'VCB')
            data: Stock OHLCV data
        """
        output_file = self.output_dir / f"{stock_symbol}_ohlcv.csv"

        # Format date before saving
        data_to_save = data.copy()
        data_to_save['date'] = pd.to_datetime(data_to_save['date']).dt.strftime('%Y-%m-%d')

        data_to_save.to_csv(output_file, index=False)

        self.logger.info(f"💾 Saved {len(data)} rows to {output_file}")

    def crawl_all_stocks(self, stock_symbols: Optional[List[str]] = None,
                        start_date: str = '2026-06-01',
                        end_date: Optional[str] = None) -> dict:
        """
        Crawl all VN30 stocks for specified date range

        Args:
            stock_symbols: List of stock symbols (None = all VN30)
            start_date: Start date (default: June 1, 2026)
            end_date: End date (default: today)

        Returns:
            Dictionary with crawl results
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        # Default: all VN30 stocks
        if stock_symbols is None:
            stock_symbols = list(VN30_STOCKS.keys())

        results = {
            'successful': [],
            'failed': [],
            'total_rows_added': 0
        }

        self.logger.info("=" * 70)
        self.logger.info(f"CRAWLING VIETNAMESE STOCK DATA")
        self.logger.info("=" * 70)
        self.logger.info(f"Date range: {start_date} to {end_date}")
        self.logger.info(f"Stocks: {len(stock_symbols)}")
        self.logger.info("")

        for stock_symbol in stock_symbols:
            if stock_symbol not in VN30_STOCKS:
                self.logger.warning(f"⚠️  Unknown stock: {stock_symbol} (skipping)")
                results['failed'].append(stock_symbol)
                continue

            yf_ticker = VN30_STOCKS[stock_symbol]

            # Crawl new data
            new_data = self.crawl_stock(stock_symbol, yf_ticker, start_date, end_date)

            if new_data is None or new_data.empty:
                results['failed'].append(stock_symbol)
                continue

            # Load existing data
            existing_data = self.load_existing_data(stock_symbol)
            rows_before = len(existing_data)

            # Merge and deduplicate
            merged_data = self.merge_and_deduplicate(existing_data, new_data)
            rows_after = len(merged_data)
            rows_added = rows_after - rows_before

            # Save merged data
            self.save_stock_data(stock_symbol, merged_data)

            results['successful'].append(stock_symbol)
            results['total_rows_added'] += rows_added

            self.logger.info(f"   Added {rows_added} new rows (total: {rows_after})")
            self.logger.info("")

        return results

    def check_data_coverage(self, stock_symbol: str,
                           start_date: str, end_date: str) -> dict:
        """
        Check data coverage for a stock

        Args:
            stock_symbol: Stock symbol
            start_date: Check start date
            end_date: Check end date

        Returns:
            Coverage statistics
        """
        existing_data = self.load_existing_data(stock_symbol)

        if existing_data.empty:
            return {
                'stock': stock_symbol,
                'total_rows': 0,
                'in_range': 0,
                'coverage': 0.0
            }

        existing_data['date'] = pd.to_datetime(existing_data['date'])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        in_range = existing_data[
            (existing_data['date'] >= start_dt) &
            (existing_data['date'] <= end_dt)
        ]

        return {
            'stock': stock_symbol,
            'total_rows': len(existing_data),
            'in_range': len(in_range),
            'coverage': len(in_range) / len(existing_data) if len(existing_data) > 0 else 0
        }


def main():
    """Main crawling function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Crawl Vietnamese stock data for TimesFM fine-tuning'
    )
    parser.add_argument(
        '--stocks',
        nargs='+',
        help='Stock symbols to crawl (default: all VN30)'
    )
    parser.add_argument(
        '--start',
        default='2026-06-01',
        help='Start date (YYYY-MM-DD, default: 2026-06-01)'
    )
    parser.add_argument(
        '--end',
        default=None,
        help='End date (YYYY-MM-DD, default: today)'
    )
    parser.add_argument(
        '--output',
        default='data/raw/prices',
        help='Output directory (default: data/raw/prices)'
    )

    args = parser.parse_args()

    # Initialize crawler
    crawler = VietnameseStockCrawler(output_dir=args.output)

    # Crawl data
    results = crawler.crawl_all_stocks(
        stock_symbols=args.stocks,
        start_date=args.start,
        end_date=args.end
    )

    # Print summary
    print("\n" + "=" * 70)
    print("CRAWL SUMMARY")
    print("=" * 70)
    print(f"✅ Successful: {len(results['successful'])} stocks")
    print(f"❌ Failed:      {len(results['failed'])} stocks")
    print(f"📊 Total rows added: {results['total_rows_added']}")
    print("")

    if results['successful']:
        print("SUCCESSFUL STOCKS:")
        for stock in results['successful']:
            print(f"  ✅ {stock}")

    if results['failed']:
        print("\nFAILED STOCKS:")
        for stock in results['failed']:
            print(f"  ❌ {stock}")

    print("\n" + "=" * 70)

    return 0 if len(results['failed']) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
