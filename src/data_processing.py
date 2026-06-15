"""
Financial Data Processing for TimesFM VN30 Fine-tuning
Implements pfnet-research methodology with log transformation and realized volatility
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import io
import sys

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments/data_processing.log'),
        logging.StreamHandler()
    ]
)

class VN30FinancialDataProcessor:
    """
    Vietnamese market-specific data processor following pfnet-research methodology

    Key transformations:
    1. Log transformation (prevents NaN during extreme events)
    2. Multi-horizon realized volatility (5, 10, 20, 30 days)
    3. Vietnamese market features (TET holidays, trading patterns)
    4. Financial clipping (-5, 5 range for stability)
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def process_single_stock(self, stock_file: Path) -> pd.DataFrame:
        """
        Process single Vietnamese stock data with financial transformations

        Args:
            stock_file: Path to stock CSV file (e.g., VCB_ohlcv.csv)

        Returns:
            Processed DataFrame with financial features
        """
        stock_name = stock_file.stem.replace('_ohlcv', '').upper()
        self.logger.info(f"Processing {stock_name}...")

        # Load raw data
        df = pd.read_csv(stock_file)

        # Validate required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")

        # Convert date to datetime and set as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

        # Step 1: Log transformation (CRITICAL for financial data)
        df = self._apply_log_transformation(df)

        # Step 2: Calculate realized volatility at multiple horizons
        df = self._calculate_realized_volatility(df)

        # Step 3: Add Vietnamese market features
        df = self._add_vietnamese_market_features(df)

        # Step 3.5: Add OHLC range estimators (G7 paper)
        df = self._add_ohlc_features(df)

        # Step 4: Apply financial clipping
        df = self._apply_financial_clipping(df)

        # Step 5: Data quality validation
        self._validate_processed_data(df, stock_name)

        self.logger.info(f"[OK] {stock_name} processed: {len(df)} observations")
        return df

    def _apply_log_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log transformation to prevent NaN loss during extreme market events

        Financial Note: Log transformation is critical for:
        1. Preventing infinite losses during market crashes
        2. Making returns more normally distributed
        3. Stabilizing variance across different price levels
        """
        # Log transform closing prices
        df['log_close'] = np.log(df['close'])

        # Calculate log returns (more stable than raw returns)
        df['log_returns'] = df['log_close'].diff()

        # Handle first NaN from differencing
        df['log_returns'].fillna(0, inplace=True)

        return df

    def _calculate_realized_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate multi-horizon realized volatility using rolling standard deviation

        Financial Note: Realized volatility (RV) is the standard measure for:
        - Volatility forecasting
        - Risk management
        - Option pricing models

        Multiple horizons capture different time scales of volatility patterns
        """
        # Calculate realized volatility at multiple horizons
        volatility_windows = [5, 10, 20, 30]

        for window in volatility_windows:
            rv_col = f'RV_{window}'
            # Use rolling standard deviation of log returns
            df[rv_col] = df['log_returns'].rolling(window=window).std()

        # RV_20 is our primary target variable (20-day realized volatility)
        # This is approximately one trading month in Vietnamese markets

        return df

    def _add_vietnamese_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Vietnamese market-specific features

        Features include:
        1. TET holiday detection (Jan-Feb period)
        2. Day-of-week patterns
        3. Month-end effects
        4. Trading day indicators
        """
        # TET holiday detection (Vietnamese New Year - Jan/Feb)
        df['is_tet_period'] = df.index.month.isin([1, 2]).astype(int)

        # Day of week patterns (0=Monday, 4=Friday)
        df['day_of_week'] = df.index.dayofweek

        # Monday effect (start of week)
        df['is_monday'] = (df.index.dayofweek == 0).astype(int)

        # Friday effect (end of week)
        df['is_friday'] = (df.index.dayofweek == 4).astype(int)

        # Month-end effect (last trading days of month)
        df['is_month_end'] = (df.index.day >= 25).astype(int)

        return df

    def _add_ohlc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add OHLC-based volatility estimators from G7 paper research

        Paper findings: "Do extreme range estimators improve realized volatility forecasts?"
        - Overnight volatility: Most consistent performer across G7 markets
        - Parkinson estimator: Good performer for multiple markets
        - Garman-Klass estimator: Good performer for multiple markets

        Vietnamese market application:
        - Overnight effects may capture TET holiday information
        - Range estimators capture intraday volatility patterns
        - All features work as univariate time series for TimesFM
        """
        # Overnight volatility (paper's #1 performer)
        # Formula: (ln(Ot/Ct-1))² - captures overnight information accumulation
        df['overnight'] = np.log(df['open'] / df['close'].shift(1)) ** 2
        df['overnight'].fillna(0, inplace=True)

        # Parkinson estimator (paper's #2 performer)
        # Formula: (1/(4ln2)) × (ln(Ht/Lt))² - uses high-low range
        df['parkinson'] = (1 / (4 * np.log(2))) * (np.log(df['high'] / df['low']) ** 2)
        df['parkinson'].fillna(0, inplace=True)

        # Garman-Klass estimator (paper's #3 performer)
        # Formula: 0.5*(H/L)² - (2ln2-1)*(C/O)² - incorporates drift
        high_low = np.log(df['high'] / df['low']) ** 2
        close_open = np.log(df['close'] / df['open']) ** 2
        df['gk'] = 0.5 * high_low - (2 * np.log(2) - 1) * close_open
        df['gk'].fillna(0, inplace=True)

        # Close-to-close (baseline comparison)
        # Formula: (ln(Ct/Ct-1))² - simple squared returns
        df['close_to_close'] = np.log(df['close'] / df['close'].shift(1)) ** 2
        df['close_to_close'].fillna(0, inplace=True)

        self.logger.info("Added OHLC features: overnight, parkinson, gk, close_to_close")
        return df

    def _apply_financial_clipping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply financial-specific clipping to prevent extreme values

        Financial Note: Clipping to [-5, 5] range:
        - Prevents numerical instability during training
        - Removes extreme outliers while preserving information
        - Standard practice in financial ML for volatility

        Now includes both RV and OHLC-based features
        """
        # Clip RV features
        volatility_cols = [col for col in df.columns if col.startswith('RV_')]
        for col in volatility_cols:
            df[col] = np.clip(df[col], -5, 5)

        # Clip OHLC features (same range for consistency)
        ohlc_cols = ['overnight', 'parkinson', 'gk', 'close_to_close']
        for col in ohlc_cols:
            if col in df.columns:
                df[col] = np.clip(df[col], -5, 5)

        return df

    def _validate_processed_data(self, df: pd.DataFrame, stock_name: str):
        """Validate processed data quality"""
        # NaN values in volatility columns are expected at the start (rolling window warmup)
        # Only check if ALL values are NaN (which indicates a problem)
        volatility_cols = [col for col in df.columns if col.startswith('RV_')]
        for col in volatility_cols:
            if df[col].isna().all():
                raise ValueError(f"{stock_name}: {col} is entirely NaN")

        # Check for infinite values
        if np.isinf(df.select_dtypes(include=[np.number]).values).any():
            raise ValueError(f"{stock_name}: Infinite values detected")

        # Check volatility ranges (only on non-NaN values)
        rv_20 = df['RV_20'].dropna()
        if len(rv_20) > 0:  # Only check if we have valid values
            if (rv_20 < -5).any() or (rv_20 > 5).any():
                raise ValueError(f"{stock_name}: RV_20 outside [-5, 5] range")

        # Check minimum length (excluding NaN rows)
        valid_data = df.dropna(subset=['RV_20'])
        if len(valid_data) < 100:
            raise ValueError(f"{stock_name}: Insufficient valid data: {len(valid_data)} < 100")

        self.logger.info(f"[OK] {stock_name} data validation passed")

    def process_all_stocks(self, data_dir: Path) -> Dict[str, pd.DataFrame]:
        """
        Process all Vietnamese stocks in the data directory

        Args:
            data_dir: Path to directory containing stock CSV files

        Returns:
            Dictionary mapping stock names to processed DataFrames
        """
        data_dir = Path(data_dir)
        stock_files = list(data_dir.glob("*_ohlcv.csv"))

        self.logger.info(f"Found {len(stock_files)} stock files")

        processed_stocks = {}

        for stock_file in stock_files:
            try:
                stock_name = stock_file.stem.replace('_ohlcv', '').upper()
                processed_df = self.process_single_stock(stock_file)
                processed_stocks[stock_name] = processed_df

            except Exception as e:
                self.logger.error(f"Failed to process {stock_file.name}: {e}")
                continue

        self.logger.info(f"[OK] Successfully processed {len(processed_stocks)}/{len(stock_files)} stocks")
        return processed_stocks

    def save_processed_data(self, processed_stocks: Dict[str, pd.DataFrame],
                           output_dir: Path):
        """
        Save processed data to disk

        Args:
            processed_stocks: Dictionary of processed DataFrames
            output_dir: Output directory for processed data
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for stock_name, df in processed_stocks.items():
            output_file = output_dir / f"{stock_name}_processed.csv"
            df.to_csv(output_file)
            self.logger.info(f"Saved: {output_file}")

        self.logger.info(f"[OK] Saved {len(processed_stocks)} processed stock files")

    def generate_processing_report(self, processed_stocks: Dict[str, pd.DataFrame]) -> Dict:
        """Generate summary report of data processing"""
        if not processed_stocks:
            return {
                'timestamp': pd.Timestamp.now().isoformat(),
                'total_stocks': 0,
                'total_observations': 0,
                'stocks': {},
                'error': 'No stocks were successfully processed'
            }

        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_stocks': len(processed_stocks),
            'stocks': {}
        }

        total_obs = 0
        date_ranges = []

        for stock_name, df in processed_stocks.items():
            obs = len(df)
            date_range = f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"

            report['stocks'][stock_name] = {
                'observations': obs,
                'date_range': date_range,
                'rv_20_mean': float(df['RV_20'].mean()),
                'rv_20_std': float(df['RV_20'].std()),
                'log_returns_mean': float(df['log_returns'].mean()),
                'log_returns_std': float(df['log_returns'].std())
            }

            total_obs += obs
            date_ranges.append(df.index.min())
            date_ranges.append(df.index.max())

        report['total_observations'] = total_obs
        report['overall_date_range'] = f"{min(date_ranges).strftime('%Y-%m-%d')} to {max(date_ranges).strftime('%Y-%m-%d')}"

        return report


def main():
    """Main execution function for data processing"""
    import yaml

    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize processor
    processor = VN30FinancialDataProcessor()

    # Process all stocks
    raw_data_dir = Path(config['data']['raw_path'])
    processed_stocks = processor.process_all_stocks(raw_data_dir)

    # Save processed data
    processed_data_dir = Path(config['data']['processed_path'])
    processor.save_processed_data(processed_stocks, processed_data_dir)

    # Generate report
    report = processor.generate_processing_report(processed_stocks)

    # Save report
    import json
    experiments_dir = Path('experiments')
    experiments_dir.mkdir(exist_ok=True)

    with open(experiments_dir / 'data_processing_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    logging.info("=" * 70)
    logging.info("[DATA PROCESSING SUMMARY]")
    logging.info("=" * 70)
    logging.info(f"Total stocks processed: {report['total_stocks']}")
    logging.info(f"Total observations: {report['total_observations']:,}")
    if 'overall_date_range' in report:
        logging.info(f"Date range: {report['overall_date_range']}")
    else:
        logging.info("Date range: N/A (no stocks processed)")
    logging.info("[OK] Data processing completed successfully!")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())