"""
Vietnamese Stock Volatility Data Preprocessing Module
Based on TimesFM and Moirai 2.0 research papers

This module handles data loading, preprocessing, and feature engineering
for Vietnamese stock market volatility forecasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VietnameseDataLoader:
    """
    Load and preprocess Vietnamese stock market data from raw CSV files.

    Based on the data structure in data/raw/prices/ with collection_summary.csv
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize data loader with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.raw_path = Path(self.config['data']['raw_path'])
        self.processed_path = Path(self.config['data']['processed_path'])

        # Create processed directory if it doesn't exist
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def load_collection_summary(self) -> pd.DataFrame:
        """Load collection summary to understand data availability"""
        summary_path = self.raw_path / "collection_summary.csv"

        if not summary_path.exists():
            logger.error(f"Collection summary not found at {summary_path}")
            return pd.DataFrame()

        summary_df = pd.read_csv(summary_path)
        logger.info(f"Loaded collection summary: {len(summary_df)} stocks")

        return summary_df

    def load_stock_data(self, symbol: str) -> pd.DataFrame:
        """
        Load OHLCV data for a specific stock symbol.

        Args:
            symbol: Stock symbol (e.g., 'VCB', 'VIC')

        Returns:
            DataFrame with OHLCV data
        """
        file_path = self.raw_path / f"{symbol}_ohlcv.csv"

        if not file_path.exists():
            logger.warning(f"Data file not found for {symbol} at {file_path}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            logger.info(f"Loaded {symbol}: {len(df)} days from {df.index.min()} to {df.index.max()}")
            return df

        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")
            return pd.DataFrame()

    def load_all_stocks(self) -> Dict[str, pd.DataFrame]:
        """Load all available stock data"""
        collection_summary = self.load_collection_summary()

        if collection_summary.empty:
            logger.error("No collection summary available")
            return {}

        # Get list of stocks from config or use all available
        stocks_to_load = self.config['data']['stocks']

        stock_data = {}
        for symbol in stocks_to_load:
            data = self.load_stock_data(symbol)
            if not data.empty:
                stock_data[symbol] = data

        logger.info(f"Successfully loaded {len(stock_data)} stocks")
        return stock_data


class VietnameseVolatilityFeatures:
    """
    Feature engineering for Vietnamese stock volatility forecasting.

    Based on TimesFM paper methodology for realized volatility features
    and Vietnamese market-specific patterns.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize feature engineer"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.volatility_windows = self.config['features']['volatility_windows']

    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create realized volatility features at multiple horizons.

        Based on TimesFM paper: "Realized volatility (RV) is the target variable"

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added volatility features
        """
        df = df.copy()

        # Calculate returns first
        df['Returns'] = df['close'].pct_change()
        df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1))

        # Realized Volatility at different horizons
        for window in self.volatility_windows:
            df[f'RV_{window}'] = df['Log_Returns'].rolling(window=window).std()
            df[f'RV_{window}_MA'] = df[f'RV_{window}'].rolling(window=10).mean()

        # Parkinson Volatility (more accurate than simple RV)
        df['Parkinson_Vol'] = np.sqrt(
            (np.log(df['high'] / df['low'])**2).rolling(window=20).mean() / (4 * np.log(2))
        )

        # Yang-Zhang Volatility (incorporates open and close)
        log_ho = np.log(df['high'] / df['open'])
        log_lo = np.log(df['low'] / df['open'])
        log_co = np.log(df['close'] / df['open'])

        df['Yang_Zhang_Vol'] = np.sqrt(
            (log_ho * log_ho.rolling(2).shift(1) +
             log_lo * log_lo.rolling(2).shift(1) +
             log_co * log_co.rolling(2).shift(1)).rolling(window=20).sum() / (3 * 20)
        )

        return df

    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical analysis indicators"""
        df = df.copy()

        # Moving averages
        for window in [10, 20, 50]:
            df[f'MA_{window}'] = df['close'].rolling(window=window).mean()

        # Exponential moving averages
        df['EMA_12'] = df['close'].ewm(span=12).mean()
        df['EMA_26'] = df['close'].ewm(span=26).mean()

        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        df['BB_Std'] = df['close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
        df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']

        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()

        return df

    def create_vietnamese_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Vietnamese market-specific features"""
        df = df.copy()

        # Day of week patterns (Vietnamese market active Mon-Fri)
        df['Day_Of_Week'] = df.index.dayofweek
        df['Is_Monday'] = (df['Day_Of_Week'] == 0).astype(int)
        df['Is_Friday'] = (df['Day_Of_Week'] == 4).astype(int)

        # Month start/end effects
        df['Month_Day'] = df.index.day
        df['Month_Start'] = (df['Month_Day'] <= 5).astype(int)
        df['Month_End'] = (df['Month_Day'] >= 25).astype(int)

        # Tet holiday period (January-February)
        df['Month'] = df.index.month
        df['Is_Tet_Period'] = df['Month'].isin([1, 2]).astype(int)

        # Earnings season patterns
        df['Is_Earnings_Season'] = df['Month'].isin([4, 5, 10, 11]).astype(int)

        return df

    def create_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market regime detection features"""
        df = df.copy()

        # Volatility regime classification
        if f'RV_20' in df.columns:
            df['Vol_Regime'] = pd.cut(
                df['RV_20'],
                bins=[0, df['RV_20'].quantile(0.33), df['RV_20'].quantile(0.67), float('inf')],
                labels=['Low', 'Medium', 'High']
            )

            # Volatility regime transitions
            df['Regime_Change'] = df['Vol_Regime'].ne(df['Vol_Regime'].shift(1)).astype(int)

        # Trend regime
        if 'MA_20' in df.columns and 'MA_50' in df.columns:
            df['Trend'] = np.where(df['MA_20'] > df['MA_50'], 1, -1)

        return df

    def process_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        logger.info(f"Processing features for data with {len(df)} rows")

        # Create all feature types
        df = self.create_volatility_features(df)
        df = self.create_technical_indicators(df)
        df = self.create_vietnamese_market_features(df)
        df = self.create_market_regime_features(df)

        logger.info(f"Final feature set: {len(df.columns)} features")
        return df


class IncrementalDataWindowManager:
    """
    Manage incremental learning data windows following TimesFM methodology.

    From TimesFM paper: "incremental fine-tuning, which allows the model to
    adapt to new financial return data over time, is essential"
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize window manager"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.window_size = self.config['incremental_learning']['window_size']
        self.window_overlap = self.config['incremental_learning']['window_overlap']
        self.context_length = self.config['features']['context_length']

    def create_incremental_windows(self, stock_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Create rolling time windows for incremental learning.

        Args:
            stock_data: Dictionary of stock DataFrames

        Returns:
            List of incremental windows
        """
        # Get all dates from the first stock
        first_stock = list(stock_data.keys())[0]
        all_dates = sorted(list(stock_data[first_stock].index))

        incremental_windows = []
        window_id = 1

        # Create rolling windows
        for i in range(0, len(all_dates) - self.window_size - self.context_length,
                      self.window_size - self.window_overlap):

            window_start = all_dates[i]
            window_end = all_dates[i + self.window_size + self.context_length]

            # Extract data for this window
            window_data = {}
            for symbol, data in stock_data.items():
                symbol_window = data[(data.index >= window_start) & (data.index <= window_end)]

                # Ensure sufficient context length
                if len(symbol_window) > self.context_length:
                    window_data[symbol] = symbol_window

            if window_data:
                incremental_windows.append({
                    'window_id': window_id,
                    'start_date': window_start,
                    'end_date': window_end,
                    'data': window_data
                })
                window_id += 1

        logger.info(f"Created {len(incremental_windows)} incremental windows")
        return incremental_windows

    def save_processed_windows(self, windows: List[Dict], output_dir: str = "data/processed"):
        """Save processed windows for later use"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for window in windows:
            window_file = output_path / f"window_{window['window_id']}.pkl"

            # Convert to serializable format
            window_data_serializable = {
                symbol: df.reset_index() for symbol, df in window['data'].items()
            }

            window_to_save = {
                'window_id': window['window_id'],
                'start_date': window['start_date'],
                'end_date': window['end_date'],
                'data': window_data_serializable
            }

            # Save as pickle (or could use other formats)
            import pickle
            with open(window_file, 'wb') as f:
                pickle.dump(window_to_save, f)

        logger.info(f"Saved {len(windows)} windows to {output_dir}")


def main():
    """Main execution function for data preprocessing"""
    logger.info("Starting Vietnamese stock data preprocessing...")

    # Initialize components
    loader = VietnameseDataLoader()
    feature_engineer = VietnameseVolatilityFeatures()
    window_manager = IncrementalDataWindowManager()

    # Load all stock data
    logger.info("Loading stock data...")
    stock_data = loader.load_all_stocks()

    if not stock_data:
        logger.error("No stock data loaded. Exiting.")
        return

    # Process features for all stocks
    logger.info("Creating features...")
    for symbol, data in stock_data.items():
        stock_data[symbol] = feature_engineer.process_all_features(data)
        logger.info(f"Processed {symbol}: {len(stock_data[symbol])} rows, {len(stock_data[symbol].columns)} features")

    # Save processed data
    logger.info("Saving processed data...")
    processed_path = Path(loader.config['data']['processed_path'])
    processed_path.mkdir(parents=True, exist_ok=True)

    for symbol, data in stock_data.items():
        output_file = processed_path / f"{symbol}_processed.csv"
        data.to_csv(output_file)
        logger.info(f"Saved {symbol} to {output_file}")

    # Create incremental windows
    logger.info("Creating incremental learning windows...")
    incremental_windows = window_manager.create_incremental_windows(stock_data)

    # Save windows
    window_manager.save_processed_windows(incremental_windows)

    logger.info("Data preprocessing completed successfully!")
    logger.info(f"Processed {len(stock_data)} stocks")
    logger.info(f"Created {len(incremental_windows)} incremental windows")


if __name__ == "__main__":
    main()
