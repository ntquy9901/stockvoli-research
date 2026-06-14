"""
Utility functions for Vietnamese stock volatility forecasting system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def validate_data_quality(df: pd.DataFrame, symbol: str) -> Dict:
    """
    Validate data quality metrics

    Args:
        df: DataFrame to validate
        symbol: Stock symbol for reporting

    Returns:
        Dictionary of quality metrics
    """
    quality_report = {
        'symbol': symbol,
        'total_rows': len(df),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_dates': df.index.duplicated().sum(),
        'date_range': f"{df.index.min()} to {df.index.max()}",
        'completeness': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    }

    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    infinite_counts = np.isinf(df[numeric_cols]).sum()
    quality_report['infinite_values'] = infinite_counts.sum()

    return quality_report


def align_data_dates(stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Align all stock data to common dates

    Args:
        stock_data: Dictionary of stock DataFrames

    Returns:
        Dictionary with aligned data
    """
    # Find common dates across all stocks
    common_dates = None

    for symbol, data in stock_data.items():
        if common_dates is None:
            common_dates = set(data.index)
        else:
            common_dates = common_dates.intersection(set(data.index))

    if common_dates:
        common_dates = sorted(list(common_dates))
        logger.info(f"Found {len(common_dates)} common dates across {len(stock_data)} stocks")

        # Align all data to common dates
        aligned_data = {}
        for symbol, data in stock_data.items():
            aligned_data[symbol] = data.loc[common_dates]

        return aligned_data
    else:
        logger.error("No common dates found across stocks")
        return stock_data


def create_target_variables(df: pd.DataFrame, target_horizons: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
    """
    Create target variables for volatility forecasting

    Args:
        df: DataFrame with features
        target_horizons: List of forecasting horizons

    Returns:
        DataFrame with added target variables
    """
    df = df.copy()

    # Create target variables (future realized volatility)
    for horizon in target_horizons:
        if f'RV_{horizon}' in df.columns:
            df[f'Target_RV_{horizon}'] = df[f'RV_{horizon}'].shift(-horizon)

    return df


def split_train_validation_test(df: pd.DataFrame,
                                  train_ratio: float = 0.7,
                                  val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets

    Args:
        df: DataFrame to split
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set

    Returns:
        Tuple of (train, validation, test) DataFrames
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    return train_df, val_df, test_df


def calculate_data_statistics(stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate comprehensive statistics for all stocks

    Args:
        stock_data: Dictionary of stock DataFrames

    Returns:
        DataFrame with statistics
    """
    stats_list = []

    for symbol, data in stock_data.items():
        stats = {
            'symbol': symbol,
            'observations': len(data),
            'start_date': data.index.min(),
            'end_date': data.index.max(),
            'missing_pct': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
        }

        # Add statistics for key columns if they exist
        if 'Returns' in data.columns:
            stats['avg_return'] = data['Returns'].mean()
            stats['std_return'] = data['Returns'].std()

        if 'RV_20' in data.columns:
            stats['avg_volatility'] = data['RV_20'].mean()
            stats['std_volatility'] = data['RV_20'].std()

        if 'volume' in data.columns:
            stats['avg_volume'] = data['volume'].mean()

        stats_list.append(stats)

    return pd.DataFrame(stats_list)


def create_model_input_data(stock_data: Dict[str, pd.DataFrame],
                           feature_cols: List[str],
                           target_col: str = 'RV_20',
                           context_length: int = 512) -> Dict:
    """
    Create model input data following TimesFM format

    Args:
        stock_data: Dictionary of stock DataFrames
        feature_cols: List of feature columns to use
        target_col: Target variable column
        context_length: Context window length

    Returns:
        Dictionary with prepared data
    """
    prepared_data = {
        'context_data': [],
        'target_data': [],
        'symbols': [],
        'dates': []
    }

    for symbol, data in stock_data.items():
        # Ensure we have all required columns
        if target_col not in data.columns:
            logger.warning(f"Target column {target_col} not found for {symbol}")
            continue

        if not all(col in data.columns for col in feature_cols):
            logger.warning(f"Missing features for {symbol}")
            continue

        # Create samples with context length
        for i in range(context_length, len(data) - 1):
            context_features = data.iloc[i-context_length:i][feature_cols].values
            target_value = data.iloc[i][target_col]

            # Skip if missing values
            if np.isnan(context_features).any() or np.isnan(target_value):
                continue

            prepared_data['context_data'].append(context_features)
            prepared_data['target_data'].append(target_value)
            prepared_data['symbols'].append(symbol)
            prepared_data['dates'].append(data.index[i])

    logger.info(f"Prepared {len(prepared_data['context_data'])} samples across {len(stock_data)} stocks")

    return prepared_data


def log_data_summary(stock_data: Dict[str, pd.DataFrame]):
    """Log summary statistics of loaded data"""
    logger.info("="*50)
    logger.info("DATA SUMMARY")
    logger.info("="*50)

    for symbol, data in stock_data.items():
        logger.info(f"{symbol}: {len(data)} rows from {data.index.min()} to {data.index.max()}")

    logger.info("="*50)


def save_feature_importance(feature_importance: Dict[str, float], output_path: str):
    """
    Save feature importance scores

    Args:
        feature_importance: Dictionary of feature names and importance scores
        output_path: Path to save the results
    """
    import json

    with open(output_path, 'w') as f:
        json.dump(feature_importance, f, indent=2)

    logger.info(f"Feature importance saved to {output_path}")


def load_feature_importance(input_path: str) -> Dict[str, float]:
    """
    Load feature importance scores

    Args:
        input_path: Path to load from

    Returns:
        Dictionary of feature importance
    """
    import json

    with open(input_path, 'r') as f:
        feature_importance = json.load(f)

    return feature_importance


def normalize_features(df: pd.DataFrame, method: str = 'standard') -> Tuple[pd.DataFrame, Dict]:
    """
    Normalize features using specified method

    Args:
        df: DataFrame to normalize
        method: Normalization method ('standard', 'minmax', 'robust')

    Returns:
        Tuple of normalized DataFrame and normalization parameters
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    normalization_params = {}

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    normalized_df = df.copy()
    normalized_df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Store normalization parameters
    normalization_params['method'] = method
    normalization_params['scaler'] = scaler

    return normalized_df, normalization_params


def handle_missing_values(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """
    Handle missing values in data

    Args:
        df: DataFrame with missing values
        method: Method to handle missing values ('ffill', 'bfill', 'interpolate', 'drop')

    Returns:
        DataFrame with handled missing values
    """
    df = df.copy()

    if method == 'ffill':
        df.fillna(method='ffill', inplace=True)
    elif method == 'bfill':
        df.fillna(method='bfill', inplace=True)
    elif method == 'interpolate':
        df.interpolate(method='linear', inplace=True)
    elif method == 'drop':
        df.dropna(inplace=True)
    else:
        raise ValueError(f"Unknown missing value method: {method}")

    # Fill any remaining missing values
    df.fillna(0, inplace=True)

    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features

    Args:
        df: DataFrame with datetime index

    Returns:
        DataFrame with added time features
    """
    df = df.copy()

    # Basic time features
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['day_of_week'] = df.index.dayofweek
    df['day_of_year'] = df.index.dayofyear
    df['week_of_year'] = df.index.isocalendar().week.values
    df['quarter'] = df.index.quarter

    return df
