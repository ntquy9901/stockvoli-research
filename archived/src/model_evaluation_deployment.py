"""
Model Evaluation and Deployment System
Complete system for evaluating TimesFM models and deploying for production use

Based on TimesFM research paper methodology and production deployment best practices
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any
import yaml
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation system for TimesFM volatility predictions

    Evaluates model performance across multiple dimensions and use cases
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize model evaluator"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.evaluation_metrics = self.config['validation']['metrics']
        logger.info("Initialized model evaluator")

    def calculate_point_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict:
        """
        Calculate point prediction metrics

        Args:
            actual: Actual values
            predicted: Predicted values

        Returns:
            Dictionary of metrics
        """
        # Handle missing values
        valid_mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = actual[valid_mask]
        predicted_clean = predicted[valid_mask]

        metrics = {}

        # MAE
        metrics['mae'] = np.mean(np.abs(actual_clean - predicted_clean))

        # RMSE
        metrics['rmse'] = np.sqrt(np.mean((actual_clean - predicted_clean) ** 2))

        # MAPE
        if np.all(actual_clean != 0):
            metrics['mape'] = np.mean(np.abs((actual_clean - predicted_clean) / actual_clean)) * 100
        else:
            metrics['mape'] = np.nan

        # Correlation
        if len(actual_clean) > 1:
            metrics['correlation'] = np.corrcoef(actual_clean, predicted_clean)[0, 1]
        else:
            metrics['correlation'] = np.nan

        # R-squared
        if len(actual_clean) > 1:
            ss_res = np.sum((actual_clean - predicted_clean) ** 2)
            ss_tot = np.sum((actual_clean - np.mean(actual_clean)) ** 2)
            metrics['r_squared'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        else:
            metrics['r_squared'] = np.nan

        # Max error
        metrics['max_error'] = np.max(np.abs(actual_clean - predicted_clean))

        # Bias (mean error)
        metrics['bias'] = np.mean(predicted_clean - actual_clean)

        # Sample size
        metrics['samples'] = len(actual_clean)

        return metrics

    def calculate_volatility_specific_metrics(self, actual: np.ndarray,
                                            predicted: np.ndarray) -> Dict:
        """
        Calculate volatility-specific metrics

        Args:
            actual: Actual volatility values
            predicted: Predicted volatility values

        Returns:
            Dictionary of volatility metrics
        """
        valid_mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = actual[valid_mask]
        predicted_clean = predicted[valid_mask]

        metrics = {}

        # Volatility direction accuracy
        if len(actual_clean) > 1:
            actual_direction = np.diff(actual_clean) > 0
            predicted_direction = np.diff(predicted_clean) > 0
            metrics['direction_accuracy'] = np.mean(actual_direction == predicted_direction)
        else:
            metrics['direction_accuracy'] = np.nan

        # Volatility regime classification accuracy
        actual_regime = pd.qcut(actual_clean, q=3, labels=['Low', 'Medium', 'High'])
        predicted_regime = pd.qcut(predicted_clean, q=3, labels=['Low', 'Medium', 'High'])
        metrics['regime_accuracy'] = np.mean(actual_regime == predicted_regime)

        # Tail risk assessment (high volatility periods)
        high_vol_threshold = np.quantile(actual_clean, 0.9)
        actual_high_vol = actual_clean >= high_vol_threshold
        predicted_high_vol = predicted_clean >= high_vol_threshold
        metrics['tail_risk_recall'] = np.sum(predicted_high_vol & actual_high_vol) / max(np.sum(actual_high_vol), 1)

        return metrics

    def calculate_temporal_stability(self, predictions: np.ndarray,
                                   actual: np.ndarray,
                                   time_index: pd.DatetimeIndex) -> Dict:
        """
        Calculate temporal stability metrics

        Args:
            predictions: Prediction array
            actual: Actual values
            time_index: Time index for the data

        Returns:
            Dictionary of temporal metrics
        """
        df = pd.DataFrame({
            'predictions': predictions,
            'actual': actual
        }, index=time_index)

        metrics = {}

        # Weekly performance variation
        if len(df) > 7:
            weekly_errors = df.resample('W').apply(
                lambda x: np.mean(np.abs(x['actual'] - x['predictions']))
            )
            metrics['weekly_error_std'] = np.std(weekly_errors)

        # Monthly performance variation
        if len(df) > 30:
            monthly_errors = df.resample('M').apply(
                lambda x: np.mean(np.abs(x['actual'] - x['predictions']))
            )
            metrics['monthly_error_std'] = np.std(monthly_errors)

        # Trend following ability
        if len(df) > 30:
            actual_trend = df['actual'].rolling(window=30).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            pred_trend = df['predictions'].rolling(window=30).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            metrics['trend_correlation'] = np.corrcoef(actual_trend.dropna(), pred_trend.dropna())[0, 1]
        else:
            metrics['trend_correlation'] = np.nan

        return metrics

    def evaluate_model_performance(self, actual_data: Dict[str, np.ndarray],
                                 predictions: Dict[str, np.ndarray],
                                 time_indices: Optional[Dict[str, pd.DatetimeIndex]] = None) -> Dict:
        """
        Comprehensive model evaluation

        Args:
            actual_data: Dictionary of actual values by stock
            predictions: Dictionary of predictions by stock
            time_indices: Optional time indices for temporal analysis

        Returns:
            Comprehensive evaluation results
        """
        logger.info("Starting comprehensive model evaluation...")

        evaluation_results = {}

        for symbol in actual_data.keys():
            if symbol not in predictions:
                logger.warning(f"No predictions for {symbol}")
                continue

            logger.info(f"Evaluating {symbol}...")

            actual = actual_data[symbol]
            predicted = predictions[symbol]

            # Point metrics
            point_metrics = self.calculate_point_metrics(actual, predicted)

            # Volatility-specific metrics
            vol_metrics = self.calculate_volatility_specific_metrics(actual, predicted)

            # Temporal stability (if time index provided)
            temporal_metrics = {}
            if time_indices and symbol in time_indices:
                temporal_metrics = self.calculate_temporal_stability(predicted, actual, time_indices[symbol])

            evaluation_results[symbol] = {
                'point_metrics': point_metrics,
                'volatility_metrics': vol_metrics,
                'temporal_metrics': temporal_metrics
            }

        # Calculate aggregate metrics
        aggregate_results = self._calculate_aggregate_metrics(evaluation_results)

        return {
            'individual_results': evaluation_results,
            'aggregate_results': aggregate_results
        }

    def _calculate_aggregate_metrics(self, individual_results: Dict) -> Dict:
        """Calculate aggregate metrics across all stocks"""
        aggregate = {
            'total_stocks': len(individual_results),
            'avg_mae': np.nan,
            'avg_rmse': np.nan,
            'avg_correlation': np.nan,
            'avg_direction_accuracy': np.nan
        }

        mae_values = []
        rmse_values = []
        correlation_values = []
        direction_accuracy_values = []

        for symbol, results in individual_results.items():
            point_metrics = results.get('point_metrics', {})
            vol_metrics = results.get('volatility_metrics', {})

            if 'mae' in point_metrics and not np.isnan(point_metrics['mae']):
                mae_values.append(point_metrics['mae'])

            if 'rmse' in point_metrics and not np.isnan(point_metrics['rmse']):
                rmse_values.append(point_metrics['rmse'])

            if 'correlation' in point_metrics and not np.isnan(point_metrics['correlation']):
                correlation_values.append(point_metrics['correlation'])

            if 'direction_accuracy' in vol_metrics and not np.isnan(vol_metrics['direction_accuracy']):
                direction_accuracy_values.append(vol_metrics['direction_accuracy'])

        if mae_values:
            aggregate['avg_mae'] = np.mean(mae_values)
        if rmse_values:
            aggregate['avg_rmse'] = np.mean(rmse_values)
        if correlation_values:
            aggregate['avg_correlation'] = np.mean(correlation_values)
        if direction_accuracy_values:
            aggregate['avg_direction_accuracy'] = np.mean(direction_accuracy_values)

        return aggregate


class ModelDeploymentSystem:
    """
    Production deployment system for TimesFM volatility predictions

    Handles model loading, prediction serving, and monitoring
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize deployment system"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Model storage
        self.model = None
        self.model_version = None
        self.model_metadata = {}

        # Prediction history
        self.prediction_history = []

        # Performance monitoring
        self.performance_metrics = []

        logger.info("Initialized deployment system")

    def load_model_for_deployment(self, model_path: str, version: str = "latest"):
        """
        Load trained model for deployment

        Args:
            model_path: Path to model checkpoint
            version: Model version identifier
        """
        logger.info(f"Loading model {version} from {model_path}...")

        try:
            # Load model
            model_file = Path(model_path)

            if model_file.suffix == '.pt' or model_file.suffix == '.pth':
                # PyTorch model
                self.model = torch.load(model_file)
            elif model_file.is_dir():
                # Transformers model
                from transformers import AutoModelForTimeSeriesForecasting
                self.model = AutoModelForTimeSeriesForecasting.from_pretrained(str(model_file))
            else:
                raise ValueError(f"Unknown model format: {model_file.suffix}")

            self.model_version = version
            self.model_metadata = {
                'loaded_at': datetime.now().isoformat(),
                'version': version,
                'path': str(model_file)
            }

            logger.info(f"✓ Model {version} loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict_volatility(self, context_data: np.ndarray,
                         symbols: Optional[List[str]] = None) -> Dict:
        """
        Make volatility predictions

        Args:
            context_data: Context features [samples, sequence_length, features]
            symbols: Optional list of stock symbols

        Returns:
            Dictionary with predictions and metadata
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_for_deployment() first.")

        logger.info(f"Making predictions for {context_data.shape[0]} samples...")

        try:
            self.model.eval()

            with torch.no_grad():
                context_tensor = torch.tensor(context_data, dtype=torch.float32)

                # Add batch dimension if needed
                if context_tensor.dim() == 2:
                    context_tensor = context_tensor.unsqueeze(0)

                # Make predictions
                if hasattr(self.model, 'forward'):
                    outputs = self.model(context_tensor)
                    predictions = outputs.squeeze().cpu().numpy()
                else:
                    predictions = self.model(context_tensor).squeeze().cpu().numpy()

            # Create prediction result
            result = {
                'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
                'model_version': self.model_version,
                'timestamp': datetime.now().isoformat(),
                'symbols': symbols if symbols else ['unknown']
            }

            # Store in history
            self.prediction_history.append(result)

            logger.info(f"✓ Predictions completed: {len(predictions)} samples")

            return result

        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

    def batch_predict(self, context_data_dict: Dict[str, np.ndarray]) -> Dict:
        """
        Batch predictions for multiple stocks

        Args:
            context_data_dict: Dictionary of context data by stock

        Returns:
            Dictionary of predictions by stock
        """
        logger.info(f"Making batch predictions for {len(context_data_dict)} stocks...")

        results = {}

        for symbol, context_data in context_data_dict.items():
            try:
                prediction = self.predict_volatility(context_data, [symbol])
                results[symbol] = prediction

            except Exception as e:
                logger.error(f"Error predicting {symbol}: {e}")
                results[symbol] = {'error': str(e)}

        success_count = sum(1 for r in results.values() if 'error' not in r)
        logger.info(f"Batch prediction completed: {success_count}/{len(context_data_dict)} successful")

        return results

    def update_performance_metrics(self, actual_values: Dict[str, float],
                                 predicted_values: Dict[str, float]):
        """
        Update performance monitoring metrics

        Args:
            actual_values: Dictionary of actual values
            predicted_values: Dictionary of predicted values
        """
        for symbol in actual_values.keys():
            if symbol in predicted_values:
                actual = actual_values[symbol]
                predicted = predicted_values[symbol]

                error = abs(actual - predicted)
                squared_error = (actual - predicted) ** 2

                metric = {
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'error': error,
                    'squared_error': squared_error,
                    'actual': actual,
                    'predicted': predicted,
                    'model_version': self.model_version
                }

                self.performance_metrics.append(metric)

        # Alert if performance degrades
        if len(self.performance_metrics) > 100:
            recent_errors = [m['error'] for m in self.performance_metrics[-100:]]
            avg_error = np.mean(recent_errors)

            threshold = self.config.get('monitoring', {}).get('alert', {}).get('threshold', 0.1)
            if avg_error > threshold:
                logger.warning(f"⚠ Performance degradation detected: MAE={avg_error:.6f}")

    def get_performance_report(self) -> Dict:
        """
        Generate performance monitoring report

        Returns:
            Performance report
        """
        if not self.performance_metrics:
            return {'status': 'no_data', 'message': 'No performance metrics collected yet'}

        # Calculate overall metrics
        errors = [m['error'] for m in self.performance_metrics]
        squared_errors = [m['squared_error'] for m in self.performance_metrics]

        report = {
            'total_predictions': len(self.performance_metrics),
            'avg_error': np.mean(errors),
            'std_error': np.std(errors),
            'rmse': np.sqrt(np.mean(squared_errors)),
            'max_error': np.max(errors),
            'model_version': self.model_version,
            'last_updated': self.performance_metrics[-1]['timestamp'] if self.performance_metrics else None
        }

        # Per-symbol metrics
        symbol_metrics = {}
        for metric in self.performance_metrics:
            symbol = metric['symbol']
            if symbol not in symbol_metrics:
                symbol_metrics[symbol] = []
            symbol_metrics[symbol].append(metric['error'])

        report['symbol_performance'] = {}
        for symbol, symbol_errors in symbol_metrics.items():
            report['symbol_performance'][symbol] = {
                'avg_error': np.mean(symbol_errors),
                'predictions': len(symbol_errors)
            }

        return report

    def export_model_package(self, export_path: str):
        """
        Export model package for deployment

        Args:
            export_path: Path to export model package
        """
        logger.info(f"Exporting model package to {export_path}...")

        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        if self.model is not None:
            model_save_path = export_dir / "model.pt"
            torch.save(self.model, model_save_path)

        # Save metadata
        metadata_path = export_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)

        # Save configuration
        config_path = export_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"✓ Model package exported to {export_path}")


class ProductionPredictionAPI:
    """
    Production API for volatility predictions

    Provides REST API interface for model predictions
    """

    def __init__(self, deployment_system: ModelDeploymentSystem):
        """Initialize API with deployment system"""
        self.deployment_system = deployment_system
        self.api_running = False

    def start_api_server(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Start FastAPI server for predictions

        Args:
            host: Server host
            port: Server port
        """
        try:
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
            from typing import List
            import uvicorn

            app = FastAPI(title="TimesFM Volatility Prediction API")

            class PredictionRequest(BaseModel):
                context_data: List[List[float]]
                symbols: Optional[List[str]] = None

            class PredictionResponse(BaseModel):
                predictions: List[float]
                model_version: str
                timestamp: str
                symbols: List[str]

            @app.post("/predict", response_model=PredictionResponse)
            async def predict(request: PredictionRequest):
                """Make volatility predictions"""
                try:
                    context_array = np.array(request.context_data, dtype=np.float32)
                    result = self.deployment_system.predict_volatility(
                        context_array,
                        request.symbols
                    )
                    return result
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

            @app.get("/health")
            async def health_check():
                """Health check endpoint"""
                return {
                    "status": "healthy",
                    "model_loaded": self.deployment_system.model is not None,
                    "model_version": self.deployment_system.model_version
                }

            @app.get("/performance")
            async def get_performance():
                """Get performance metrics"""
                return self.deployment_system.get_performance_report()

            @app.get("/models")
            async def list_models():
                """List available models"""
                models_dir = Path("models/timesfm")
                if models_dir.exists():
                    models = [str(m) for m in models_dir.glob("*")]
                    return {"models": models}
                return {"models": []}

            logger.info(f"Starting API server on {host}:{port}...")
            self.api_running = True
            uvicorn.run(app, host=host, port=port)

        except ImportError:
            logger.error("FastAPI not available. Install with: pip install fastapi uvicorn")
        except Exception as e:
            logger.error(f"Error starting API server: {e}")


def generate_evaluation_report(evaluation_results: Dict, output_path: str = "evaluation_report.json"):
    """
    Generate comprehensive evaluation report

    Args:
        evaluation_results: Results from model evaluation
        output_path: Path to save report
    """
    logger.info(f"Generating evaluation report: {output_path}")

    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return obj

    def recursive_convert(data):
        if isinstance(data, dict):
            return {key: recursive_convert(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [recursive_convert(item) for item in data]
        else:
            return convert_to_serializable(data)

    serializable_results = recursive_convert(evaluation_results)

    # Add metadata
    report = {
        'generated_at': datetime.now().isoformat(),
        'evaluation_type': 'comprehensive_model_evaluation',
        'results': serializable_results
    }

    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation report saved to {output_path}")


def main():
    """Main execution for testing"""
    logger.info("Initializing model evaluation and deployment system...")

    # Test components
    evaluator = ModelEvaluator()
    deployment_system = ModelDeploymentSystem()

    logger.info("Model evaluation and deployment system initialized successfully!")


if __name__ == "__main__":
    main()
