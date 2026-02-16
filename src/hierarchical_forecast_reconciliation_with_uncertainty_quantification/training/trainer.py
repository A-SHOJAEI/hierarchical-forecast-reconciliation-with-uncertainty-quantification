"""
Training pipeline for hierarchical ensemble forecasting with MLflow integration.

This module provides a comprehensive training framework that supports model
selection, hyperparameter optimization, and experiment tracking.
"""

import copy
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import TimeSeriesSplit

# Optional imports with graceful fallback
try:
    import mlflow
    import mlflow.pytorch
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ..data.loader import M5DataLoader, HierarchicalDataBuilder
from ..data.preprocessing import M5Preprocessor, HierarchyBuilder
from ..models.model import HierarchicalEnsembleForecaster
from ..evaluation.metrics import HierarchicalMetrics
from ..utils.config import load_config, setup_logging, set_random_seeds

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class HierarchicalForecastTrainer:
    """
    Comprehensive training pipeline for hierarchical ensemble forecasting.

    Manages the complete training process including data loading, preprocessing,
    model training, validation, and experiment tracking with MLflow.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        data_path: Optional[str] = None,
        experiment_name: Optional[str] = None
    ) -> None:
        """
        Initialize the training pipeline.

        Args:
            config: Configuration dictionary with all training parameters.
            data_path: Path to M5 data directory.
            experiment_name: MLflow experiment name.
        """
        self.config = config
        self.data_path = data_path or config.get('data', {}).get('path', 'data/m5')
        self.experiment_name = experiment_name or config.get('training', {}).get('experiment_name', 'hierarchical_forecast')

        # Set up logging
        logging_config = config.get('logging', {})
        setup_logging(
            level=logging_config.get('level', 'INFO'),
            log_file=logging_config.get('file'),
            console=logging_config.get('console', True)
        )
        self.logger = logging.getLogger(__name__)

        # Set random seeds
        seed = config.get('random_seed', 42)
        set_random_seeds(seed)

        # Initialize components
        self.data_loader = M5DataLoader(self.data_path)
        self.hierarchy_builder = HierarchicalDataBuilder(
            config['data']['aggregation_levels']
        )
        self.preprocessor = M5Preprocessor()
        self.evaluator = HierarchicalMetrics()

        # Training state
        self.is_fitted = False
        self.best_model: Optional[HierarchicalEnsembleForecaster] = None
        self.training_history: List[Dict[str, Any]] = []
        self.aggregation_matrix: Optional[sparse.csr_matrix] = None

        # MLflow setup
        self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        """Set up MLflow experiment tracking."""
        self.mlflow_enabled = False
        if not HAS_MLFLOW:
            self.logger.warning("MLflow not installed, tracking disabled")
            return
        try:
            mlflow.set_tracking_uri(self.config.get('training', {}).get('tracking_uri', 'mlruns'))
            mlflow.set_experiment(self.experiment_name)
            self.mlflow_enabled = True
            self.logger.info(f"MLflow experiment: {self.experiment_name}")
        except Exception as e:
            self.logger.warning(f"MLflow setup failed: {e}")

    def train(
        self,
        optimize_hyperparameters: bool = False,
        n_trials: Optional[int] = None
    ) -> HierarchicalEnsembleForecaster:
        """
        Execute the complete training pipeline.

        Args:
            optimize_hyperparameters: Whether to perform hyperparameter optimization.
            n_trials: Number of optimization trials (if optimizing).

        Returns:
            Trained ensemble forecaster.

        Raises:
            Exception: If training fails.
        """
        self.logger.info("Starting hierarchical forecast training pipeline...")

        try:
            # Start MLflow run if available
            mlflow_context = None
            if self.mlflow_enabled:
                try:
                    mlflow_context = mlflow.start_run(
                        run_name=self.config.get('training', {}).get('run_name')
                    )
                    mlflow_context.__enter__()
                    # Log configuration (flattened)
                    try:
                        flat_config = self._flatten_config(self.config)
                        mlflow.log_params(flat_config)
                    except Exception as e:
                        self.logger.warning(f"Failed to log params to MLflow: {e}")
                except Exception as e:
                    self.logger.warning(f"Failed to start MLflow run: {e}")
                    mlflow_context = None

            # Load and prepare data
            train_data, val_data, test_data = self._prepare_data()

            # Build hierarchical structure
            hierarchy_data = self._build_hierarchy(train_data)

            if optimize_hyperparameters and HAS_OPTUNA:
                # Hyperparameter optimization
                best_params = self._optimize_hyperparameters(
                    train_data, val_data, hierarchy_data, n_trials
                )
                self._update_config_with_best_params(best_params)

            # Train final model
            model = self._train_model(train_data, hierarchy_data)

            # Validate model
            val_metrics = self._validate_model(model, val_data, test_data)

            # Log results
            self._log_training_results(model, val_metrics)

            self.best_model = model
            self.is_fitted = True

            self.logger.info("Training pipeline completed successfully")

            # End MLflow run
            if mlflow_context is not None:
                try:
                    mlflow_context.__exit__(None, None, None)
                except Exception:
                    pass

            return model

        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            # End MLflow run on error
            if self.mlflow_enabled:
                try:
                    mlflow.end_run()
                except Exception:
                    pass
            raise

    def _prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and preprocess data for training."""
        self.logger.info("Preparing training data...")

        # Load M5 data
        sales_data, calendar_data, prices_data = self.data_loader.load_data()

        # Prepare time series data
        data_config = self.config['data']
        time_series_data = self.data_loader.prepare_time_series_data(
            start_day=1,
            end_day=data_config['train_days'] + data_config['validation_days'] + data_config['test_days'],
            min_nonzero_ratio=data_config.get('min_nonzero_ratio', 0.1)
        )

        # Add calendar and price features
        enriched_data = self.data_loader.add_calendar_features(time_series_data)
        if prices_data is not None:
            enriched_data = self.data_loader.add_price_features(enriched_data)

        # Preprocess data
        processed_data = self.preprocessor.fit_transform(enriched_data)

        # Create train/validation/test splits
        train_data, val_data, test_data = self.preprocessor.create_train_test_split(
            processed_data,
            data_config['train_days'],
            data_config['validation_days'],
            data_config['test_days']
        )

        self.logger.info(f"Data preparation complete:")
        self.logger.info(f"  Train: {train_data.shape}")
        self.logger.info(f"  Validation: {val_data.shape}")
        self.logger.info(f"  Test: {test_data.shape}")

        return train_data, val_data, test_data

    def _build_hierarchy(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Build hierarchical aggregations."""
        self.logger.info("Building hierarchical structure...")

        hierarchy_data = self.hierarchy_builder.build_hierarchy(data)

        # Build aggregation matrix
        matrix_builder = HierarchyBuilder(self.config['data']['aggregation_levels'])
        self.aggregation_matrix = matrix_builder.build_aggregation_matrix(hierarchy_data)

        self.logger.info(f"Hierarchy structure: {matrix_builder.get_hierarchy_structure(hierarchy_data)}")
        return hierarchy_data

    def _train_model(
        self,
        train_data: pd.DataFrame,
        hierarchy_data: Dict[str, pd.DataFrame]
    ) -> HierarchicalEnsembleForecaster:
        """Train the ensemble forecaster."""
        self.logger.info("Training ensemble forecaster...")

        start_time = time.time()

        # Initialize ensemble forecaster
        model = HierarchicalEnsembleForecaster(
            statistical_configs=self.config['models']['statistical'],
            deep_learning_configs=self.config['models']['deep_learning'],
            ensemble_weights=self.config['ensemble']['weights'],
            reconciler_config=self.config['reconciliation']
        )

        # Train the model
        model.fit(train_data, self.aggregation_matrix)

        training_time = time.time() - start_time
        self.logger.info(f"Model training completed in {training_time:.2f} seconds")

        # Log training time
        if self.mlflow_enabled:
            try:
                mlflow.log_metric("training_time_seconds", training_time)
            except Exception as e:
                self.logger.warning(f"Failed to log training time to MLflow: {e}")

        return model

    def _validate_model(
        self,
        model: HierarchicalEnsembleForecaster,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Validate the trained model."""
        self.logger.info("Validating model performance...")

        # Generate predictions for validation period
        if 'date' in val_data.columns:
            val_horizon = len(val_data['date'].unique())
        else:
            val_horizon = 28  # default

        val_predictions = model.predict(
            horizon=val_horizon,
            return_intervals=True,
            confidence_levels=[0.1, 0.05]
        )

        # Compute validation metrics
        val_metrics = self.evaluator.compute_all_metrics(
            predictions=val_predictions['forecasts'],
            actuals=self._extract_actuals(val_data),
            intervals=val_predictions,
            hierarchy_data={},
            confidence_levels=[0.1, 0.05]
        )

        # Log metrics to MLflow
        if self.mlflow_enabled:
            try:
                for metric_name, value in val_metrics.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        mlflow.log_metric(f"val_{metric_name}", value)
            except Exception as e:
                self.logger.warning(f"Failed to log metrics to MLflow: {e}")

        self.logger.info(f"Validation metrics: {val_metrics}")
        return val_metrics

    def _optimize_hyperparameters(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        hierarchy_data: Dict[str, pd.DataFrame],
        n_trials: Optional[int] = None
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        self.logger.info("Starting hyperparameter optimization...")

        if n_trials is None:
            n_trials = self.config.get('optimization', {}).get('n_trials', 50)

        # Create optimization study
        study = optuna.create_study(
            direction='minimize',
            study_name=f"{self.experiment_name}_optimization"
        )

        # Add MLflow callback if available
        mlflow_callback = None
        if self.mlflow_enabled:
            try:
                from optuna.integration.mlflow import MLflowCallback
                mlflow_callback = MLflowCallback(
                    tracking_uri=mlflow.get_tracking_uri(),
                    metric_name="val_loss"
                )
            except Exception as e:
                self.logger.warning(f"MLflow callback not available: {e}")

        # Define objective function
        def objective(trial: optuna.Trial) -> float:
            return self._objective_function(trial, train_data, val_data, hierarchy_data)

        # Run optimization
        callbacks = [mlflow_callback] if mlflow_callback is not None else []
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=self.config.get('optimization', {}).get('timeout'),
            callbacks=callbacks
        )

        best_params = study.best_params
        self.logger.info(f"Best hyperparameters: {best_params}")
        self.logger.info(f"Best validation score: {study.best_value}")

        # Log optimization results
        if self.mlflow_enabled:
            try:
                mlflow.log_params({k: str(v) for k, v in best_params.items()})
                mlflow.log_metric("best_validation_score", study.best_value)
                mlflow.log_metric("optimization_trials", len(study.trials))
            except Exception as e:
                self.logger.warning(f"Failed to log optimization results to MLflow: {e}")

        return best_params

    def _objective_function(
        self,
        trial: optuna.Trial,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        hierarchy_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Objective function for hyperparameter optimization."""
        try:
            # Sample hyperparameters
            sampled_params = self._sample_hyperparameters(trial)

            # Update configuration with sampled parameters
            trial_config = self._update_config_with_trial_params(sampled_params)

            # Create and train model with sampled parameters
            model = HierarchicalEnsembleForecaster(
                statistical_configs=trial_config['models']['statistical'],
                deep_learning_configs=trial_config['models']['deep_learning'],
                ensemble_weights=trial_config['ensemble']['weights'],
                reconciler_config=trial_config['reconciliation']
            )

            model.fit(train_data, self.aggregation_matrix)

            # Validate on validation set
            val_horizon = len(val_data['date'].unique())
            val_predictions = model.predict(horizon=val_horizon, return_intervals=False)

            # Compute validation loss (WRMSSE)
            actuals = self._extract_actuals(val_data)
            val_loss = self.evaluator.compute_wrmsse(
                val_predictions['forecasts'], actuals
            )

            return val_loss

        except Exception as e:
            self.logger.warning(f"Trial failed: {e}")
            return float('inf')

    def _sample_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample hyperparameters for optimization."""
        search_space = self.config.get('optimization', {}).get('search_space', {})

        params = {}

        # TFT parameters
        if 'tft_learning_rate' in search_space:
            params['tft_learning_rate'] = trial.suggest_float(
                'tft_learning_rate', *search_space['tft_learning_rate']
            )
        if 'tft_hidden_size' in search_space:
            params['tft_hidden_size'] = trial.suggest_int(
                'tft_hidden_size', *search_space['tft_hidden_size']
            )
        if 'tft_dropout' in search_space:
            params['tft_dropout'] = trial.suggest_float(
                'tft_dropout', *search_space['tft_dropout']
            )

        # N-BEATS parameters
        if 'nbeats_num_stacks' in search_space:
            params['nbeats_num_stacks'] = trial.suggest_int(
                'nbeats_num_stacks', *search_space['nbeats_num_stacks']
            )
        if 'nbeats_layer_widths' in search_space:
            params['nbeats_layer_widths'] = trial.suggest_int(
                'nbeats_layer_widths', *search_space['nbeats_layer_widths']
            )

        # Reconciliation parameters
        if 'reconciliation_lambda_reg' in search_space:
            params['reconciliation_lambda_reg'] = trial.suggest_float(
                'reconciliation_lambda_reg', *search_space['reconciliation_lambda_reg']
            )

        # Ensemble weights
        if 'ensemble_tft_weight' in search_space:
            params['ensemble_tft_weight'] = trial.suggest_float(
                'ensemble_tft_weight', *search_space['ensemble_tft_weight']
            )

        return params

    def _update_config_with_trial_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration with trial parameters."""
        trial_config = copy.deepcopy(self.config)

        # Update TFT parameters
        if 'tft_learning_rate' in params:
            trial_config['models']['deep_learning']['tft']['learning_rate'] = params['tft_learning_rate']
        if 'tft_hidden_size' in params:
            trial_config['models']['deep_learning']['tft']['hidden_size'] = params['tft_hidden_size']
        if 'tft_dropout' in params:
            trial_config['models']['deep_learning']['tft']['dropout'] = params['tft_dropout']

        # Update N-BEATS parameters
        if 'nbeats_num_stacks' in params:
            trial_config['models']['deep_learning']['nbeats']['num_stacks'] = params['nbeats_num_stacks']
        if 'nbeats_layer_widths' in params:
            trial_config['models']['deep_learning']['nbeats']['layer_widths'] = params['nbeats_layer_widths']

        # Update reconciliation parameters
        if 'reconciliation_lambda_reg' in params:
            trial_config['reconciliation']['lambda_reg'] = params['reconciliation_lambda_reg']

        # Update ensemble weights
        if 'ensemble_tft_weight' in params:
            tft_weight = params['ensemble_tft_weight']
            # Adjust other weights proportionally
            remaining_weight = 1.0 - tft_weight
            trial_config['ensemble']['weights']['tft'] = tft_weight
            trial_config['ensemble']['weights']['ets'] = remaining_weight * 0.4
            trial_config['ensemble']['weights']['arima'] = remaining_weight * 0.4
            trial_config['ensemble']['weights']['nbeats'] = remaining_weight * 0.2

        return trial_config

    def _update_config_with_best_params(self, best_params: Dict[str, Any]) -> None:
        """Update configuration with best parameters found during optimization."""
        self.config = self._update_config_with_trial_params(best_params)

    def _extract_actuals(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract actual values from data for evaluation."""
        actuals = {}
        for series_id, series_data in data.groupby('id'):
            actuals[series_id] = series_data.sort_values('date')['sales'].values
        return actuals

    def _log_training_results(
        self,
        model: HierarchicalEnsembleForecaster,
        metrics: Dict[str, float]
    ) -> None:
        """Log training results to MLflow."""
        if self.mlflow_enabled:
            try:
                # Note: model is not a PyTorch model, skip pytorch logging
                self.logger.info("Logging training results...")
            except Exception as e:
                self.logger.warning(f"Failed to log model to MLflow: {e}")

        # Log artifacts
        self._save_and_log_artifacts(model, metrics)

    def _save_and_log_artifacts(
        self,
        model: HierarchicalEnsembleForecaster,
        metrics: Dict[str, float]
    ) -> None:
        """Save and log training artifacts."""
        import pickle
        import json

        # Create artifacts directory
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        try:
            # Save configuration
            config_path = artifacts_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2, default=str)

            # Save metrics (convert numpy types to native Python)
            clean_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (np.integer,)):
                    clean_metrics[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    clean_metrics[k] = float(v)
                else:
                    clean_metrics[k] = v

            metrics_path = artifacts_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(clean_metrics, f, indent=2)

            # Save aggregation matrix
            if self.aggregation_matrix is not None:
                matrix_path = artifacts_dir / "aggregation_matrix.pkl"
                with open(matrix_path, 'wb') as f:
                    pickle.dump(self.aggregation_matrix, f)

            # Log to MLflow if available
            if self.mlflow_enabled:
                try:
                    mlflow.log_artifact(str(config_path))
                    mlflow.log_artifact(str(metrics_path))
                    if self.aggregation_matrix is not None:
                        mlflow.log_artifact(str(matrix_path))
                except Exception as e:
                    self.logger.warning(f"Failed to log artifacts to MLflow: {e}")

            self.logger.info("Artifacts saved and logged successfully")

        except Exception as e:
            self.logger.warning(f"Failed to save artifacts: {e}")

    def _flatten_config(self, config: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        """Flatten nested configuration for MLflow logging."""
        items = []
        for k, v in config.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_config(v, new_key).items())
            elif isinstance(v, list):
                # MLflow params must be strings; convert lists
                items.append((new_key, str(v)[:250]))
            elif v is None:
                items.append((new_key, "null"))
            else:
                items.append((new_key, str(v)[:250]))
        return dict(items)

    def cross_validate(
        self,
        n_folds: int = 5,
        gap: int = 0
    ) -> Dict[str, List[float]]:
        """
        Perform time series cross-validation.

        Args:
            n_folds: Number of cross-validation folds.
            gap: Gap between training and validation sets.

        Returns:
            Dictionary with cross-validation metrics.
        """
        self.logger.info(f"Starting {n_folds}-fold cross-validation...")

        # Prepare data
        train_data, val_data, test_data = self._prepare_data()
        full_data = pd.concat([train_data, val_data], ignore_index=True)

        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_folds, gap=gap, test_size=len(val_data))

        cv_metrics = {metric: [] for metric in self.config['evaluation']['metrics']}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(full_data)):
            self.logger.info(f"Cross-validation fold {fold + 1}/{n_folds}")

            try:
                # Split data
                fold_train = full_data.iloc[train_idx]
                fold_val = full_data.iloc[val_idx]

                # Build hierarchy
                hierarchy_data = self._build_hierarchy(fold_train)

                # Train model
                model = self._train_model(fold_train, hierarchy_data)

                # Validate
                fold_metrics = self._validate_model(model, fold_val, test_data)

                # Store metrics
                for metric, value in fold_metrics.items():
                    if metric in cv_metrics:
                        cv_metrics[metric].append(value)

            except Exception as e:
                self.logger.warning(f"Fold {fold + 1} failed: {e}")

        # Compute mean and std
        cv_results = {}
        for metric, values in cv_metrics.items():
            if values:
                cv_results[f"{metric}_mean"] = np.mean(values)
                cv_results[f"{metric}_std"] = np.std(values)

        self.logger.info(f"Cross-validation completed: {cv_results}")
        return cv_results

    def save_model(self, model_path: str) -> None:
        """
        Save trained model to disk.

        Args:
            model_path: Path to save the model.

        Raises:
            ValueError: If model is not trained.
        """
        if not self.is_fitted or self.best_model is None:
            raise ValueError("No trained model to save. Run train() first.")

        import pickle

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.best_model,
                    'config': self.config,
                    'preprocessor': self.preprocessor,
                    'aggregation_matrix': self.aggregation_matrix
                }, f)

            self.logger.info(f"Model saved to {model_path}")

        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise

    def load_model(self, model_path: str) -> HierarchicalEnsembleForecaster:
        """
        Load trained model from disk.

        Args:
            model_path: Path to the saved model.

        Returns:
            Loaded model.

        Raises:
            FileNotFoundError: If model file is not found.
        """
        import pickle

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)

            self.best_model = saved_data['model']
            self.config = saved_data['config']
            self.preprocessor = saved_data['preprocessor']
            self.aggregation_matrix = saved_data['aggregation_matrix']
            self.is_fitted = True

            self.logger.info(f"Model loaded from {model_path}")
            return self.best_model

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise