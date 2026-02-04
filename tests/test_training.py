"""Tests for training pipeline."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from hierarchical_forecast_reconciliation_with_uncertainty_quantification.training.trainer import (
    HierarchicalForecastTrainer
)
from hierarchical_forecast_reconciliation_with_uncertainty_quantification.evaluation.metrics import (
    HierarchicalMetrics
)


class TestHierarchicalForecastTrainer:
    """Test hierarchical forecast trainer functionality."""

    def test_init(self, sample_config, sample_m5_files):
        """Test trainer initialization."""
        trainer = HierarchicalForecastTrainer(
            config=sample_config,
            data_path=str(sample_m5_files),
            experiment_name="test_experiment"
        )

        assert trainer.config == sample_config
        assert trainer.data_path == str(sample_m5_files)
        assert trainer.experiment_name == "test_experiment"
        assert not trainer.is_fitted

    def test_init_with_config_path(self, sample_config):
        """Test initialization with default paths from config."""
        config_with_path = sample_config.copy()
        config_with_path['data']['path'] = '/test/path'

        trainer = HierarchicalForecastTrainer(config=config_with_path)
        assert trainer.data_path == '/test/path'

    @patch('hierarchical_forecast_reconciliation_with_uncertainty_quantification.training.trainer.mlflow')
    def test_mlflow_setup(self, mock_mlflow, sample_config, sample_m5_files):
        """Test MLflow setup."""
        trainer = HierarchicalForecastTrainer(
            config=sample_config,
            data_path=str(sample_m5_files)
        )

        # MLflow should be configured
        mock_mlflow.set_tracking_uri.assert_called()
        mock_mlflow.set_experiment.assert_called()

    def test_prepare_data(self, sample_config, sample_m5_files):
        """Test data preparation pipeline."""
        trainer = HierarchicalForecastTrainer(
            config=sample_config,
            data_path=str(sample_m5_files)
        )

        # Mock the data loader to avoid actual file operations
        with patch.object(trainer.data_loader, 'load_data') as mock_load, \
             patch.object(trainer.data_loader, 'prepare_time_series_data') as mock_prepare, \
             patch.object(trainer.data_loader, 'add_calendar_features') as mock_calendar, \
             patch.object(trainer.preprocessor, 'fit_transform') as mock_preprocess, \
             patch.object(trainer.preprocessor, 'create_train_test_split') as mock_split:

            # Setup mock returns
            mock_load.return_value = (Mock(), Mock(), Mock())
            mock_prepare.return_value = pd.DataFrame({
                'id': ['series_1'] * 10,
                'date': pd.date_range('2022-01-01', periods=10),
                'sales': np.random.uniform(1, 100, 10)
            })
            mock_calendar.return_value = mock_prepare.return_value
            mock_preprocess.return_value = mock_prepare.return_value
            mock_split.return_value = (
                mock_prepare.return_value.head(5),  # train
                mock_prepare.return_value.iloc[5:8],  # val
                mock_prepare.return_value.tail(2)   # test
            )

            train_data, val_data, test_data = trainer._prepare_data()

            assert isinstance(train_data, pd.DataFrame)
            assert isinstance(val_data, pd.DataFrame)
            assert isinstance(test_data, pd.DataFrame)

            # Check that all steps were called
            mock_load.assert_called_once()
            mock_prepare.assert_called_once()
            mock_calendar.assert_called_once()
            mock_preprocess.assert_called_once()
            mock_split.assert_called_once()

    def test_build_hierarchy(self, sample_config, sample_time_series_data):
        """Test hierarchy building."""
        trainer = HierarchicalForecastTrainer(config=sample_config)

        hierarchy_data = trainer._build_hierarchy(sample_time_series_data)

        assert isinstance(hierarchy_data, dict)
        assert len(hierarchy_data) > 0
        assert trainer.aggregation_matrix is not None

    def test_train_model_mock(self, sample_config, sample_time_series_data, sample_aggregation_matrix):
        """Test model training with mocks."""
        trainer = HierarchicalForecastTrainer(config=sample_config)
        trainer.aggregation_matrix = sample_aggregation_matrix

        with patch('hierarchical_forecast_reconciliation_with_uncertainty_quantification.training.trainer.HierarchicalEnsembleForecaster') as mock_ensemble:
            # Setup mock ensemble
            mock_ensemble_instance = Mock()
            mock_ensemble_instance.fit.return_value = mock_ensemble_instance
            mock_ensemble.return_value = mock_ensemble_instance

            model = trainer._train_model(sample_time_series_data, {})

            assert model == mock_ensemble_instance
            mock_ensemble_instance.fit.assert_called_once()

    def test_validate_model(self, sample_config, sample_time_series_data):
        """Test model validation."""
        trainer = HierarchicalForecastTrainer(config=sample_config)

        # Mock model with predictions
        mock_model = Mock()
        mock_predictions = {
            "forecasts": {
                "series_1": np.random.uniform(50, 150, 28),
                "series_2": np.random.uniform(50, 150, 28)
            },
            "lower_90": {
                "series_1": np.random.uniform(40, 140, 28),
                "series_2": np.random.uniform(40, 140, 28)
            },
            "upper_90": {
                "series_1": np.random.uniform(60, 160, 28),
                "series_2": np.random.uniform(60, 160, 28)
            }
        }
        mock_model.predict.return_value = mock_predictions

        val_data = sample_time_series_data.head(50)
        test_data = sample_time_series_data.tail(50)

        with patch.object(trainer, '_extract_actuals') as mock_extract:
            mock_extract.return_value = {
                "series_1": np.random.uniform(45, 155, 28),
                "series_2": np.random.uniform(45, 155, 28)
            }

            metrics = trainer._validate_model(mock_model, val_data, test_data)

            assert isinstance(metrics, dict)
            assert len(metrics) > 0
            mock_model.predict.assert_called_once()

    def test_extract_actuals(self, sample_config, sample_time_series_data):
        """Test actual values extraction."""
        trainer = HierarchicalForecastTrainer(config=sample_config)

        actuals = trainer._extract_actuals(sample_time_series_data)

        assert isinstance(actuals, dict)
        assert len(actuals) > 0

        # Check that each series has actual values
        for series_id, values in actuals.items():
            assert isinstance(values, np.ndarray)
            assert len(values) > 0

    def test_sample_hyperparameters(self, sample_config):
        """Test hyperparameter sampling."""
        # Add optimization config
        config_with_opt = sample_config.copy()
        config_with_opt['optimization'] = {
            'search_space': {
                'tft_learning_rate': [0.001, 0.1],
                'tft_hidden_size': [8, 64],
                'nbeats_num_stacks': [10, 50],
                'reconciliation_lambda_reg': [0.001, 0.1]
            }
        }

        trainer = HierarchicalForecastTrainer(config=config_with_opt)

        # Mock optuna trial
        mock_trial = Mock()
        mock_trial.suggest_float.return_value = 0.05
        mock_trial.suggest_int.return_value = 32

        params = trainer._sample_hyperparameters(mock_trial)

        assert isinstance(params, dict)
        assert len(params) > 0

        # Check that suggest methods were called
        assert mock_trial.suggest_float.called
        assert mock_trial.suggest_int.called

    def test_update_config_with_trial_params(self, sample_config):
        """Test configuration update with trial parameters."""
        trainer = HierarchicalForecastTrainer(config=sample_config)

        trial_params = {
            'tft_learning_rate': 0.05,
            'tft_hidden_size': 32,
            'nbeats_num_stacks': 25,
            'reconciliation_lambda_reg': 0.02,
            'ensemble_tft_weight': 0.4
        }

        updated_config = trainer._update_config_with_trial_params(trial_params)

        # Check that parameters were updated
        assert updated_config['models']['deep_learning']['tft']['learning_rate'] == 0.05
        assert updated_config['models']['deep_learning']['tft']['hidden_size'] == 32
        assert updated_config['models']['deep_learning']['nbeats']['num_stacks'] == 25
        assert updated_config['reconciliation']['lambda_reg'] == 0.02
        assert updated_config['ensemble']['weights']['tft'] == 0.4

    @patch('hierarchical_forecast_reconciliation_with_uncertainty_quantification.training.trainer.mlflow')
    def test_train_full_pipeline_mock(self, mock_mlflow, sample_config, sample_m5_files):
        """Test full training pipeline with extensive mocking."""
        # Setup MLflow mocks
        mock_mlflow.start_run.return_value.__enter__ = Mock(return_value=Mock())
        mock_mlflow.start_run.return_value.__exit__ = Mock(return_value=None)

        trainer = HierarchicalForecastTrainer(
            config=sample_config,
            data_path=str(sample_m5_files)
        )

        # Mock all major components
        with patch.object(trainer, '_prepare_data') as mock_prepare, \
             patch.object(trainer, '_build_hierarchy') as mock_hierarchy, \
             patch.object(trainer, '_train_model') as mock_train, \
             patch.object(trainer, '_validate_model') as mock_validate, \
             patch.object(trainer, '_log_training_results') as mock_log:

            # Setup mock returns
            mock_data = pd.DataFrame({
                'id': ['series_1'] * 10,
                'date': pd.date_range('2022-01-01', periods=10),
                'sales': np.random.uniform(1, 100, 10)
            })

            mock_prepare.return_value = (mock_data, mock_data, mock_data)
            mock_hierarchy.return_value = {'item_store': mock_data}

            mock_model = Mock()
            mock_train.return_value = mock_model

            mock_metrics = {'WRMSSE': 0.5, 'MASE': 1.2}
            mock_validate.return_value = mock_metrics

            # Run training
            result_model = trainer.train(optimize_hyperparameters=False)

            assert result_model == mock_model
            assert trainer.is_fitted
            assert trainer.best_model == mock_model

            # Check that all steps were called
            mock_prepare.assert_called_once()
            mock_hierarchy.assert_called_once()
            mock_train.assert_called_once()
            mock_validate.assert_called_once()
            mock_log.assert_called_once()

    def test_save_and_load_model(self, sample_config, temp_directory):
        """Test model saving and loading."""
        trainer = HierarchicalForecastTrainer(config=sample_config)

        # Create mock fitted model
        mock_model = Mock()
        trainer.best_model = mock_model
        trainer.is_fitted = True
        trainer.aggregation_matrix = np.eye(3)

        # Test save
        model_path = temp_directory / "test_model.pkl"
        trainer.save_model(str(model_path))

        assert model_path.exists()

        # Test load
        trainer2 = HierarchicalForecastTrainer(config=sample_config)
        loaded_model = trainer2.load_model(str(model_path))

        assert loaded_model is not None
        assert trainer2.is_fitted
        assert trainer2.config == sample_config

    def test_save_model_not_fitted(self, sample_config):
        """Test saving model when not fitted."""
        trainer = HierarchicalForecastTrainer(config=sample_config)

        with pytest.raises(ValueError, match="No trained model to save"):
            trainer.save_model("test_model.pkl")

    def test_load_model_not_found(self, sample_config):
        """Test loading non-existent model."""
        trainer = HierarchicalForecastTrainer(config=sample_config)

        with pytest.raises(FileNotFoundError):
            trainer.load_model("non_existent_model.pkl")

    def test_flatten_config(self, sample_config):
        """Test configuration flattening for MLflow."""
        trainer = HierarchicalForecastTrainer(config=sample_config)

        flattened = trainer._flatten_config(sample_config)

        assert isinstance(flattened, dict)
        assert len(flattened) > len(sample_config)  # Should have more keys due to flattening

        # Check that nested keys are flattened
        assert any("models.statistical" in key for key in flattened.keys())
        assert any("training." in key for key in flattened.keys())

    def test_cross_validate_mock(self, sample_config, sample_m5_files):
        """Test cross-validation with mocks."""
        trainer = HierarchicalForecastTrainer(
            config=sample_config,
            data_path=str(sample_m5_files)
        )

        with patch.object(trainer, '_prepare_data') as mock_prepare, \
             patch.object(trainer, '_build_hierarchy') as mock_hierarchy, \
             patch.object(trainer, '_train_model') as mock_train, \
             patch.object(trainer, '_validate_model') as mock_validate:

            # Setup mocks
            mock_data = pd.DataFrame({
                'id': ['series_1'] * 100,
                'date': pd.date_range('2022-01-01', periods=100),
                'sales': np.random.uniform(1, 100, 100)
            })

            mock_prepare.return_value = (mock_data, mock_data.tail(20), mock_data.tail(10))
            mock_hierarchy.return_value = {'item_store': mock_data}
            mock_train.return_value = Mock()
            mock_validate.return_value = {'WRMSSE': 0.5, 'MASE': 1.2}

            # Run cross-validation
            cv_results = trainer.cross_validate(n_folds=3)

            assert isinstance(cv_results, dict)
            # Should have mean and std for each metric
            assert any("_mean" in key for key in cv_results.keys())
            assert any("_std" in key for key in cv_results.keys())


class TestHierarchicalMetricsIntegration:
    """Test metrics integration with training pipeline."""

    def test_metrics_calculator_integration(self, sample_predictions, sample_actuals):
        """Test that metrics calculator works with trainer outputs."""
        metrics_calc = HierarchicalMetrics()

        # Compute metrics that trainer would compute
        metrics = metrics_calc.compute_all_metrics(
            predictions=sample_predictions,
            actuals=sample_actuals,
            confidence_levels=[0.1, 0.05]
        )

        assert isinstance(metrics, dict)
        assert "WRMSSE" in metrics
        assert "MASE" in metrics
        assert "sMAPE" in metrics

        # All metrics should be numeric
        for metric_name, value in metrics.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)

    def test_performance_report_generation(self, sample_predictions, sample_actuals):
        """Test performance report generation."""
        metrics_calc = HierarchicalMetrics()

        metrics = metrics_calc.compute_all_metrics(
            predictions=sample_predictions,
            actuals=sample_actuals
        )

        target_metrics = {
            "WRMSSE": 0.52,
            "MASE": 1.5,
            "sMAPE": 15.0
        }

        report = metrics_calc.create_performance_report(metrics, target_metrics)

        assert isinstance(report, str)
        assert len(report) > 0
        assert "Performance Report" in report
        assert "WRMSSE" in report