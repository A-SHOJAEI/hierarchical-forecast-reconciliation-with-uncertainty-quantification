"""Tests for model implementations."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from hierarchical_forecast_reconciliation_with_uncertainty_quantification.models.model import (
    StatisticalForecaster,
    DeepLearningForecaster,
    ProbabilisticReconciler,
    HierarchicalEnsembleForecaster
)


class TestStatisticalForecaster:
    """Test statistical forecasting models."""

    def test_init_ets(self):
        """Test ETS model initialization."""
        forecaster = StatisticalForecaster(
            method="ets",
            ets_config={"seasonal_periods": 7}
        )

        assert forecaster.method == "ets"
        assert forecaster.ets_config["seasonal_periods"] == 7
        assert not forecaster.is_fitted

    def test_init_arima(self):
        """Test ARIMA model initialization."""
        forecaster = StatisticalForecaster(
            method="arima",
            arima_config={"seasonal_order": [1, 1, 1, 7]}
        )

        assert forecaster.method == "arima"
        assert forecaster.arima_config["seasonal_order"] == [1, 1, 1, 7]

    def test_init_invalid_method(self):
        """Test initialization with invalid method."""
        with pytest.raises(ValueError, match="Unknown statistical method"):
            StatisticalForecaster(method="invalid_method")

    def test_fit_ets_simple_data(self, sample_time_series_data):
        """Test fitting ETS model with simple data."""
        forecaster = StatisticalForecaster(method="ets")

        # Use a subset of data for faster testing
        simple_data = sample_time_series_data.head(100)
        forecaster.fit(simple_data)

        assert forecaster.is_fitted
        assert len(forecaster.fitted_models) > 0

    def test_predict_without_fit(self):
        """Test prediction without fitting."""
        forecaster = StatisticalForecaster(method="ets")

        with pytest.raises(ValueError, match="Model not fitted"):
            forecaster.predict(horizon=7)

    def test_predict_with_intervals(self, sample_time_series_data):
        """Test prediction with confidence intervals."""
        forecaster = StatisticalForecaster(method="ets")

        # Fit with simple data
        simple_data = sample_time_series_data.head(50)
        forecaster.fit(simple_data)

        # Generate predictions
        predictions = forecaster.predict(
            horizon=7,
            return_intervals=True,
            confidence_levels=[0.1, 0.05]
        )

        assert "forecasts" in predictions
        assert "lower_90" in predictions
        assert "upper_90" in predictions
        assert "lower_95" in predictions
        assert "upper_95" in predictions

        # Check that all series have predictions
        for series_id in forecaster.fitted_models.keys():
            assert series_id in predictions["forecasts"]

    def test_predict_without_intervals(self, sample_time_series_data):
        """Test prediction without intervals."""
        forecaster = StatisticalForecaster(method="ets")

        # Fit with simple data
        simple_data = sample_time_series_data.head(50)
        forecaster.fit(simple_data)

        # Generate predictions without intervals
        predictions = forecaster.predict(
            horizon=7,
            return_intervals=False
        )

        assert "forecasts" in predictions
        assert "lower_90" not in predictions
        assert "upper_90" not in predictions


class TestDeepLearningForecaster:
    """Test deep learning forecasting models."""

    def test_init_tft(self):
        """Test TFT model initialization."""
        forecaster = DeepLearningForecaster(
            model_type="tft",
            tft_config={"max_epochs": 5, "hidden_size": 16}
        )

        assert forecaster.model_type == "tft"
        assert forecaster.tft_config["max_epochs"] == 5
        assert not forecaster.is_fitted

    def test_init_nbeats(self):
        """Test N-BEATS model initialization."""
        forecaster = DeepLearningForecaster(
            model_type="nbeats",
            nbeats_config={"num_stacks": 10}
        )

        assert forecaster.model_type == "nbeats"
        assert forecaster.nbeats_config["num_stacks"] == 10

    def test_init_invalid_model(self):
        """Test initialization with invalid model type."""
        with pytest.raises(ValueError, match="Unknown deep learning model"):
            DeepLearningForecaster(model_type="invalid_model")

    @pytest.mark.slow
    def test_fit_tft_mock(self, sample_time_series_data):
        """Test fitting TFT model with mocked dependencies."""
        with patch('hierarchical_forecast_reconciliation_with_uncertainty_quantification.models.model.TFTModel') as mock_tft:
            # Mock TFT model
            mock_model_instance = Mock()
            mock_tft.return_value = mock_model_instance

            forecaster = DeepLearningForecaster(
                model_type="tft",
                tft_config={"max_epochs": 2, "hidden_size": 8}
            )

            # Use small subset for testing
            simple_data = sample_time_series_data.head(30)
            forecaster.fit(simple_data)

            assert forecaster.is_fitted
            assert mock_model_instance.fit.called

    def test_predict_without_fit(self):
        """Test prediction without fitting."""
        forecaster = DeepLearningForecaster(model_type="tft")

        with pytest.raises(ValueError, match="Model not fitted"):
            forecaster.predict(horizon=7)

    def test_prepare_darts_data(self, sample_time_series_data):
        """Test data preparation for Darts format."""
        forecaster = DeepLearningForecaster(model_type="tft")

        # Use small subset
        simple_data = sample_time_series_data.head(30)

        try:
            time_series_list = forecaster._prepare_darts_data(simple_data, "sales")
            assert isinstance(time_series_list, list)
            assert len(time_series_list) > 0
        except ImportError:
            # Skip if Darts is not available
            pytest.skip("Darts not available for testing")


class TestProbabilisticReconciler:
    """Test probabilistic reconciliation functionality."""

    def test_init(self):
        """Test reconciler initialization."""
        reconciler = ProbabilisticReconciler(
            method="probabilistic_mint",
            weights="wls",
            lambda_reg=0.01
        )

        assert reconciler.method == "probabilistic_mint"
        assert reconciler.weights == "wls"
        assert reconciler.lambda_reg == 0.01
        assert reconciler.preserve_uncertainty is True

    def test_fit_ols_weights(self, sample_aggregation_matrix):
        """Test fitting with OLS weights."""
        reconciler = ProbabilisticReconciler(weights="ols")
        reconciler.fit(sample_aggregation_matrix)

        assert reconciler.S is not None
        assert reconciler.W is not None
        assert reconciler.G is not None

        # Check weight matrix is identity for OLS
        assert np.allclose(reconciler.W, np.eye(reconciler.S.shape[0]))

    def test_fit_with_residuals(self, sample_aggregation_matrix):
        """Test fitting with residual-based weights."""
        # Create mock residuals
        residuals = {}
        n_series = sample_aggregation_matrix.shape[0]
        for i in range(n_series):
            residuals[f"series_{i}"] = np.random.normal(0, 1, 100)

        reconciler = ProbabilisticReconciler(weights="wls")
        reconciler.fit(sample_aggregation_matrix, residuals)

        assert reconciler.S is not None
        assert reconciler.W is not None
        assert reconciler.G is not None

    def test_reconcile_forecasts(self, sample_aggregation_matrix, sample_predictions):
        """Test forecast reconciliation."""
        reconciler = ProbabilisticReconciler(weights="ols")
        reconciler.fit(sample_aggregation_matrix)

        # Create compatible predictions
        n_series = sample_aggregation_matrix.shape[0]
        compatible_predictions = {}
        for i in range(n_series):
            compatible_predictions[f"series_{i}"] = np.random.uniform(50, 150, 28)

        reconciled_forecasts, reconciled_intervals = reconciler.reconcile(
            compatible_predictions
        )

        assert isinstance(reconciled_forecasts, dict)
        assert len(reconciled_forecasts) == len(compatible_predictions)

        # Check that reconciled values are different from original
        for series_id in compatible_predictions:
            if series_id in reconciled_forecasts:
                original = compatible_predictions[series_id]
                reconciled = reconciled_forecasts[series_id]
                assert len(original) == len(reconciled)

    def test_reconcile_with_intervals(self, sample_aggregation_matrix, sample_predictions, sample_intervals):
        """Test reconciliation with prediction intervals."""
        reconciler = ProbabilisticReconciler(
            weights="ols",
            preserve_uncertainty=True
        )
        reconciler.fit(sample_aggregation_matrix)

        # Create compatible data
        n_series = sample_aggregation_matrix.shape[0]
        compatible_predictions = {}
        compatible_intervals = {"lower_90": {}, "upper_90": {}}

        for i in range(n_series):
            series_id = f"series_{i}"
            compatible_predictions[series_id] = np.random.uniform(50, 150, 28)
            compatible_intervals["lower_90"][series_id] = np.random.uniform(40, 140, 28)
            compatible_intervals["upper_90"][series_id] = np.random.uniform(60, 160, 28)

        reconciled_forecasts, reconciled_intervals = reconciler.reconcile(
            compatible_predictions,
            compatible_intervals
        )

        assert reconciled_intervals is not None
        assert "lower_90" in reconciled_intervals
        assert "upper_90" in reconciled_intervals

    def test_reconcile_without_fit(self, sample_predictions):
        """Test reconciliation without fitting."""
        reconciler = ProbabilisticReconciler()

        with pytest.raises(ValueError, match="Reconciler not fitted"):
            reconciler.reconcile(sample_predictions)

    def test_compute_coherence_score(self, sample_aggregation_matrix):
        """Test coherence score computation."""
        reconciler = ProbabilisticReconciler()
        reconciler.S = sample_aggregation_matrix

        # Create test predictions
        n_series = sample_aggregation_matrix.shape[0]
        test_predictions = {}
        for i in range(n_series):
            test_predictions[f"series_{i}"] = np.random.uniform(50, 150, 28)

        coherence_score = reconciler.compute_coherence_score(test_predictions)

        assert isinstance(coherence_score, float)
        assert 0.0 <= coherence_score <= 1.0


class TestHierarchicalEnsembleForecaster:
    """Test ensemble forecasting functionality."""

    def test_init(self, sample_config):
        """Test ensemble forecaster initialization."""
        ensemble = HierarchicalEnsembleForecaster(
            statistical_configs=sample_config["models"]["statistical"],
            deep_learning_configs=sample_config["models"]["deep_learning"],
            ensemble_weights=sample_config["ensemble"]["weights"],
            reconciler_config=sample_config["reconciliation"]
        )

        assert ensemble.statistical_configs is not None
        assert ensemble.deep_learning_configs is not None
        assert ensemble.ensemble_weights is not None
        assert not ensemble.is_fitted

    def test_fit_mock_models(self, sample_config, sample_time_series_data, sample_aggregation_matrix):
        """Test ensemble fitting with mocked individual models."""
        # Mock the individual forecaster classes
        with patch('hierarchical_forecast_reconciliation_with_uncertainty_quantification.models.model.StatisticalForecaster') as mock_stat, \
             patch('hierarchical_forecast_reconciliation_with_uncertainty_quantification.models.model.DeepLearningForecaster') as mock_dl:

            # Setup mock instances
            mock_stat_instance = Mock()
            mock_stat_instance.fit.return_value = mock_stat_instance
            mock_stat.return_value = mock_stat_instance

            mock_dl_instance = Mock()
            mock_dl_instance.fit.return_value = mock_dl_instance
            mock_dl.return_value = mock_dl_instance

            ensemble = HierarchicalEnsembleForecaster(
                statistical_configs=sample_config["models"]["statistical"],
                deep_learning_configs=sample_config["models"]["deep_learning"],
                ensemble_weights=sample_config["ensemble"]["weights"],
                reconciler_config=sample_config["reconciliation"]
            )

            # Use small dataset for testing
            simple_data = sample_time_series_data.head(50)

            ensemble.fit(simple_data, sample_aggregation_matrix)

            assert ensemble.is_fitted
            assert mock_stat_instance.fit.called
            assert mock_dl_instance.fit.called

    def test_predict_without_fit(self, sample_config):
        """Test prediction without fitting."""
        ensemble = HierarchicalEnsembleForecaster(
            statistical_configs=sample_config["models"]["statistical"],
            deep_learning_configs=sample_config["models"]["deep_learning"],
            ensemble_weights=sample_config["ensemble"]["weights"],
            reconciler_config=sample_config["reconciliation"]
        )

        with pytest.raises(ValueError, match="Ensemble not fitted"):
            ensemble.predict(horizon=7)

    def test_combine_predictions(self, sample_config):
        """Test prediction combination logic."""
        ensemble = HierarchicalEnsembleForecaster(
            statistical_configs=sample_config["models"]["statistical"],
            deep_learning_configs=sample_config["models"]["deep_learning"],
            ensemble_weights=sample_config["ensemble"]["weights"],
            reconciler_config=sample_config["reconciliation"]
        )

        # Create mock predictions from different models
        all_predictions = {
            "ets": {
                "forecasts": {
                    "series_1": np.array([10, 11, 12]),
                    "series_2": np.array([20, 21, 22])
                }
            },
            "arima": {
                "forecasts": {
                    "series_1": np.array([9, 10, 11]),
                    "series_2": np.array([19, 20, 21])
                }
            }
        }

        confidence_levels = [0.1]

        ensemble_forecasts, ensemble_intervals = ensemble._combine_predictions(
            all_predictions, confidence_levels, return_intervals=False
        )

        assert isinstance(ensemble_forecasts, dict)
        assert "series_1" in ensemble_forecasts
        assert "series_2" in ensemble_forecasts

        # Check that combined forecasts are weighted averages
        expected_series_1 = (
            sample_config["ensemble"]["weights"]["ets"] * np.array([10, 11, 12]) +
            sample_config["ensemble"]["weights"]["arima"] * np.array([9, 10, 11])
        ) / (sample_config["ensemble"]["weights"]["ets"] + sample_config["ensemble"]["weights"]["arima"])

        np.testing.assert_allclose(
            ensemble_forecasts["series_1"],
            expected_series_1,
            rtol=1e-10
        )

    def test_combine_predictions_with_intervals(self, sample_config):
        """Test combining predictions with intervals."""
        ensemble = HierarchicalEnsembleForecaster(
            statistical_configs=sample_config["models"]["statistical"],
            deep_learning_configs=sample_config["models"]["deep_learning"],
            ensemble_weights=sample_config["ensemble"]["weights"],
            reconciler_config=sample_config["reconciliation"]
        )

        # Create mock predictions with intervals
        all_predictions = {
            "ets": {
                "forecasts": {"series_1": np.array([10, 11, 12])},
                "lower_90": {"series_1": np.array([8, 9, 10])},
                "upper_90": {"series_1": np.array([12, 13, 14])}
            },
            "arima": {
                "forecasts": {"series_1": np.array([9, 10, 11])},
                "lower_90": {"series_1": np.array([7, 8, 9])},
                "upper_90": {"series_1": np.array([11, 12, 13])}
            }
        }

        confidence_levels = [0.1]

        ensemble_forecasts, ensemble_intervals = ensemble._combine_predictions(
            all_predictions, confidence_levels, return_intervals=True
        )

        assert ensemble_intervals is not None
        assert "lower_90" in ensemble_intervals
        assert "upper_90" in ensemble_intervals
        assert "series_1" in ensemble_intervals["lower_90"]