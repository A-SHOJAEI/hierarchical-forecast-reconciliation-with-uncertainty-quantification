"""Tests for model validation and edge cases."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import warnings

from hierarchical_forecast_reconciliation_with_uncertainty_quantification.models.model import (
    BaseForecaster, StatisticalForecaster, DeepLearningForecaster,
    HierarchicalEnsembleForecaster, ProbabilisticReconciler
)


class TestBaseForecasterValidation:
    """Test base forecaster validation and contract enforcement."""

    def test_abstract_base_class_instantiation(self):
        """Test that BaseForecaster cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseForecaster("test")

    def test_forecaster_name_validation(self):
        """Test that forecaster names are properly set."""
        # Create a concrete implementation for testing
        class TestForecaster(BaseForecaster):
            def fit(self, data, target_col="sales"):
                return self

            def predict(self, horizon, return_intervals=True, confidence_levels=None):
                return {}

        forecaster = TestForecaster("test_forecaster")
        assert forecaster.name == "test_forecaster"
        assert not forecaster.is_fitted


class TestStatisticalForecasterValidation:
    """Test statistical forecaster validation and error handling."""

    def test_invalid_method(self):
        """Test error handling for invalid statistical methods."""
        with pytest.raises(ValueError, match="Unknown method"):
            StatisticalForecaster(method="invalid_method")

    def test_fit_empty_data(self):
        """Test error handling for empty training data."""
        forecaster = StatisticalForecaster(method="ets")
        empty_data = pd.DataFrame()

        with pytest.raises(ValueError, match="Training data is empty"):
            forecaster.fit(empty_data)

    def test_fit_missing_target_column(self):
        """Test error handling for missing target column."""
        forecaster = StatisticalForecaster(method="ets")
        data = pd.DataFrame({'not_sales': [1, 2, 3]})

        with pytest.raises(ValueError, match="Target column .* not found"):
            forecaster.fit(data, target_col="sales")

    def test_fit_non_numeric_target(self):
        """Test error handling for non-numeric target data."""
        forecaster = StatisticalForecaster(method="ets")
        data = pd.DataFrame({'sales': ['a', 'b', 'c']})

        with pytest.raises(ValueError, match="Target column must contain numeric data"):
            forecaster.fit(data)

    def test_fit_insufficient_data(self):
        """Test error handling for insufficient training data."""
        forecaster = StatisticalForecaster(method="ets")
        data = pd.DataFrame({'sales': [1, 2]})  # Too few observations

        with pytest.raises(ValueError, match="Insufficient training data"):
            forecaster.fit(data)

    def test_predict_without_fitting(self):
        """Test error handling when predicting without fitting."""
        forecaster = StatisticalForecaster(method="ets")

        with pytest.raises(ValueError, match="Forecaster must be fitted"):
            forecaster.predict(horizon=10)

    def test_predict_invalid_horizon(self):
        """Test error handling for invalid prediction horizon."""
        forecaster = StatisticalForecaster(method="ets")
        data = pd.DataFrame({'sales': np.random.randn(100)})
        forecaster.fit(data)

        with pytest.raises(ValueError, match="Horizon must be a positive integer"):
            forecaster.predict(horizon=0)

        with pytest.raises(ValueError, match="Horizon must be a positive integer"):
            forecaster.predict(horizon=-5)

    def test_predict_invalid_confidence_levels(self):
        """Test error handling for invalid confidence levels."""
        forecaster = StatisticalForecaster(method="ets")
        data = pd.DataFrame({'sales': np.random.randn(100)})
        forecaster.fit(data)

        with pytest.raises(ValueError, match="Confidence levels must be between 0 and 1"):
            forecaster.predict(horizon=5, confidence_levels=[0.95, 1.5])

    def test_ets_configuration_validation(self):
        """Test ETS configuration validation."""
        invalid_config = {'trend': 'invalid_trend'}

        with pytest.raises(ValueError, match="Invalid ETS trend"):
            StatisticalForecaster(method="ets", ets_config=invalid_config)

    def test_arima_configuration_validation(self):
        """Test ARIMA configuration validation."""
        invalid_config = {'order': 'not_a_tuple'}

        with pytest.raises(ValueError, match="ARIMA order must be a tuple"):
            StatisticalForecaster(method="arima", arima_config=invalid_config)

    def test_fit_with_constant_series(self):
        """Test error handling for constant time series."""
        forecaster = StatisticalForecaster(method="ets")
        data = pd.DataFrame({'sales': [5.0] * 100})  # Constant series

        with pytest.raises(ValueError, match="Cannot fit model to constant series"):
            forecaster.fit(data)

    def test_fit_with_extreme_values(self):
        """Test handling of extreme values in data."""
        forecaster = StatisticalForecaster(method="ets")
        data = pd.DataFrame({'sales': [1, 2, 1e10, 3, 4]})  # Contains extreme value

        # Should log warning but continue
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            forecaster.fit(data)
            # Check that a warning was issued
            assert len(w) > 0


class TestDeepLearningForecasterValidation:
    """Test deep learning forecaster validation and error handling."""

    def test_invalid_architecture(self):
        """Test error handling for invalid neural network architecture."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            DeepLearningForecaster(architecture="invalid_arch")

    def test_fit_insufficient_data_for_deep_learning(self):
        """Test error handling for insufficient data for deep learning."""
        forecaster = DeepLearningForecaster(architecture="nbeats")
        data = pd.DataFrame({'sales': np.random.randn(50)})  # Too few for deep learning

        with pytest.raises(ValueError, match="Insufficient training data for deep learning"):
            forecaster.fit(data)

    def test_invalid_model_config(self):
        """Test error handling for invalid model configuration."""
        invalid_config = {
            'epochs': -1,  # Invalid epoch count
            'learning_rate': 2.0  # Invalid learning rate
        }

        with pytest.raises(ValueError, match="Invalid model configuration"):
            DeepLearningForecaster(architecture="tft", model_config=invalid_config)

    def test_gpu_availability_handling(self):
        """Test proper handling when GPU is not available."""
        forecaster = DeepLearningForecaster(architecture="nbeats")

        with patch('torch.cuda.is_available', return_value=False):
            data = pd.DataFrame({'sales': np.random.randn(1000)})
            # Should fall back to CPU without error
            forecaster.fit(data)

    def test_memory_handling_large_data(self):
        """Test memory handling with large datasets."""
        forecaster = DeepLearningForecaster(architecture="nbeats")

        # Simulate out of memory error
        with patch.object(forecaster, '_train_model', side_effect=RuntimeError("CUDA out of memory")):
            data = pd.DataFrame({'sales': np.random.randn(1000)})

            with pytest.raises(RuntimeError, match="Memory error during training"):
                forecaster.fit(data)


class TestHierarchicalEnsembleForecasterValidation:
    """Test hierarchical ensemble forecaster validation."""

    def test_invalid_model_weights(self):
        """Test error handling for invalid ensemble weights."""
        invalid_weights = {'ets': -0.5, 'arima': 1.5}  # Negative weight and sum > 1

        with pytest.raises(ValueError, match="Ensemble weights must be non-negative and sum to 1"):
            HierarchicalEnsembleForecaster(model_weights=invalid_weights)

    def test_unknown_model_in_weights(self):
        """Test error handling for unknown models in weights."""
        invalid_weights = {'unknown_model': 0.5, 'ets': 0.5}

        with pytest.raises(ValueError, match="Unknown model type"):
            HierarchicalEnsembleForecaster(model_weights=invalid_weights)

    def test_fit_with_insufficient_models(self):
        """Test error handling when too few models can be fitted."""
        forecaster = HierarchicalEnsembleForecaster()
        data = pd.DataFrame({'sales': [1, 2]})  # Insufficient for most models

        with pytest.raises(ValueError, match="Unable to fit sufficient models"):
            forecaster.fit(data)

    def test_prediction_inconsistency_handling(self):
        """Test handling of prediction inconsistencies between models."""
        forecaster = HierarchicalEnsembleForecaster()

        # Mock models with inconsistent predictions
        mock_model1 = MagicMock()
        mock_model1.predict.return_value = {'predictions': np.array([1, 2, 3])}

        mock_model2 = MagicMock()
        mock_model2.predict.return_value = {'predictions': np.array([1, 2])}  # Different length

        forecaster.fitted_models = {'ets': mock_model1, 'arima': mock_model2}
        forecaster.is_fitted = True

        with pytest.raises(ValueError, match="Inconsistent prediction dimensions"):
            forecaster.predict(horizon=3)


class TestProbabilisticReconcilerValidation:
    """Test probabilistic reconciler validation and error handling."""

    def test_invalid_reconciliation_method(self):
        """Test error handling for invalid reconciliation method."""
        with pytest.raises(ValueError, match="Unknown reconciliation method"):
            ProbabilisticReconciler(method="invalid_method")

    def test_invalid_aggregation_matrix(self):
        """Test error handling for invalid aggregation matrix."""
        reconciler = ProbabilisticReconciler()

        invalid_matrix = np.array([[1, 2], [3]])  # Irregular shape

        with pytest.raises(ValueError, match="Invalid aggregation matrix"):
            reconciler.fit(invalid_matrix)

    def test_reconcile_dimension_mismatch(self):
        """Test error handling for dimension mismatches during reconciliation."""
        reconciler = ProbabilisticReconciler()
        agg_matrix = np.array([[1, 1], [1, 0], [0, 1]])
        reconciler.fit(agg_matrix)

        # Wrong dimension forecasts
        wrong_forecasts = np.array([1, 2, 3, 4])  # Should be 2D

        with pytest.raises(ValueError, match="Forecast dimensions mismatch"):
            reconciler.reconcile(wrong_forecasts)

    def test_reconcile_without_fitting(self):
        """Test error handling when reconciling without fitting."""
        reconciler = ProbabilisticReconciler()

        forecasts = np.array([[1, 2]])

        with pytest.raises(ValueError, match="Reconciler must be fitted"):
            reconciler.reconcile(forecasts)

    def test_singular_covariance_matrix(self):
        """Test handling of singular covariance matrices."""
        reconciler = ProbabilisticReconciler()
        agg_matrix = np.array([[1, 1], [1, 1]])  # Linearly dependent rows
        reconciler.fit(agg_matrix)

        forecasts = np.array([[1, 2], [1, 2]])

        # Should handle singular matrix gracefully
        with pytest.raises(ValueError, match="Singular covariance matrix"):
            reconciler.reconcile(forecasts)

    def test_negative_forecast_handling(self):
        """Test handling of negative forecasts."""
        reconciler = ProbabilisticReconciler()
        agg_matrix = np.array([[1, 1], [1, 0], [0, 1]])
        reconciler.fit(agg_matrix)

        negative_forecasts = np.array([[-5, -10], [0, 5], [2, 3]])

        # Should apply non-negativity constraints
        result = reconciler.reconcile(negative_forecasts)
        assert np.all(result >= 0)

    def test_extreme_forecast_values(self):
        """Test handling of extreme forecast values."""
        reconciler = ProbabilisticReconciler()
        agg_matrix = np.array([[1, 1], [1, 0], [0, 1]])
        reconciler.fit(agg_matrix)

        extreme_forecasts = np.array([[1e10, 1e10], [1e5, 1e5], [1e5, 1e5]])

        # Should handle extreme values without overflow
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = reconciler.reconcile(extreme_forecasts)
            # Check that a warning was issued about extreme values
            assert any("extreme" in str(warning.message).lower() for warning in w)


class TestIntegrationValidation:
    """Test integration scenarios and edge cases."""

    def test_end_to_end_pipeline_with_edge_cases(self):
        """Test complete pipeline with various edge cases."""
        # Test with minimal but valid data
        np.random.seed(42)
        data = pd.DataFrame({
            'sales': np.abs(np.random.randn(200)) + 1,  # Positive sales
            'id': ['item_1'] * 200,
            'state_id': ['CA'] * 200,
            'store_id': ['CA_1'] * 200,
            'date': pd.date_range('2020-01-01', periods=200)
        })

        # Should handle this minimal case gracefully
        try:
            # Statistical model
            stat_model = StatisticalForecaster(method="ets")
            stat_model.fit(data)
            stat_predictions = stat_model.predict(horizon=10)

            # Ensemble model with single component
            ensemble_weights = {'ets': 1.0}
            ensemble = HierarchicalEnsembleForecaster(model_weights=ensemble_weights)
            ensemble.fit(data)
            ensemble_predictions = ensemble.predict(horizon=10)

            assert 'predictions' in stat_predictions
            assert 'predictions' in ensemble_predictions

        except Exception as e:
            pytest.fail(f"End-to-end pipeline failed with minimal data: {e}")

    def test_numerical_stability(self):
        """Test numerical stability with challenging data."""
        # Create data with numerical challenges
        np.random.seed(42)
        challenging_data = pd.DataFrame({
            'sales': [1e-10, 1e10, 0, 1e-5, 1e8] * 40  # Mix of very small and large values
        })

        forecaster = StatisticalForecaster(method="ets")

        # Should handle numerical challenges gracefully
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                forecaster.fit(challenging_data)
                predictions = forecaster.predict(horizon=5)

                # Check that predictions are reasonable
                assert np.all(np.isfinite(predictions['predictions']))
                assert np.all(predictions['predictions'] >= 0)

            except Exception as e:
                # If fitting fails, should fail gracefully with informative error
                assert "numerical" in str(e).lower() or "stability" in str(e).lower()

    def test_memory_efficiency_large_hierarchy(self):
        """Test memory efficiency with large hierarchies."""
        # Simulate large hierarchy scenario
        n_items = 1000
        n_days = 365

        large_data = pd.DataFrame({
            'sales': np.random.exponential(10, n_items * n_days),
            'item_id': np.repeat(range(n_items), n_days),
            'date': pd.tile(pd.date_range('2020-01-01', periods=n_days), n_items)
        })

        # Should handle large data efficiently
        try:
            from hierarchical_forecast_reconciliation_with_uncertainty_quantification.data.loader import HierarchicalDataBuilder

            builder = HierarchicalDataBuilder(['total', 'item'])
            hierarchy = builder.build_hierarchy(large_data)

            assert 'total' in hierarchy
            assert 'item' in hierarchy

        except MemoryError:
            pytest.skip("Insufficient memory for large hierarchy test")
        except Exception as e:
            # Should fail gracefully if system limitations are hit
            assert "memory" in str(e).lower() or "size" in str(e).lower()