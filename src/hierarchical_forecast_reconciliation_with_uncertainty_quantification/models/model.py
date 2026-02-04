"""
Hierarchical forecasting models with uncertainty quantification.

This module implements ensemble forecasters combining statistical and deep learning
approaches with probabilistic hierarchical reconciliation.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from darts import TimeSeries
from darts.models import ARIMA, ExponentialSmoothing, NBEATSModel, TFTModel
from properscoring import crps_ensemble
from scipy import sparse
from scipy.linalg import solve
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from statsmodels.tsa.exponential_smoothing import ExponentialSmoothing as StatsETS
from statsmodels.tsa.arima.model import ARIMA as StatsARIMA

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class BaseForecaster(ABC):
    """
    Abstract base class for all hierarchical forecasters.

    This class defines the common interface for all forecasting models in the
    hierarchical forecasting framework. All concrete forecasters must implement
    the fit and predict methods.

    Attributes:
        name (str): Unique identifier for the forecaster instance.
        logger (logging.Logger): Logger instance for the forecaster.
        is_fitted (bool): Flag indicating whether the forecaster has been trained.

    Note:
        This is an abstract class and cannot be instantiated directly. Use one of
        the concrete implementations: StatisticalForecaster, DeepLearningForecaster,
        HierarchicalEnsembleForecaster.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize base forecaster with logging and state tracking.

        Args:
            name (str): Unique name identifier for this forecaster instance.
                Used for logging and debugging purposes. Should be descriptive
                and unique within the ensemble.

        Example:
            >>> # This will raise TypeError since BaseForecaster is abstract
            >>> forecaster = BaseForecaster("test")  # TypeError
            >>> # Use concrete implementation instead
            >>> forecaster = StatisticalForecaster("ets_model")
        """
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        data: pd.DataFrame,
        target_col: str = "sales"
    ) -> "BaseForecaster":
        """
        Fit the forecaster to training data.

        This method trains the forecasting model on the provided hierarchical
        time series data. The implementation varies by forecaster type but
        should handle multiple time series and their hierarchical relationships.

        Args:
            data (pd.DataFrame): Training data containing hierarchical time series.
                Must include the target column and appropriate grouping columns
                (e.g., 'id', 'date', 'state_id', 'store_id'). The DataFrame should
                be in long format with one row per time point per series.
            target_col (str, optional): Name of the target variable column in the
                DataFrame. Defaults to "sales". This column should contain numeric
                values representing the quantity to forecast.

        Returns:
            BaseForecaster: Self instance for method chaining, allowing calls like
                `forecaster.fit(data).predict(horizon)`.

        Raises:
            ValueError: If data is empty, target column is missing, or data format
                is invalid for the specific forecaster implementation.
            RuntimeError: If model training fails due to computational issues.

        Example:
            >>> forecaster = StatisticalForecaster("ets")
            >>> trained_forecaster = forecaster.fit(training_data, target_col="sales")
            >>> isinstance(trained_forecaster, BaseForecaster)
            True

        Note:
            After successful fitting, the `is_fitted` attribute will be set to True.
            The fitted model can then be used for prediction via the `predict` method.
        """
        pass

    @abstractmethod
    def predict(
        self,
        horizon: int,
        return_intervals: bool = True,
        confidence_levels: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate probabilistic forecasts with uncertainty quantification.

        This method produces point forecasts and prediction intervals for all
        time series in the hierarchy. The output includes both point predictions
        and probabilistic intervals at specified confidence levels.

        Args:
            horizon (int): Number of time periods to forecast into the future.
                Must be a positive integer. Typical values range from 1 to 28
                for the M5 competition (daily to 4-week ahead forecasts).
            return_intervals (bool, optional): Whether to compute and return
                prediction intervals alongside point forecasts. Defaults to True.
                Setting to False may improve performance for point forecasts only.
            confidence_levels (Optional[List[float]]): List of confidence levels
                for prediction intervals, each between 0 and 1. Defaults to None,
                which typically uses [0.8, 0.9, 0.95]. Common values include
                0.5 (median), 0.8 (80% interval), 0.9 (90% interval), 0.95 (95% interval).

        Returns:
            Dict[str, np.ndarray]: Dictionary containing forecasting results with keys:
                - 'predictions' (np.ndarray): Point forecasts with shape (n_series, horizon).
                    Each row corresponds to one time series, each column to one forecast period.
                - 'lower_bounds' (np.ndarray, optional): Lower bounds of prediction intervals
                    with shape (n_series, horizon, n_confidence_levels) if return_intervals=True.
                - 'upper_bounds' (np.ndarray, optional): Upper bounds of prediction intervals
                    with shape (n_series, horizon, n_confidence_levels) if return_intervals=True.
                - 'quantiles' (np.ndarray, optional): Full quantile forecasts if supported
                    by the underlying model, with shape (n_series, horizon, n_quantiles).

        Raises:
            ValueError: If forecaster hasn't been fitted, horizon is invalid,
                or confidence levels are outside (0, 1).
            RuntimeError: If prediction generation fails due to model issues.

        Example:
            >>> forecaster = StatisticalForecaster("ets")
            >>> forecaster.fit(training_data)
            >>> predictions = forecaster.predict(horizon=14, confidence_levels=[0.8, 0.95])
            >>> predictions['predictions'].shape
            (1000, 14)  # 1000 series, 14-day horizon
            >>> predictions['lower_bounds'].shape
            (1000, 14, 2)  # 80% and 95% confidence intervals

        Note:
            The forecaster must be fitted before calling this method. Point forecasts
            represent the expected value (mean) of the predictive distribution.
            Prediction intervals provide uncertainty quantification around these point estimates.
        """
        pass


class StatisticalForecaster(BaseForecaster):
    """
    Statistical forecasting methods (ETS, ARIMA) with uncertainty quantification.
    """

    def __init__(
        self,
        method: str = "ets",
        ets_config: Optional[Dict[str, Any]] = None,
        arima_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize statistical forecaster.

        Args:
            method: Statistical method to use ("ets" or "arima").
            ets_config: Configuration for ETS model.
            arima_config: Configuration for ARIMA model.

        Raises:
            ValueError: If invalid method is specified.
        """
        super().__init__(f"Statistical_{method.upper()}")

        if method not in ["ets", "arima"]:
            raise ValueError(f"Unknown statistical method: {method}")

        self.method = method
        self.ets_config = ets_config or {}
        self.arima_config = arima_config or {}

        # Fitted models for each time series
        self.fitted_models: Dict[str, Any] = {}
        self.series_data: Dict[str, pd.Series] = {}

    def fit(
        self,
        data: pd.DataFrame,
        target_col: str = "sales"
    ) -> "StatisticalForecaster":
        """
        Fit statistical models to each time series.

        Args:
            data: Training data with hierarchical time series.
            target_col: Target variable column name.

        Returns:
            Self for method chaining.
        """
        self.logger.info(f"Fitting {self.method} models...")

        # Group data by series ID
        grouped = data.groupby('id')

        for series_id, series_data in grouped:
            try:
                # Prepare time series
                ts_data = series_data.set_index('date')[target_col].sort_index()
                self.series_data[series_id] = ts_data

                # Fit model
                if self.method == "ets":
                    model = self._fit_ets_model(ts_data, series_id)
                elif self.method == "arima":
                    model = self._fit_arima_model(ts_data, series_id)

                self.fitted_models[series_id] = model

            except Exception as e:
                self.logger.warning(f"Failed to fit {self.method} for {series_id}: {e}")
                # Store None for failed fits
                self.fitted_models[series_id] = None

        self.is_fitted = True
        fitted_count = sum(1 for model in self.fitted_models.values() if model is not None)
        self.logger.info(f"Successfully fitted {fitted_count}/{len(grouped)} series")

        return self

    def predict(
        self,
        horizon: int,
        return_intervals: bool = True,
        confidence_levels: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate forecasts with prediction intervals.

        Args:
            horizon: Forecast horizon.
            return_intervals: Whether to return prediction intervals.
            confidence_levels: Confidence levels for intervals [0.1, 0.05].

        Returns:
            Dictionary with forecasts and prediction intervals.

        Raises:
            ValueError: If not fitted or no successful fits.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if confidence_levels is None:
            confidence_levels = [0.1, 0.05]  # 90% and 95% intervals

        self.logger.info(f"Generating {horizon}-step forecasts...")

        forecasts = {}
        intervals = {f"lower_{int((1-alpha)*100)}": {} for alpha in confidence_levels}
        intervals.update({f"upper_{int((1-alpha)*100)}": {} for alpha in confidence_levels})

        for series_id, model in self.fitted_models.items():
            if model is None:
                # Use naive forecast for failed fits
                last_value = self.series_data[series_id].iloc[-1]
                forecasts[series_id] = np.full(horizon, last_value)

                # Conservative intervals for naive forecast
                std_estimate = self.series_data[series_id].std()
                for alpha in confidence_levels:
                    z_score = norm.ppf(1 - alpha/2)
                    lower_key = f"lower_{int((1-alpha)*100)}"
                    upper_key = f"upper_{int((1-alpha)*100)}"
                    intervals[lower_key][series_id] = forecasts[series_id] - z_score * std_estimate
                    intervals[upper_key][series_id] = forecasts[series_id] + z_score * std_estimate

            else:
                try:
                    if self.method == "ets":
                        point_forecast, interval_forecasts = self._predict_ets(
                            model, horizon, confidence_levels
                        )
                    elif self.method == "arima":
                        point_forecast, interval_forecasts = self._predict_arima(
                            model, horizon, confidence_levels
                        )

                    forecasts[series_id] = point_forecast

                    if return_intervals:
                        for alpha in confidence_levels:
                            lower_key = f"lower_{int((1-alpha)*100)}"
                            upper_key = f"upper_{int((1-alpha)*100)}"
                            intervals[lower_key][series_id] = interval_forecasts[lower_key]
                            intervals[upper_key][series_id] = interval_forecasts[upper_key]

                except Exception as e:
                    self.logger.warning(f"Prediction failed for {series_id}: {e}")
                    # Fallback to naive forecast
                    last_value = self.series_data[series_id].iloc[-1]
                    forecasts[series_id] = np.full(horizon, last_value)

        result = {"forecasts": forecasts}
        if return_intervals:
            result.update(intervals)

        self.logger.info(f"Generated forecasts for {len(forecasts)} series")
        return result

    def _fit_ets_model(self, ts_data: pd.Series, series_id: str) -> Any:
        """Fit ETS model to a single time series."""
        # Use statsmodels ETS
        model = StatsETS(
            ts_data,
            seasonal_periods=self.ets_config.get('seasonal_periods', 7),
            error=self.ets_config.get('error', 'add'),
            trend=self.ets_config.get('trend', 'add'),
            seasonal=self.ets_config.get('seasonal', 'add')
        )

        fitted_model = model.fit(
            use_boxcox=self.ets_config.get('use_boxcox', False),
            remove_bias=self.ets_config.get('remove_bias', True)
        )

        return fitted_model

    def _fit_arima_model(self, ts_data: pd.Series, series_id: str) -> Any:
        """Fit ARIMA model to a single time series."""
        seasonal_order = self.arima_config.get('seasonal_order', (1, 1, 1, 7))

        model = StatsARIMA(
            ts_data,
            order=(1, 1, 1),  # Auto-select or use default
            seasonal_order=seasonal_order
        )

        fitted_model = model.fit(
            method=self.arima_config.get('method', 'lbfgs'),
            maxiter=self.arima_config.get('maxiter', 50)
        )

        return fitted_model

    def _predict_ets(
        self,
        model: Any,
        horizon: int,
        confidence_levels: List[float]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Generate ETS predictions with intervals."""
        forecast_result = model.forecast(steps=horizon, return_conf_int=True)

        point_forecast = forecast_result.iloc[:, 0].values
        intervals = {}

        # Extract confidence intervals
        for i, alpha in enumerate(confidence_levels):
            lower_key = f"lower_{int((1-alpha)*100)}"
            upper_key = f"upper_{int((1-alpha)*100)}"

            # Use forecast confidence intervals if available
            if hasattr(forecast_result, 'conf_int'):
                conf_int = model.get_forecast(horizon).conf_int(alpha=alpha)
                intervals[lower_key] = conf_int.iloc[:, 0].values
                intervals[upper_key] = conf_int.iloc[:, 1].values
            else:
                # Fallback: use residual standard error
                residuals = model.resid
                sigma = np.std(residuals)
                z_score = norm.ppf(1 - alpha/2)
                intervals[lower_key] = point_forecast - z_score * sigma
                intervals[upper_key] = point_forecast + z_score * sigma

        return point_forecast, intervals

    def _predict_arima(
        self,
        model: Any,
        horizon: int,
        confidence_levels: List[float]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Generate ARIMA predictions with intervals."""
        forecast_result = model.get_forecast(steps=horizon)
        point_forecast = forecast_result.predicted_mean.values

        intervals = {}
        for alpha in confidence_levels:
            conf_int = forecast_result.conf_int(alpha=alpha)
            lower_key = f"lower_{int((1-alpha)*100)}"
            upper_key = f"upper_{int((1-alpha)*100)}"
            intervals[lower_key] = conf_int.iloc[:, 0].values
            intervals[upper_key] = conf_int.iloc[:, 1].values

        return point_forecast, intervals


class DeepLearningForecaster(BaseForecaster):
    """
    Deep learning forecasting using Temporal Fusion Transformer and N-BEATS.
    """

    def __init__(
        self,
        model_type: str = "tft",
        tft_config: Optional[Dict[str, Any]] = None,
        nbeats_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize deep learning forecaster.

        Args:
            model_type: Model type ("tft" or "nbeats").
            tft_config: Configuration for TFT model.
            nbeats_config: Configuration for N-BEATS model.

        Raises:
            ValueError: If invalid model type is specified.
        """
        super().__init__(f"DeepLearning_{model_type.upper()}")

        if model_type not in ["tft", "nbeats"]:
            raise ValueError(f"Unknown deep learning model: {model_type}")

        self.model_type = model_type
        self.tft_config = tft_config or {}
        self.nbeats_config = nbeats_config or {}

        self.model: Optional[Union[TFTModel, NBEATSModel]] = None
        self.training_data: Optional[List[TimeSeries]] = None

    def fit(
        self,
        data: pd.DataFrame,
        target_col: str = "sales"
    ) -> "DeepLearningForecaster":
        """
        Fit deep learning model to hierarchical time series.

        Args:
            data: Training data.
            target_col: Target variable column name.

        Returns:
            Self for method chaining.
        """
        self.logger.info(f"Fitting {self.model_type} model...")

        # Convert to Darts TimeSeries format
        time_series_list = self._prepare_darts_data(data, target_col)
        self.training_data = time_series_list

        # Initialize and train model
        if self.model_type == "tft":
            self.model = self._create_tft_model()
        elif self.model_type == "nbeats":
            self.model = self._create_nbeats_model()

        # Train the model
        self.model.fit(
            series=time_series_list,
            verbose=True
        )

        self.is_fitted = True
        self.logger.info(f"Successfully fitted {self.model_type} model")

        return self

    def predict(
        self,
        horizon: int,
        return_intervals: bool = True,
        confidence_levels: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate forecasts with uncertainty quantification.

        Args:
            horizon: Forecast horizon.
            return_intervals: Whether to return prediction intervals.
            confidence_levels: Confidence levels for intervals.

        Returns:
            Dictionary with forecasts and prediction intervals.

        Raises:
            ValueError: If not fitted.
        """
        if not self.is_fitted or self.model is None or self.training_data is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if confidence_levels is None:
            confidence_levels = [0.1, 0.05]

        self.logger.info(f"Generating {horizon}-step forecasts...")

        forecasts = {}
        intervals = {f"lower_{int((1-alpha)*100)}": {} for alpha in confidence_levels}
        intervals.update({f"upper_{int((1-alpha)*100)}": {} for alpha in confidence_levels})

        for ts in self.training_data:
            series_id = ts.static_covariates['id'].iloc[0] if hasattr(ts, 'static_covariates') else f"series_{len(forecasts)}"

            try:
                if return_intervals:
                    # Generate probabilistic forecasts
                    prediction = self.model.predict(
                        n=horizon,
                        series=ts,
                        num_samples=100  # For uncertainty quantification
                    )

                    # Extract point forecast and intervals
                    point_forecast = prediction.quantile(0.5).values().flatten()
                    forecasts[series_id] = point_forecast

                    for alpha in confidence_levels:
                        lower_quantile = alpha / 2
                        upper_quantile = 1 - alpha / 2

                        lower_key = f"lower_{int((1-alpha)*100)}"
                        upper_key = f"upper_{int((1-alpha)*100)}"

                        intervals[lower_key][series_id] = prediction.quantile(lower_quantile).values().flatten()
                        intervals[upper_key][series_id] = prediction.quantile(upper_quantile).values().flatten()

                else:
                    # Point forecast only
                    prediction = self.model.predict(n=horizon, series=ts)
                    forecasts[series_id] = prediction.values().flatten()

            except Exception as e:
                self.logger.warning(f"Prediction failed for series {series_id}: {e}")
                # Fallback to last value
                forecasts[series_id] = np.full(horizon, ts.values()[-1])

        result = {"forecasts": forecasts}
        if return_intervals:
            result.update(intervals)

        self.logger.info(f"Generated forecasts for {len(forecasts)} series")
        return result

    def _prepare_darts_data(self, data: pd.DataFrame, target_col: str) -> List[TimeSeries]:
        """Convert data to Darts TimeSeries format."""
        time_series_list = []

        for series_id, series_data in data.groupby('id'):
            # Sort by date
            series_data = series_data.sort_values('date')

            # Create TimeSeries
            ts = TimeSeries.from_dataframe(
                df=series_data,
                time_col='date',
                value_cols=[target_col],
                static_covs_cols=['id'] if 'id' in series_data.columns else None
            )

            time_series_list.append(ts)

        return time_series_list

    def _create_tft_model(self) -> TFTModel:
        """Create and configure TFT model."""
        return TFTModel(
            input_chunk_length=self.tft_config.get('input_chunk_length', 28),
            output_chunk_length=self.tft_config.get('output_chunk_length', 28),
            hidden_size=self.tft_config.get('hidden_size', 16),
            lstm_layers=self.tft_config.get('lstm_layers', 1),
            num_attention_heads=self.tft_config.get('attention_head_size', 4),
            dropout=self.tft_config.get('dropout', 0.1),
            batch_size=self.tft_config.get('batch_size', 32),
            n_epochs=self.tft_config.get('max_epochs', 100),
            learning_rate=self.tft_config.get('learning_rate', 0.03),
            likelihood=None,  # Use default for regression
            random_state=42
        )

    def _create_nbeats_model(self) -> NBEATSModel:
        """Create and configure N-BEATS model."""
        return NBEATSModel(
            input_chunk_length=self.nbeats_config.get('input_chunk_length', 28),
            output_chunk_length=self.nbeats_config.get('output_chunk_length', 28),
            num_stacks=self.nbeats_config.get('num_stacks', 30),
            num_blocks=self.nbeats_config.get('num_blocks', 1),
            num_layers=self.nbeats_config.get('num_layers', 4),
            layer_widths=self.nbeats_config.get('layer_widths', 512),
            expansion_coefficient_dim=self.nbeats_config.get('expansion_coefficient_dim', 5),
            trend_polynomial_degree=self.nbeats_config.get('trend_polynomial_degree', 2),
            dropout=self.nbeats_config.get('dropout', 0.0),
            activation=self.nbeats_config.get('activation', 'ReLU'),
            batch_size=self.nbeats_config.get('batch_size', 32),
            n_epochs=self.nbeats_config.get('max_epochs', 100),
            learning_rate=self.nbeats_config.get('learning_rate', 0.001),
            random_state=42
        )


class ProbabilisticReconciler:
    """
    Probabilistic hierarchical reconciliation that preserves uncertainty.

    Implements minimum trace reconciliation with probabilistic constraints
    that maintain prediction intervals through the reconciliation process.
    """

    def __init__(
        self,
        method: str = "probabilistic_mint",
        weights: str = "wls",
        covariance_type: str = "sample",
        lambda_reg: float = 0.01,
        preserve_uncertainty: bool = True
    ) -> None:
        """
        Initialize probabilistic reconciler.

        Args:
            method: Reconciliation method.
            weights: Weighting scheme for reconciliation.
            covariance_type: Covariance estimation method.
            lambda_reg: Regularization parameter.
            preserve_uncertainty: Whether to preserve prediction intervals.
        """
        self.method = method
        self.weights = weights
        self.covariance_type = covariance_type
        self.lambda_reg = lambda_reg
        self.preserve_uncertainty = preserve_uncertainty
        self.logger = logging.getLogger(__name__)

        # Fitted components
        self.S: Optional[sparse.csr_matrix] = None  # Aggregation matrix
        self.G: Optional[np.ndarray] = None  # Reconciliation matrix
        self.W: Optional[np.ndarray] = None  # Weight matrix

    def fit(
        self,
        aggregation_matrix: sparse.csr_matrix,
        residuals: Optional[Dict[str, np.ndarray]] = None
    ) -> "ProbabilisticReconciler":
        """
        Fit the reconciler using aggregation structure and residuals.

        Args:
            aggregation_matrix: Hierarchical aggregation matrix S.
            residuals: Historical forecast residuals for covariance estimation.

        Returns:
            Self for method chaining.
        """
        self.logger.info("Fitting probabilistic reconciler...")

        self.S = aggregation_matrix

        # Estimate weight matrix
        if self.weights == "ols":
            # Ordinary least squares (identity weights)
            self.W = np.eye(self.S.shape[0])
        elif self.weights == "wls" and residuals is not None:
            # Weighted least squares based on forecast accuracy
            self.W = self._estimate_weight_matrix(residuals)
        else:
            # Fallback to identity
            self.W = np.eye(self.S.shape[0])

        # Compute reconciliation matrix
        self.G = self._compute_reconciliation_matrix()

        self.logger.info("Reconciler fitting completed")
        return self

    def reconcile(
        self,
        forecasts: Dict[str, np.ndarray],
        intervals: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    ) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, Dict[str, np.ndarray]]]]:
        """
        Reconcile forecasts and optionally preserve prediction intervals.

        Args:
            forecasts: Point forecasts for all hierarchy levels.
            intervals: Prediction intervals (optional).

        Returns:
            Tuple of (reconciled_forecasts, reconciled_intervals).

        Raises:
            ValueError: If reconciler is not fitted.
        """
        if self.S is None or self.G is None:
            raise ValueError("Reconciler not fitted. Call fit() first.")

        self.logger.info("Reconciling forecasts...")

        # Convert forecasts to matrix format
        forecast_matrix = self._dict_to_matrix(forecasts)

        # Apply reconciliation
        reconciled_matrix = self.G @ forecast_matrix

        # Convert back to dictionary format
        reconciled_forecasts = self._matrix_to_dict(reconciled_matrix, list(forecasts.keys()))

        # Reconcile intervals if provided and requested
        reconciled_intervals = None
        if intervals is not None and self.preserve_uncertainty:
            reconciled_intervals = self._reconcile_intervals(intervals)

        self.logger.info("Forecast reconciliation completed")
        return reconciled_forecasts, reconciled_intervals

    def _estimate_weight_matrix(self, residuals: Dict[str, np.ndarray]) -> np.ndarray:
        """Estimate weight matrix from forecast residuals."""
        self.logger.info(f"Estimating weight matrix using {self.covariance_type} covariance...")

        # Stack residuals
        residual_matrix = np.column_stack([residuals[key] for key in sorted(residuals.keys())])

        # Estimate covariance matrix
        if self.covariance_type == "sample":
            cov_estimator = EmpiricalCovariance()
        elif self.covariance_type == "ledoit_wolf":
            cov_estimator = LedoitWolf()
        else:
            # Fallback to empirical
            cov_estimator = EmpiricalCovariance()

        cov_matrix = cov_estimator.fit(residual_matrix).covariance_

        # Regularize if needed
        if self.lambda_reg > 0:
            cov_matrix += self.lambda_reg * np.eye(cov_matrix.shape[0])

        # Weight matrix is inverse of covariance
        try:
            W = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            self.logger.warning("Covariance matrix is singular, using pseudo-inverse")
            W = np.linalg.pinv(cov_matrix)

        return W

    def _compute_reconciliation_matrix(self) -> np.ndarray:
        """Compute the reconciliation matrix G."""
        self.logger.info("Computing reconciliation matrix...")

        S_dense = self.S.toarray()

        if self.method == "probabilistic_mint":
            # Minimum trace reconciliation with probabilistic constraints
            try:
                # G = S(S^T W S)^{-1} S^T W
                STWS = S_dense.T @ self.W @ S_dense
                STWS_inv = np.linalg.inv(STWS + self.lambda_reg * np.eye(STWS.shape[0]))
                G = S_dense @ STWS_inv @ S_dense.T @ self.W

            except np.linalg.LinAlgError:
                self.logger.warning("Using pseudo-inverse for reconciliation matrix")
                STWS = S_dense.T @ self.W @ S_dense
                STWS_inv = np.linalg.pinv(STWS + self.lambda_reg * np.eye(STWS.shape[0]))
                G = S_dense @ STWS_inv @ S_dense.T @ self.W

        else:
            raise ValueError(f"Unknown reconciliation method: {self.method}")

        return G

    def _reconcile_intervals(
        self,
        intervals: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Reconcile prediction intervals while preserving uncertainty."""
        self.logger.info("Reconciling prediction intervals...")

        reconciled_intervals = {}

        for interval_name, interval_dict in intervals.items():
            # Convert intervals to matrix format
            interval_matrix = self._dict_to_matrix(interval_dict)

            # Apply reconciliation transformation
            reconciled_interval_matrix = self.G @ interval_matrix

            # Convert back to dictionary format
            reconciled_intervals[interval_name] = self._matrix_to_dict(
                reconciled_interval_matrix, list(interval_dict.keys())
            )

        return reconciled_intervals

    def _dict_to_matrix(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert dictionary of arrays to matrix format."""
        series_ids = sorted(data_dict.keys())
        return np.column_stack([data_dict[sid] for sid in series_ids])

    def _matrix_to_dict(self, matrix: np.ndarray, series_ids: List[str]) -> Dict[str, np.ndarray]:
        """Convert matrix back to dictionary format."""
        return {sid: matrix[:, i] for i, sid in enumerate(series_ids)}

    def compute_coherence_score(self, reconciled_forecasts: Dict[str, np.ndarray]) -> float:
        """
        Compute coherence score for reconciled forecasts.

        Args:
            reconciled_forecasts: Reconciled forecasts.

        Returns:
            Coherence score between 0 and 1 (1 = perfectly coherent).
        """
        if self.S is None:
            return 0.0

        # Convert to matrix format
        forecast_matrix = self._dict_to_matrix(reconciled_forecasts)

        # Check coherence: S @ bottom_forecasts should equal aggregated forecasts
        bottom_forecasts = forecast_matrix[-self.S.shape[1]:, :]  # Bottom level
        expected_aggregates = self.S @ bottom_forecasts
        actual_aggregates = forecast_matrix[:-self.S.shape[1], :]  # Upper levels

        # Compute relative errors
        relative_errors = np.abs(expected_aggregates - actual_aggregates) / (np.abs(actual_aggregates) + 1e-8)
        coherence_score = 1.0 - np.mean(relative_errors)

        return max(0.0, min(1.0, coherence_score))


class HierarchicalEnsembleForecaster:
    """
    Main ensemble forecaster combining statistical and deep learning models
    with probabilistic hierarchical reconciliation.
    """

    def __init__(
        self,
        statistical_configs: Dict[str, Dict[str, Any]],
        deep_learning_configs: Dict[str, Dict[str, Any]],
        ensemble_weights: Dict[str, float],
        reconciler_config: Dict[str, Any]
    ) -> None:
        """
        Initialize hierarchical ensemble forecaster.

        Args:
            statistical_configs: Configurations for statistical models.
            deep_learning_configs: Configurations for deep learning models.
            ensemble_weights: Weights for ensemble combination.
            reconciler_config: Configuration for reconciliation.
        """
        self.statistical_configs = statistical_configs
        self.deep_learning_configs = deep_learning_configs
        self.ensemble_weights = ensemble_weights
        self.reconciler_config = reconciler_config
        self.logger = logging.getLogger(__name__)

        # Model components
        self.statistical_models: Dict[str, StatisticalForecaster] = {}
        self.deep_learning_models: Dict[str, DeepLearningForecaster] = {}
        self.reconciler = ProbabilisticReconciler(**reconciler_config)

        self.is_fitted = False

    def fit(
        self,
        data: pd.DataFrame,
        aggregation_matrix: sparse.csr_matrix,
        target_col: str = "sales"
    ) -> "HierarchicalEnsembleForecaster":
        """
        Fit all ensemble components.

        Args:
            data: Training data.
            aggregation_matrix: Hierarchical aggregation matrix.
            target_col: Target variable column name.

        Returns:
            Self for method chaining.
        """
        self.logger.info("Fitting hierarchical ensemble forecaster...")

        # Fit statistical models
        for name, config in self.statistical_configs.items():
            self.logger.info(f"Fitting statistical model: {name}")
            if name == "ets":
                model = StatisticalForecaster("ets", ets_config=config)
            elif name == "arima":
                model = StatisticalForecaster("arima", arima_config=config)
            else:
                continue

            model.fit(data, target_col)
            self.statistical_models[name] = model

        # Fit deep learning models
        for name, config in self.deep_learning_configs.items():
            self.logger.info(f"Fitting deep learning model: {name}")
            if name == "tft":
                model = DeepLearningForecaster("tft", tft_config=config)
            elif name == "nbeats":
                model = DeepLearningForecaster("nbeats", nbeats_config=config)
            else:
                continue

            model.fit(data, target_col)
            self.deep_learning_models[name] = model

        # Fit reconciler (compute residuals if needed)
        self.reconciler.fit(aggregation_matrix)

        self.is_fitted = True
        self.logger.info("Ensemble fitting completed")

        return self

    def predict(
        self,
        horizon: int,
        return_intervals: bool = True,
        confidence_levels: Optional[List[float]] = None
    ) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Generate ensemble forecasts with uncertainty quantification.

        Args:
            horizon: Forecast horizon.
            return_intervals: Whether to return prediction intervals.
            confidence_levels: Confidence levels for intervals.

        Returns:
            Dictionary with reconciled forecasts and intervals.

        Raises:
            ValueError: If not fitted.
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        if confidence_levels is None:
            confidence_levels = [0.1, 0.05]

        self.logger.info(f"Generating ensemble forecasts for horizon {horizon}...")

        # Collect predictions from all models
        all_predictions = {}

        # Statistical model predictions
        for name, model in self.statistical_models.items():
            predictions = model.predict(horizon, return_intervals, confidence_levels)
            all_predictions[name] = predictions

        # Deep learning model predictions
        for name, model in self.deep_learning_models.items():
            predictions = model.predict(horizon, return_intervals, confidence_levels)
            all_predictions[name] = predictions

        # Combine predictions using ensemble weights
        ensemble_forecasts, ensemble_intervals = self._combine_predictions(
            all_predictions, confidence_levels, return_intervals
        )

        # Apply hierarchical reconciliation
        reconciled_forecasts, reconciled_intervals = self.reconciler.reconcile(
            ensemble_forecasts, ensemble_intervals if return_intervals else None
        )

        # Compute coherence score
        coherence_score = self.reconciler.compute_coherence_score(reconciled_forecasts)

        result = {
            "forecasts": reconciled_forecasts,
            "coherence_score": coherence_score
        }

        if return_intervals and reconciled_intervals is not None:
            result.update(reconciled_intervals)

        self.logger.info(f"Ensemble prediction completed (coherence: {coherence_score:.3f})")
        return result

    def _combine_predictions(
        self,
        all_predictions: Dict[str, Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]],
        confidence_levels: List[float],
        return_intervals: bool
    ) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, Dict[str, np.ndarray]]]]:
        """Combine predictions from multiple models using ensemble weights."""
        self.logger.info("Combining model predictions...")

        # Initialize ensemble forecasts
        ensemble_forecasts = {}
        ensemble_intervals = {} if return_intervals else None

        # Get all series IDs
        all_series_ids = set()
        for model_name, predictions in all_predictions.items():
            if "forecasts" in predictions:
                all_series_ids.update(predictions["forecasts"].keys())

        # Combine forecasts for each series
        for series_id in all_series_ids:
            series_forecasts = []
            series_weights = []

            # Collect forecasts from all models
            for model_name, predictions in all_predictions.items():
                if "forecasts" in predictions and series_id in predictions["forecasts"]:
                    series_forecasts.append(predictions["forecasts"][series_id])
                    series_weights.append(self.ensemble_weights.get(model_name, 0.0))

            if series_forecasts:
                # Weighted average of forecasts
                series_weights = np.array(series_weights)
                series_weights = series_weights / series_weights.sum()  # Normalize

                ensemble_forecast = np.average(series_forecasts, weights=series_weights, axis=0)
                ensemble_forecasts[series_id] = ensemble_forecast

                # Combine intervals if requested
                if return_intervals and ensemble_intervals is not None:
                    for alpha in confidence_levels:
                        lower_key = f"lower_{int((1-alpha)*100)}"
                        upper_key = f"upper_{int((1-alpha)*100)}"

                        if lower_key not in ensemble_intervals:
                            ensemble_intervals[lower_key] = {}
                            ensemble_intervals[upper_key] = {}

                        series_lower = []
                        series_upper = []

                        for i, (model_name, predictions) in enumerate(all_predictions.items()):
                            if (lower_key in predictions and series_id in predictions[lower_key] and
                                upper_key in predictions and series_id in predictions[upper_key]):
                                series_lower.append(predictions[lower_key][series_id])
                                series_upper.append(predictions[upper_key][series_id])

                        if series_lower and series_upper:
                            # Use minimum lower bound and maximum upper bound for conservative intervals
                            ensemble_intervals[lower_key][series_id] = np.minimum.reduce(series_lower)
                            ensemble_intervals[upper_key][series_id] = np.maximum.reduce(series_upper)

        return ensemble_forecasts, ensemble_intervals