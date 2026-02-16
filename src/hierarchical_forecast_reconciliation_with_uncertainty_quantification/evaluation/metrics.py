"""
Comprehensive evaluation metrics for hierarchical forecasting with uncertainty quantification.

This module implements all major forecasting evaluation metrics including those
specific to hierarchical time series and probabilistic forecasting.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from properscoring import crps_ensemble
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore', category=RuntimeWarning)


class IntervalMetrics:
    """Metrics for evaluating prediction intervals."""

    @staticmethod
    def coverage_probability(
        actuals: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray
    ) -> float:
        """
        Calculate empirical coverage probability.

        Args:
            actuals: Actual values.
            lower_bounds: Lower bounds of prediction intervals.
            upper_bounds: Upper bounds of prediction intervals.

        Returns:
            Coverage probability (0-1).
        """
        coverage = np.mean((actuals >= lower_bounds) & (actuals <= upper_bounds))
        return float(coverage)

    @staticmethod
    def interval_width(
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray
    ) -> float:
        """
        Calculate mean interval width.

        Args:
            lower_bounds: Lower bounds of prediction intervals.
            upper_bounds: Upper bounds of prediction intervals.

        Returns:
            Mean interval width.
        """
        return float(np.mean(upper_bounds - lower_bounds))

    @staticmethod
    def interval_score(
        actuals: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        alpha: float = 0.1
    ) -> float:
        """
        Calculate interval score (smaller is better).

        Args:
            actuals: Actual values.
            lower_bounds: Lower bounds of prediction intervals.
            upper_bounds: Upper bounds of prediction intervals.
            alpha: Significance level (e.g., 0.1 for 90% intervals).

        Returns:
            Interval score.
        """
        # Interval width
        width = upper_bounds - lower_bounds

        # Penalties for coverage violations
        lower_penalty = 2 * alpha * np.maximum(lower_bounds - actuals, 0)
        upper_penalty = 2 * alpha * np.maximum(actuals - upper_bounds, 0)

        # Total score
        scores = width + lower_penalty + upper_penalty
        return float(np.mean(scores))

    @staticmethod
    def quantile_score(
        actuals: np.ndarray,
        predictions: np.ndarray,
        quantile: float
    ) -> float:
        """
        Calculate quantile score (pinball loss).

        Args:
            actuals: Actual values.
            predictions: Predicted quantiles.
            quantile: Quantile level (0-1).

        Returns:
            Quantile score.
        """
        errors = actuals - predictions
        scores = np.maximum(quantile * errors, (quantile - 1) * errors)
        return float(np.mean(scores))

    @staticmethod
    def mean_scaled_interval_score(
        actuals: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        naive_errors: np.ndarray,
        alpha: float = 0.1
    ) -> float:
        """
        Calculate Mean Scaled Interval Score (MSIS).

        Args:
            actuals: Actual values.
            lower_bounds: Lower bounds of prediction intervals.
            upper_bounds: Upper bounds of prediction intervals.
            naive_errors: Naive forecast errors for scaling.
            alpha: Significance level.

        Returns:
            MSIS value.
        """
        interval_scores = IntervalMetrics.interval_score(
            actuals, lower_bounds, upper_bounds, alpha
        )

        # Scale by naive forecast MAE
        scaling_factor = np.mean(np.abs(naive_errors))
        if scaling_factor == 0:
            return float('inf')

        return interval_scores / scaling_factor


class CoherenceMetrics:
    """Metrics for evaluating hierarchical coherence."""

    @staticmethod
    def coherence_error(
        predictions: Dict[str, np.ndarray],
        aggregation_matrix: np.ndarray,
        bottom_level_keys: List[str]
    ) -> float:
        """
        Calculate relative coherence error.

        Args:
            predictions: Dictionary of predictions for all hierarchy levels.
            aggregation_matrix: Matrix defining hierarchical structure.
            bottom_level_keys: Keys for bottom-level series.

        Returns:
            Relative coherence error (0 = perfect coherence).
        """
        try:
            # Extract bottom-level predictions
            bottom_predictions = np.column_stack([
                predictions[key] for key in bottom_level_keys if key in predictions
            ])

            # Expected aggregated predictions
            expected_aggregates = aggregation_matrix @ bottom_predictions

            # Actual aggregated predictions
            upper_level_keys = [key for key in predictions.keys() if key not in bottom_level_keys]
            if not upper_level_keys:
                return 0.0

            actual_aggregates = np.column_stack([
                predictions[key] for key in upper_level_keys if key in predictions
            ])

            # Compute relative error
            abs_diff = np.abs(expected_aggregates - actual_aggregates)
            relative_error = abs_diff / (np.abs(actual_aggregates) + 1e-8)

            return float(np.mean(relative_error))

        except Exception:
            return 1.0  # Return maximum error if computation fails

    @staticmethod
    def structural_coherence_score(
        predictions: Dict[str, np.ndarray],
        hierarchy_structure: Dict[str, List[str]]
    ) -> float:
        """
        Calculate structural coherence score based on hierarchy relationships.

        Args:
            predictions: Dictionary of predictions for all hierarchy levels.
            hierarchy_structure: Dictionary defining parent-child relationships.

        Returns:
            Coherence score (1 = perfect coherence, 0 = no coherence).
        """
        coherence_scores = []

        for parent, children in hierarchy_structure.items():
            if parent not in predictions:
                continue

            parent_pred = predictions[parent]
            child_preds = [predictions[child] for child in children if child in predictions]

            if not child_preds:
                continue

            # Sum of children should equal parent
            children_sum = np.sum(child_preds, axis=0)
            relative_error = np.abs(parent_pred - children_sum) / (np.abs(parent_pred) + 1e-8)
            coherence_score = 1.0 - np.mean(relative_error)

            coherence_scores.append(max(0.0, coherence_score))

        return float(np.mean(coherence_scores)) if coherence_scores else 0.0


class HierarchicalMetrics:
    """
    Comprehensive metrics for hierarchical forecasting evaluation.

    Implements standard forecasting metrics along with specialized metrics
    for hierarchical time series and probabilistic forecasting.
    """

    def __init__(self) -> None:
        """Initialize hierarchical metrics calculator."""
        self.logger = logging.getLogger(__name__)

    def compute_wrmsse(
        self,
        predictions: Dict[str, np.ndarray],
        actuals: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute Weighted Root Mean Squared Scaled Error (WRMSSE).

        WRMSSE is the primary metric for the M5 competition, combining
        accuracy across all hierarchy levels with appropriate weighting.

        Args:
            predictions: Dictionary of predictions by series ID.
            actuals: Dictionary of actual values by series ID.
            weights: Optional weights for each series (defaults to equal weights).

        Returns:
            WRMSSE value (lower is better).
        """
        try:
            rmsse_values = []
            series_weights = []

            for series_id in predictions.keys():
                if series_id not in actuals:
                    continue

                pred = np.array(predictions[series_id])
                actual = np.array(actuals[series_id])

                # Calculate RMSSE for this series
                rmsse = self._compute_rmsse_single_series(pred, actual)
                rmsse_values.append(rmsse)

                # Use provided weight or equal weighting
                weight = weights.get(series_id, 1.0) if weights else 1.0
                series_weights.append(weight)

            if not rmsse_values:
                return float('inf')

            # Weighted average
            series_weights = np.array(series_weights)
            series_weights = series_weights / series_weights.sum()  # Normalize

            wrmsse = np.average(rmsse_values, weights=series_weights)
            return float(wrmsse)

        except Exception as e:
            self.logger.warning(f"WRMSSE computation failed: {e}")
            return float('inf')

    def _compute_rmsse_single_series(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> float:
        """Compute RMSSE for a single time series."""
        if len(predictions) != len(actuals) or len(actuals) == 0:
            return float('inf')

        # MSE of predictions
        mse_forecast = mean_squared_error(actuals, predictions)

        # MSE of naive forecast (previous period)
        if len(actuals) < 2:
            return float('inf')

        naive_predictions = np.append(actuals[0], actuals[:-1])
        mse_naive = mean_squared_error(actuals, naive_predictions)

        if mse_naive == 0:
            return 0.0 if mse_forecast == 0 else float('inf')

        return float(np.sqrt(mse_forecast / mse_naive))

    def compute_mase(
        self,
        predictions: Dict[str, np.ndarray],
        actuals: Dict[str, np.ndarray],
        seasonal_period: int = 7
    ) -> float:
        """
        Compute Mean Absolute Scaled Error (MASE).

        Args:
            predictions: Dictionary of predictions by series ID.
            actuals: Dictionary of actual values by series ID.
            seasonal_period: Seasonal period for naive forecast.

        Returns:
            MASE value (lower is better).
        """
        try:
            mase_values = []

            for series_id in predictions.keys():
                if series_id not in actuals:
                    continue

                pred = np.array(predictions[series_id])
                actual = np.array(actuals[series_id])

                # MAE of predictions
                mae_forecast = mean_absolute_error(actual, pred)

                # MAE of seasonal naive forecast
                if len(actual) <= seasonal_period:
                    naive_mae = np.mean(np.abs(actual[1:] - actual[:-1]))
                else:
                    seasonal_naive = actual[:-seasonal_period]
                    naive_mae = mean_absolute_error(actual[seasonal_period:], seasonal_naive)

                if naive_mae == 0:
                    mase = 0.0 if mae_forecast == 0 else float('inf')
                else:
                    mase = mae_forecast / naive_mae

                mase_values.append(mase)

            return float(np.mean(mase_values)) if mase_values else float('inf')

        except Exception as e:
            self.logger.warning(f"MASE computation failed: {e}")
            return float('inf')

    def compute_smape(
        self,
        predictions: Dict[str, np.ndarray],
        actuals: Dict[str, np.ndarray]
    ) -> float:
        """
        Compute symmetric Mean Absolute Percentage Error (sMAPE).

        Args:
            predictions: Dictionary of predictions by series ID.
            actuals: Dictionary of actual values by series ID.

        Returns:
            sMAPE value as percentage (lower is better).
        """
        try:
            smape_values = []

            for series_id in predictions.keys():
                if series_id not in actuals:
                    continue

                pred = np.array(predictions[series_id])
                actual = np.array(actuals[series_id])

                # Compute sMAPE
                numerator = np.abs(actual - pred)
                denominator = (np.abs(actual) + np.abs(pred)) / 2

                # Handle division by zero
                valid_mask = denominator != 0
                if np.sum(valid_mask) == 0:
                    smape = 0.0
                else:
                    smape = 100 * np.mean(numerator[valid_mask] / denominator[valid_mask])

                smape_values.append(smape)

            return float(np.mean(smape_values)) if smape_values else float('inf')

        except Exception as e:
            self.logger.warning(f"sMAPE computation failed: {e}")
            return float('inf')

    def compute_crps(
        self,
        predictions_ensemble: Dict[str, np.ndarray],
        actuals: Dict[str, np.ndarray]
    ) -> float:
        """
        Compute Continuous Ranked Probability Score (CRPS).

        Args:
            predictions_ensemble: Dictionary of prediction ensembles.
            actuals: Dictionary of actual values.

        Returns:
            CRPS value (lower is better).
        """
        try:
            crps_values = []

            for series_id in predictions_ensemble.keys():
                if series_id not in actuals:
                    continue

                pred_ensemble = np.array(predictions_ensemble[series_id])
                actual = np.array(actuals[series_id])

                # Ensure ensemble has correct shape
                if pred_ensemble.ndim == 1:
                    pred_ensemble = pred_ensemble.reshape(1, -1)

                # Compute CRPS for each time step
                for t in range(min(pred_ensemble.shape[1], len(actual))):
                    crps_t = crps_ensemble(actual[t], pred_ensemble[:, t])
                    crps_values.append(crps_t)

            return float(np.mean(crps_values)) if crps_values else float('inf')

        except Exception as e:
            self.logger.warning(f"CRPS computation failed: {e}")
            return float('inf')

    def compute_interval_metrics(
        self,
        actuals: Dict[str, np.ndarray],
        intervals: Dict[str, Dict[str, np.ndarray]],
        confidence_levels: List[float]
    ) -> Dict[str, float]:
        """
        Compute prediction interval evaluation metrics.

        Args:
            actuals: Dictionary of actual values.
            intervals: Dictionary of prediction intervals.
            confidence_levels: Confidence levels for intervals.

        Returns:
            Dictionary of interval metrics.
        """
        metrics = {}

        for alpha in confidence_levels:
            confidence = int((1 - alpha) * 100)
            lower_key = f"lower_{confidence}"
            upper_key = f"upper_{confidence}"

            if lower_key not in intervals or upper_key not in intervals:
                continue

            # Aggregate values across all series
            all_actuals = []
            all_lower = []
            all_upper = []

            for series_id in actuals.keys():
                if (series_id in intervals[lower_key] and
                    series_id in intervals[upper_key]):

                    all_actuals.extend(actuals[series_id])
                    all_lower.extend(intervals[lower_key][series_id])
                    all_upper.extend(intervals[upper_key][series_id])

            if all_actuals:
                all_actuals = np.array(all_actuals)
                all_lower = np.array(all_lower)
                all_upper = np.array(all_upper)

                # Coverage probability
                coverage = IntervalMetrics.coverage_probability(
                    all_actuals, all_lower, all_upper
                )
                metrics[f"coverage_{confidence}"] = coverage

                # Mean interval width
                width = IntervalMetrics.interval_width(all_lower, all_upper)
                metrics[f"width_{confidence}"] = width

                # Interval score
                score = IntervalMetrics.interval_score(
                    all_actuals, all_lower, all_upper, alpha
                )
                metrics[f"interval_score_{confidence}"] = score

        return metrics

    def compute_reconciliation_metrics(
        self,
        predictions: Dict[str, np.ndarray],
        aggregation_matrix: Optional[np.ndarray] = None,
        hierarchy_structure: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, float]:
        """
        Compute hierarchical reconciliation metrics.

        Args:
            predictions: Dictionary of predictions for all levels.
            aggregation_matrix: Matrix defining hierarchical aggregation.
            hierarchy_structure: Dictionary of parent-child relationships.

        Returns:
            Dictionary of coherence metrics.
        """
        metrics = {}

        # Coherence score based on aggregation matrix
        if aggregation_matrix is not None:
            # Assume bottom level is the last subset of keys
            all_keys = list(predictions.keys())
            n_bottom = aggregation_matrix.shape[1]
            bottom_keys = all_keys[-n_bottom:] if len(all_keys) >= n_bottom else all_keys

            coherence_error = CoherenceMetrics.coherence_error(
                predictions, aggregation_matrix, bottom_keys
            )
            metrics["reconciliation_coherence"] = max(0.0, 1.0 - coherence_error)

        # Structural coherence score
        if hierarchy_structure is not None:
            structural_score = CoherenceMetrics.structural_coherence_score(
                predictions, hierarchy_structure
            )
            metrics["structural_coherence"] = structural_score

        return metrics

    def compute_all_metrics(
        self,
        predictions: Dict[str, np.ndarray],
        actuals: Dict[str, np.ndarray],
        intervals: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        hierarchy_data: Optional[Dict[str, pd.DataFrame]] = None,
        aggregation_matrix: Optional[np.ndarray] = None,
        confidence_levels: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.

        Args:
            predictions: Dictionary of predictions.
            actuals: Dictionary of actual values.
            intervals: Dictionary of prediction intervals.
            hierarchy_data: Hierarchical data for structure analysis.
            aggregation_matrix: Aggregation matrix for coherence analysis.
            confidence_levels: Confidence levels for interval evaluation.

        Returns:
            Dictionary of all computed metrics.
        """
        if confidence_levels is None:
            confidence_levels = [0.1, 0.05]

        self.logger.info("Computing all evaluation metrics...")

        metrics = {}

        # Point forecast metrics
        metrics["WRMSSE"] = self.compute_wrmsse(predictions, actuals)
        metrics["MASE"] = self.compute_mase(predictions, actuals)
        metrics["sMAPE"] = self.compute_smape(predictions, actuals)

        # Interval metrics
        if intervals is not None:
            interval_metrics = self.compute_interval_metrics(
                actuals, intervals, confidence_levels
            )
            metrics.update(interval_metrics)

        # Reconciliation metrics
        if aggregation_matrix is not None:
            reconciliation_metrics = self.compute_reconciliation_metrics(
                predictions, aggregation_matrix
            )
            metrics.update(reconciliation_metrics)

        # Add CRPS if ensemble predictions available
        # (This would require ensemble format predictions)

        self.logger.info(f"Computed {len(metrics)} evaluation metrics")
        return metrics

    def create_performance_report(
        self,
        metrics: Dict[str, float],
        target_metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Create a formatted performance report.

        Args:
            metrics: Dictionary of computed metrics.
            target_metrics: Optional target values for comparison.

        Returns:
            Formatted performance report string.
        """
        report_lines = ["=== Hierarchical Forecasting Performance Report ===\n"]

        # Point forecast metrics
        report_lines.append("Point Forecast Metrics:")
        point_metrics = ["WRMSSE", "MASE", "sMAPE"]
        for metric in point_metrics:
            if metric in metrics:
                value = metrics[metric]
                target = target_metrics.get(metric, None) if target_metrics else None

                line = f"  {metric}: {value:.4f}"
                if target is not None:
                    status = "✓" if value <= target else "✗"
                    line += f" (target: {target:.4f}) {status}"

                report_lines.append(line)

        # Interval metrics
        interval_metrics = [key for key in metrics.keys() if "coverage_" in key or "width_" in key]
        if interval_metrics:
            report_lines.append("\nPrediction Interval Metrics:")
            for metric in sorted(interval_metrics):
                value = metrics[metric]
                target = target_metrics.get(metric, None) if target_metrics else None

                line = f"  {metric}: {value:.4f}"
                if target is not None:
                    if "coverage_" in metric:
                        # For coverage, we want to be close to the nominal level
                        status = "✓" if abs(value - target) <= 0.05 else "✗"
                    else:
                        status = "✓" if value <= target else "✗"
                    line += f" (target: {target:.4f}) {status}"

                report_lines.append(line)

        # Reconciliation metrics
        reconciliation_metrics = [key for key in metrics.keys() if "coherence" in key]
        if reconciliation_metrics:
            report_lines.append("\nReconciliation Metrics:")
            for metric in sorted(reconciliation_metrics):
                value = metrics[metric]
                target = target_metrics.get(metric, None) if target_metrics else None

                line = f"  {metric}: {value:.4f}"
                if target is not None:
                    status = "✓" if value >= target else "✗"
                    line += f" (target: {target:.4f}) {status}"

                report_lines.append(line)

        # Overall assessment
        if target_metrics:
            met_targets = 0
            total_targets = 0

            for metric, target in target_metrics.items():
                if metric in metrics:
                    value = metrics[metric]
                    total_targets += 1

                    if "coverage_" in metric:
                        # Coverage should be close to nominal
                        if abs(value - target) <= 0.05:
                            met_targets += 1
                    elif "coherence" in metric:
                        # Coherence should be high
                        if value >= target:
                            met_targets += 1
                    else:
                        # Other metrics should be low
                        if value <= target:
                            met_targets += 1

            report_lines.append(f"\nOverall Performance: {met_targets}/{total_targets} targets met")

        return "\n".join(report_lines)

    def statistical_significance_test(
        self,
        predictions1: Dict[str, np.ndarray],
        predictions2: Dict[str, np.ndarray],
        actuals: Dict[str, np.ndarray],
        test_type: str = "dm"
    ) -> Dict[str, float]:
        """
        Perform statistical significance tests between two forecasting methods.

        Args:
            predictions1: Predictions from first method.
            predictions2: Predictions from second method.
            actuals: Actual values.
            test_type: Type of test ("dm" for Diebold-Mariano).

        Returns:
            Dictionary with test statistics and p-values.
        """
        if test_type == "dm":
            return self._diebold_mariano_test(predictions1, predictions2, actuals)
        else:
            raise ValueError(f"Unknown test type: {test_type}")

    def _diebold_mariano_test(
        self,
        predictions1: Dict[str, np.ndarray],
        predictions2: Dict[str, np.ndarray],
        actuals: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Perform Diebold-Mariano test for forecast accuracy."""
        try:
            # Collect squared errors for both methods
            errors1_all = []
            errors2_all = []

            for series_id in actuals.keys():
                if series_id in predictions1 and series_id in predictions2:
                    actual = np.array(actuals[series_id])
                    pred1 = np.array(predictions1[series_id])
                    pred2 = np.array(predictions2[series_id])

                    # Squared errors
                    errors1 = (actual - pred1) ** 2
                    errors2 = (actual - pred2) ** 2

                    errors1_all.extend(errors1)
                    errors2_all.extend(errors2)

            if not errors1_all:
                return {"dm_statistic": 0.0, "p_value": 1.0}

            errors1_all = np.array(errors1_all)
            errors2_all = np.array(errors2_all)

            # Loss differential
            d = errors1_all - errors2_all

            # Test statistic
            d_mean = np.mean(d)
            d_var = np.var(d, ddof=1)

            if d_var == 0:
                return {"dm_statistic": 0.0, "p_value": 1.0}

            dm_stat = d_mean / np.sqrt(d_var / len(d))
            p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

            return {
                "dm_statistic": float(dm_stat),
                "p_value": float(p_value)
            }

        except Exception as e:
            self.logger.warning(f"Diebold-Mariano test failed: {e}")
            return {"dm_statistic": 0.0, "p_value": 1.0}