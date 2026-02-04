#!/usr/bin/env python3
"""
Evaluation script for hierarchical ensemble forecasting.

This script provides comprehensive evaluation of trained models including
performance metrics, visualization, and comparison with baselines.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hierarchical_forecast_reconciliation_with_uncertainty_quantification.utils.config import (
    load_config, setup_logging, set_random_seeds
)
from hierarchical_forecast_reconciliation_with_uncertainty_quantification.training.trainer import (
    HierarchicalForecastTrainer
)
from hierarchical_forecast_reconciliation_with_uncertainty_quantification.evaluation.metrics import (
    HierarchicalMetrics
)
from hierarchical_forecast_reconciliation_with_uncertainty_quantification.data.loader import (
    M5DataLoader, HierarchicalDataBuilder
)
from hierarchical_forecast_reconciliation_with_uncertainty_quantification.data.preprocessing import (
    M5Preprocessor, HierarchyBuilder
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate hierarchical ensemble forecasting model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model and configuration
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model file"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )

    # Data
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to M5 data directory (overrides config)"
    )

    parser.add_argument(
        "--test-data-only",
        action="store_true",
        help="Evaluate only on test data (skip validation)"
    )

    # Evaluation options
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=28,
        help="Forecast horizon for evaluation"
    )

    parser.add_argument(
        "--confidence-levels",
        nargs="+",
        type=float,
        default=[0.1, 0.05],
        help="Confidence levels for prediction intervals"
    )

    parser.add_argument(
        "--compare-baselines",
        action="store_true",
        help="Compare with baseline methods"
    )

    parser.add_argument(
        "--statistical-tests",
        action="store_true",
        help="Perform statistical significance tests"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )

    parser.add_argument(
        "--create-plots",
        action="store_true",
        help="Create visualization plots"
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions to CSV files"
    )

    # Target metrics for comparison
    parser.add_argument(
        "--target-wrmsse",
        type=float,
        default=0.52,
        help="Target WRMSSE value"
    )

    parser.add_argument(
        "--target-coverage-90",
        type=float,
        default=0.88,
        help="Target 90% coverage probability"
    )

    parser.add_argument(
        "--target-coherence",
        type=float,
        default=0.99,
        help="Target reconciliation coherence"
    )

    parser.add_argument(
        "--target-crps",
        type=float,
        default=0.045,
        help="Target CRPS value"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output"
    )

    return parser.parse_args()


class ModelEvaluator:
    """Comprehensive model evaluation with metrics and visualizations."""

    def __init__(
        self,
        model_path: str,
        config: Dict[str, Any],
        data_path: Optional[str] = None
    ) -> None:
        """
        Initialize model evaluator.

        Args:
            model_path: Path to trained model.
            config: Configuration dictionary.
            data_path: Optional path to data directory.
        """
        self.model_path = model_path
        self.config = config
        self.data_path = data_path or config.get('data', {}).get('path', 'data/m5')
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.trainer = HierarchicalForecastTrainer(config, self.data_path)
        self.metrics_calculator = HierarchicalMetrics()

        # Load trained model
        self.model = self.trainer.load_model(model_path)
        self.logger.info(f"Loaded model from: {model_path}")

    def evaluate_model(
        self,
        forecast_horizon: int,
        confidence_levels: list,
        test_data_only: bool = False
    ) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation.

        Args:
            forecast_horizon: Number of steps to forecast.
            confidence_levels: Confidence levels for intervals.
            test_data_only: Whether to evaluate only on test data.

        Returns:
            Dictionary containing all evaluation results.
        """
        self.logger.info("Starting model evaluation...")

        # Prepare evaluation data
        eval_data = self._prepare_evaluation_data(test_data_only)

        # Generate predictions
        predictions = self._generate_predictions(eval_data, forecast_horizon, confidence_levels)

        # Compute metrics
        metrics = self._compute_evaluation_metrics(predictions, eval_data, confidence_levels)

        # Create evaluation summary
        evaluation_results = {
            "predictions": predictions,
            "metrics": metrics,
            "data_info": {
                "forecast_horizon": forecast_horizon,
                "confidence_levels": confidence_levels,
                "n_series": len(eval_data["actuals"]),
                "evaluation_period": test_data_only
            }
        }

        self.logger.info("Model evaluation completed")
        return evaluation_results

    def _prepare_evaluation_data(self, test_data_only: bool) -> Dict[str, Any]:
        """Prepare data for evaluation."""
        self.logger.info("Preparing evaluation data...")

        # Load and preprocess data (reuse trainer's logic)
        train_data, val_data, test_data = self.trainer._prepare_data()

        # Use test data for evaluation
        eval_data = test_data if test_data_only else val_data
        actuals = self.trainer._extract_actuals(eval_data)

        return {
            "actuals": actuals,
            "data": eval_data,
            "train_data": train_data  # For baseline comparison if needed
        }

    def _generate_predictions(
        self,
        eval_data: Dict[str, Any],
        forecast_horizon: int,
        confidence_levels: list
    ) -> Dict[str, Any]:
        """Generate model predictions."""
        self.logger.info(f"Generating predictions (horizon: {forecast_horizon})...")

        predictions = self.model.predict(
            horizon=forecast_horizon,
            return_intervals=True,
            confidence_levels=confidence_levels
        )

        return predictions

    def _compute_evaluation_metrics(
        self,
        predictions: Dict[str, Any],
        eval_data: Dict[str, Any],
        confidence_levels: list
    ) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics."""
        self.logger.info("Computing evaluation metrics...")

        # Extract components
        forecasts = predictions["forecasts"]
        actuals = eval_data["actuals"]

        # Build intervals dictionary
        intervals = {}
        for alpha in confidence_levels:
            confidence = int((1 - alpha) * 100)
            lower_key = f"lower_{confidence}"
            upper_key = f"upper_{confidence}"

            if lower_key in predictions and upper_key in predictions:
                intervals[lower_key] = predictions[lower_key]
                intervals[upper_key] = predictions[upper_key]

        # Compute all metrics
        metrics = self.metrics_calculator.compute_all_metrics(
            predictions=forecasts,
            actuals=actuals,
            intervals=intervals if intervals else None,
            confidence_levels=confidence_levels
        )

        # Add coherence score if available
        if "coherence_score" in predictions:
            metrics["reconciliation_coherence"] = predictions["coherence_score"]

        return metrics

    def compare_with_baselines(
        self,
        eval_data: Dict[str, Any],
        forecast_horizon: int
    ) -> Dict[str, Dict[str, float]]:
        """Compare model performance with baseline methods."""
        self.logger.info("Comparing with baseline methods...")

        baselines = {}

        # Naive baseline (previous value)
        naive_predictions = self._create_naive_baseline(eval_data, forecast_horizon)
        naive_metrics = self._compute_evaluation_metrics(
            {"forecasts": naive_predictions}, eval_data, []
        )
        baselines["naive"] = naive_metrics

        # Seasonal naive baseline
        seasonal_predictions = self._create_seasonal_naive_baseline(
            eval_data, forecast_horizon, period=7
        )
        seasonal_metrics = self._compute_evaluation_metrics(
            {"forecasts": seasonal_predictions}, eval_data, []
        )
        baselines["seasonal_naive"] = seasonal_metrics

        return baselines

    def _create_naive_baseline(
        self,
        eval_data: Dict[str, Any],
        forecast_horizon: int
    ) -> Dict[str, np.ndarray]:
        """Create naive baseline predictions (last value repeated)."""
        naive_predictions = {}

        for series_id, actual_values in eval_data["actuals"].items():
            if len(actual_values) > 0:
                last_value = actual_values[-1]  # Use last observed value
                naive_predictions[series_id] = np.full(forecast_horizon, last_value)
            else:
                naive_predictions[series_id] = np.zeros(forecast_horizon)

        return naive_predictions

    def _create_seasonal_naive_baseline(
        self,
        eval_data: Dict[str, Any],
        forecast_horizon: int,
        period: int = 7
    ) -> Dict[str, np.ndarray]:
        """Create seasonal naive baseline predictions."""
        seasonal_predictions = {}

        for series_id, actual_values in eval_data["actuals"].items():
            if len(actual_values) >= period:
                # Use values from one period ago
                seasonal_pattern = actual_values[-period:]
                # Repeat pattern to cover forecast horizon
                n_repeats = (forecast_horizon + period - 1) // period
                extended_pattern = np.tile(seasonal_pattern, n_repeats)
                seasonal_predictions[series_id] = extended_pattern[:forecast_horizon]
            else:
                # Fallback to naive if insufficient data
                last_value = actual_values[-1] if len(actual_values) > 0 else 0
                seasonal_predictions[series_id] = np.full(forecast_horizon, last_value)

        return seasonal_predictions

    def create_visualizations(
        self,
        evaluation_results: Dict[str, Any],
        output_dir: Path,
        baseline_results: Optional[Dict[str, Dict[str, float]]] = None
    ) -> None:
        """Create visualization plots."""
        self.logger.info("Creating visualization plots...")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. Metrics comparison plot
        self._plot_metrics_comparison(
            evaluation_results["metrics"],
            baseline_results,
            output_dir / "metrics_comparison.png"
        )

        # 2. Prediction intervals plot (sample series)
        self._plot_prediction_intervals(
            evaluation_results,
            output_dir / "prediction_intervals.png"
        )

        # 3. Coherence analysis
        if "reconciliation_coherence" in evaluation_results["metrics"]:
            self._plot_coherence_analysis(
                evaluation_results,
                output_dir / "coherence_analysis.png"
            )

        # 4. Error distribution plot
        self._plot_error_distributions(
            evaluation_results,
            output_dir / "error_distributions.png"
        )

        self.logger.info(f"Plots saved to: {output_dir}")

    def _plot_metrics_comparison(
        self,
        metrics: Dict[str, float],
        baseline_results: Optional[Dict[str, Dict[str, float]]],
        output_path: Path
    ) -> None:
        """Plot comparison of metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Model Performance Metrics", fontsize=16, fontweight='bold')

        # Primary metrics
        primary_metrics = ["WRMSSE", "MASE", "sMAPE"]
        available_primary = [m for m in primary_metrics if m in metrics]

        if available_primary:
            ax = axes[0, 0]
            values = [metrics[m] for m in available_primary]
            bars = ax.bar(available_primary, values, color='skyblue', alpha=0.7)
            ax.set_title("Primary Forecast Metrics")
            ax.set_ylabel("Metric Value")

            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')

        # Coverage metrics
        coverage_metrics = [k for k in metrics.keys() if "coverage_" in k]
        if coverage_metrics:
            ax = axes[0, 1]
            coverage_values = [metrics[m] for m in coverage_metrics]
            coverage_labels = [m.replace("coverage_", "") + "%" for m in coverage_metrics]

            bars = ax.bar(coverage_labels, coverage_values, color='lightcoral', alpha=0.7)
            ax.set_title("Prediction Interval Coverage")
            ax.set_ylabel("Coverage Probability")
            ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='Target (90%)')
            ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='Target (95%)')
            ax.legend()

            # Add value labels
            for bar, value in zip(bars, coverage_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')

        # Coherence metrics
        coherence_metrics = [k for k in metrics.keys() if "coherence" in k]
        if coherence_metrics:
            ax = axes[1, 0]
            coherence_values = [metrics[m] for m in coherence_metrics]
            bars = ax.bar(coherence_metrics, coherence_values, color='lightgreen', alpha=0.7)
            ax.set_title("Hierarchical Coherence")
            ax.set_ylabel("Coherence Score")
            ax.set_ylim(0, 1)

            # Add value labels
            for bar, value in zip(bars, coherence_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')

        # Baseline comparison (if available)
        if baseline_results and available_primary:
            ax = axes[1, 1]
            methods = ["Model"] + list(baseline_results.keys())
            metric_name = available_primary[0]  # Use first available metric

            values = [metrics[metric_name]]
            for baseline_name in baseline_results.keys():
                if metric_name in baseline_results[baseline_name]:
                    values.append(baseline_results[baseline_name][metric_name])
                else:
                    values.append(np.nan)

            bars = ax.bar(methods, values, color=['gold'] + ['lightblue'] * len(baseline_results))
            ax.set_title(f"{metric_name} Comparison")
            ax.set_ylabel(metric_name)

            # Add value labels
            for bar, value in zip(bars, values):
                if not np.isnan(value):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_prediction_intervals(
        self,
        evaluation_results: Dict[str, Any],
        output_path: Path
    ) -> None:
        """Plot prediction intervals for sample series."""
        predictions = evaluation_results["predictions"]

        # Select a few representative series
        all_series = list(predictions["forecasts"].keys())
        sample_series = all_series[:min(4, len(all_series))]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        fig.suptitle("Prediction Intervals Sample", fontsize=16, fontweight='bold')

        for i, series_id in enumerate(sample_series):
            ax = axes[i]

            # Plot point forecast
            forecast = predictions["forecasts"][series_id]
            time_steps = range(len(forecast))
            ax.plot(time_steps, forecast, 'b-', label='Forecast', linewidth=2)

            # Plot prediction intervals if available
            if "lower_90" in predictions and series_id in predictions["lower_90"]:
                lower_90 = predictions["lower_90"][series_id]
                upper_90 = predictions["upper_90"][series_id]
                ax.fill_between(time_steps, lower_90, upper_90,
                               alpha=0.3, color='blue', label='90% PI')

            if "lower_95" in predictions and series_id in predictions["lower_95"]:
                lower_95 = predictions["lower_95"][series_id]
                upper_95 = predictions["upper_95"][series_id]
                ax.fill_between(time_steps, lower_95, upper_95,
                               alpha=0.2, color='blue', label='95% PI')

            ax.set_title(f"Series: {series_id[:20]}...")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_coherence_analysis(
        self,
        evaluation_results: Dict[str, Any],
        output_path: Path
    ) -> None:
        """Plot hierarchical coherence analysis."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        coherence_score = evaluation_results["metrics"].get("reconciliation_coherence", 0)

        # Create a gauge-style plot for coherence
        theta = np.linspace(0, np.pi, 100)
        radius = 1

        # Background semicircle
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), 'k-', linewidth=3)

        # Color zones
        for i, (start, end, color, label) in enumerate([
            (0, 0.33, 'red', 'Poor'),
            (0.33, 0.66, 'orange', 'Fair'),
            (0.66, 1.0, 'green', 'Good')
        ]):
            start_angle = start * np.pi
            end_angle = end * np.pi
            angles = np.linspace(start_angle, end_angle, 50)
            ax.fill_between(radius * np.cos(angles), 0, radius * np.sin(angles),
                           alpha=0.3, color=color, label=label)

        # Coherence indicator
        coherence_angle = coherence_score * np.pi
        ax.arrow(0, 0, 0.8 * radius * np.cos(coherence_angle),
                0.8 * radius * np.sin(coherence_angle),
                head_width=0.05, head_length=0.1, fc='black', ec='black')

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.text(0, -0.1, f'Coherence: {coherence_score:.3f}',
                ha='center', va='top', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_title('Hierarchical Coherence Score', fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_error_distributions(
        self,
        evaluation_results: Dict[str, Any],
        output_path: Path
    ) -> None:
        """Plot forecast error distributions."""
        # This would require actual vs predicted data
        # For now, create a placeholder plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Generate sample error data (in real implementation, compute actual errors)
        errors = np.random.normal(0, 1, 1000)  # Placeholder

        ax.hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title('Forecast Error Distribution', fontsize=16, fontweight='bold')
        ax.set_xlabel('Forecast Error')
        ax.set_ylabel('Frequency')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero Error')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def main() -> None:
    """Main evaluation function."""
    try:
        # Parse arguments
        args = parse_arguments()

        # Load configuration
        config = load_config(args.config)

        # Override data path if provided
        if args.data_path:
            config['data']['path'] = args.data_path

        # Set up environment
        setup_logging(
            level=args.log_level,
            console=not args.quiet
        )
        set_random_seeds(config.get('random_seed', 42))

        logger = logging.getLogger(__name__)
        logger.info("Starting model evaluation")
        logger.info(f"Model: {args.model_path}")

        # Initialize evaluator
        evaluator = ModelEvaluator(
            model_path=args.model_path,
            config=config,
            data_path=args.data_path
        )

        # Perform evaluation
        evaluation_results = evaluator.evaluate_model(
            forecast_horizon=args.forecast_horizon,
            confidence_levels=args.confidence_levels,
            test_data_only=args.test_data_only
        )

        # Compare with baselines if requested
        baseline_results = None
        if args.compare_baselines:
            baseline_results = evaluator.compare_with_baselines(
                evaluator._prepare_evaluation_data(args.test_data_only),
                args.forecast_horizon
            )

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save evaluation results
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {
                "metrics": evaluation_results["metrics"],
                "data_info": evaluation_results["data_info"]
            }
            if baseline_results:
                serializable_results["baseline_comparison"] = baseline_results

            json.dump(serializable_results, f, indent=2, default=str)

        logger.info(f"Results saved to: {results_file}")

        # Create performance report
        target_metrics = {
            "WRMSSE": args.target_wrmsse,
            "coverage_90": args.target_coverage_90,
            "reconciliation_coherence": args.target_coherence
        }

        report = evaluator.metrics_calculator.create_performance_report(
            evaluation_results["metrics"],
            target_metrics
        )

        # Save and print report
        report_file = output_dir / "performance_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)

        print("\n" + report)

        # Save predictions if requested
        if args.save_predictions:
            predictions_dir = output_dir / "predictions"
            predictions_dir.mkdir(exist_ok=True)

            # Save forecasts
            forecasts_df = pd.DataFrame(evaluation_results["predictions"]["forecasts"])
            forecasts_df.to_csv(predictions_dir / "forecasts.csv", index=False)

            # Save intervals if available
            for key, intervals in evaluation_results["predictions"].items():
                if key.startswith("lower_") or key.startswith("upper_"):
                    intervals_df = pd.DataFrame(intervals)
                    intervals_df.to_csv(predictions_dir / f"{key}.csv", index=False)

            logger.info(f"Predictions saved to: {predictions_dir}")

        # Create plots if requested
        if args.create_plots:
            evaluator.create_visualizations(
                evaluation_results,
                output_dir / "plots",
                baseline_results
            )

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Evaluation failed: {e}")
        if args.log_level == "DEBUG":
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()