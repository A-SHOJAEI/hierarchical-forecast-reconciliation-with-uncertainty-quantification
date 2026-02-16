#!/usr/bin/env python3
"""
Prediction script for hierarchical forecast reconciliation.

This script loads a trained model and generates predictions with uncertainty
quantification for hierarchical time series forecasting.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from hierarchical_forecast_reconciliation_with_uncertainty_quantification.data.loader import (
    M5DataLoader,
    HierarchicalDataBuilder,
)
from hierarchical_forecast_reconciliation_with_uncertainty_quantification.data.preprocessing import (
    M5Preprocessor,
)
from hierarchical_forecast_reconciliation_with_uncertainty_quantification.models.model import (
    HierarchicalEnsembleForecaster,
    StatisticalForecaster,
)
from hierarchical_forecast_reconciliation_with_uncertainty_quantification.utils.config import (
    load_config,
)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the prediction script."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_trained_model(
    model_path: Optional[Path],
    config: Dict,
) -> HierarchicalEnsembleForecaster:
    """
    Load a trained model from disk or train a new model.

    Args:
        model_path: Path to saved model (currently not implemented - trains new model).
        config: Configuration dictionary.

    Returns:
        Trained HierarchicalEnsembleForecaster.
    """
    logger = logging.getLogger(__name__)

    if model_path and model_path.exists():
        logger.info(f"Loading model from {model_path}...")
        # Note: Model serialization not implemented, training new model
        logger.warning("Model loading not implemented, training new model instead")

    # Load and prepare data
    logger.info("Loading and preprocessing data...")
    data_loader = M5DataLoader(config=config["data"])
    data = data_loader.load()

    preprocessor = M5Preprocessor(config=config["data"])
    train_data, val_data, test_data = preprocessor.split_data(data)

    hierarchy_builder = HierarchicalDataBuilder(config=config["data"])
    hierarchy_data = hierarchy_builder.build_hierarchy(train_data)

    # Train model
    logger.info("Training ensemble model...")
    model = HierarchicalEnsembleForecaster(
        statistical_configs=config["models"]["statistical"],
        deep_learning_configs=config["models"].get("deep_learning", {}),
        ensemble_weights=config["ensemble"]["weights"],
        reconciler_config=config["reconciliation"],
    )

    model.fit(
        data=hierarchy_data["hierarchical_data"],
        aggregation_matrix=hierarchy_data["aggregation_matrix"],
        target_col="sales",
    )

    return model


def generate_predictions(
    model: HierarchicalEnsembleForecaster,
    horizon: int,
    confidence_levels: List[float],
) -> Dict:
    """
    Generate predictions with uncertainty quantification.

    Args:
        model: Trained forecaster.
        horizon: Forecast horizon (number of periods).
        confidence_levels: List of confidence levels (e.g., [0.1, 0.05] for 90% and 95%).

    Returns:
        Dictionary containing forecasts and prediction intervals.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Generating {horizon}-step predictions...")

    predictions = model.predict(
        horizon=horizon,
        return_intervals=True,
        confidence_levels=confidence_levels,
    )

    return predictions


def format_predictions_for_display(predictions: Dict, top_n: int = 10) -> pd.DataFrame:
    """
    Format predictions for human-readable display.

    Args:
        predictions: Raw prediction dictionary.
        top_n: Number of series to display.

    Returns:
        DataFrame with formatted predictions.
    """
    forecasts = predictions["forecasts"]
    series_ids = sorted(forecasts.keys())[:top_n]

    rows = []
    for series_id in series_ids:
        forecast = forecasts[series_id]
        row = {
            "series_id": series_id,
            "horizon_1": forecast[0] if len(forecast) > 0 else None,
            "horizon_7": forecast[6] if len(forecast) > 6 else None,
            "horizon_14": forecast[13] if len(forecast) > 13 else None,
            "mean": np.mean(forecast),
        }

        # Add intervals if available
        if "lower_90" in predictions:
            lower_90 = predictions["lower_90"].get(series_id)
            upper_90 = predictions["upper_90"].get(series_id)
            if lower_90 is not None and upper_90 is not None:
                row["interval_90_width"] = np.mean(upper_90 - lower_90)

        rows.append(row)

    return pd.DataFrame(rows)


def save_predictions(predictions: Dict, output_path: Path) -> None:
    """
    Save predictions to JSON file.

    Args:
        predictions: Prediction dictionary.
        output_path: Output file path.
    """
    logger = logging.getLogger(__name__)

    # Convert numpy arrays to lists for JSON serialization
    serializable_predictions = {}

    for key, value in predictions.items():
        if isinstance(value, dict):
            serializable_predictions[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        elif isinstance(value, np.ndarray):
            serializable_predictions[key] = value.tolist()
        else:
            serializable_predictions[key] = value

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(serializable_predictions, f, indent=2)

    logger.info(f"Predictions saved to {output_path}")


def main():
    """Main prediction pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate predictions with hierarchical forecast reconciliation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 28-day forecast with default config
  python scripts/predict.py --horizon 28

  # Use custom config and save predictions
  python scripts/predict.py --config configs/default.yaml --horizon 14 --output predictions.json

  # Load pre-trained model (not yet implemented)
  python scripts/predict.py --model-path outputs/best_model.pkl --horizon 7
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to pre-trained model file (optional)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=28,
        help="Forecast horizon in days (default: 28)",
    )
    parser.add_argument(
        "--confidence-levels",
        type=float,
        nargs="+",
        default=[0.1, 0.05],
        help="Confidence levels for prediction intervals (default: 0.1 0.05 for 90%% and 95%%)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/predictions.json",
        help="Output path for predictions JSON (default: outputs/predictions.json)",
    )
    parser.add_argument(
        "--display-top-n",
        type=int,
        default=10,
        help="Number of top series to display (default: 10)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Load or train model
        model_path = Path(args.model_path) if args.model_path else None
        model = load_trained_model(model_path, config)

        # Generate predictions
        predictions = generate_predictions(
            model=model,
            horizon=args.horizon,
            confidence_levels=args.confidence_levels,
        )

        # Display summary
        logger.info(f"\nPrediction Summary:")
        logger.info(f"  Series forecasted: {len(predictions['forecasts'])}")
        logger.info(f"  Forecast horizon: {args.horizon}")
        logger.info(f"  Coherence score: {predictions.get('coherence_score', 0.0):.4f}")

        # Display sample predictions
        display_df = format_predictions_for_display(predictions, args.display_top_n)
        print("\nSample Predictions (first {} series):".format(args.display_top_n))
        print(display_df.to_string(index=False))

        # Save predictions
        output_path = Path(args.output)
        save_predictions(predictions, output_path)

        logger.info("\nPrediction completed successfully!")

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
