#!/usr/bin/env python3
"""
Training script for hierarchical ensemble forecasting.

This script provides a command-line interface for training the hierarchical
ensemble forecaster with configurable parameters and MLflow tracking.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hierarchical_forecast_reconciliation_with_uncertainty_quantification.utils.config import (
    load_config, setup_logging, set_random_seeds, validate_config
)
from hierarchical_forecast_reconciliation_with_uncertainty_quantification.training.trainer import (
    HierarchicalForecastTrainer
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train hierarchical ensemble forecasting model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Configuration
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

    # Training options
    parser.add_argument(
        "--optimize-hyperparameters",
        action="store_true",
        help="Perform hyperparameter optimization"
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        help="Number of optimization trials (overrides config)"
    )

    parser.add_argument(
        "--cross-validate",
        action="store_true",
        help="Perform cross-validation"
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds"
    )

    # MLflow options
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="MLflow experiment name (overrides config)"
    )

    parser.add_argument(
        "--run-name",
        type=str,
        help="MLflow run name (overrides config)"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save trained model"
    )

    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save trained model to disk"
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


def setup_environment(args: argparse.Namespace, config: dict) -> None:
    """Set up logging and random seeds."""
    # Configure logging
    log_level = args.log_level
    console_output = not args.quiet

    setup_logging(
        level=log_level,
        log_file=config.get('logging', {}).get('file'),
        console=console_output
    )

    # Set random seeds
    seed = config.get('random_seed', 42)
    set_random_seeds(seed)

    logging.info(f"Environment setup complete (seed: {seed})")


def override_config_from_args(config: dict, args: argparse.Namespace) -> dict:
    """Override configuration parameters from command line arguments."""
    # Override data path
    if args.data_path:
        config['data']['path'] = args.data_path

    # Override MLflow settings
    if args.experiment_name:
        config['training']['experiment_name'] = args.experiment_name

    if args.run_name:
        config['training']['run_name'] = args.run_name

    # Override optimization settings
    if args.n_trials:
        config['optimization']['n_trials'] = args.n_trials

    return config


def main() -> None:
    """Main training function."""
    try:
        # Parse arguments
        args = parse_arguments()

        # Load configuration
        config = load_config(args.config)
        config = override_config_from_args(config, args)

        # Validate configuration
        validate_config(config)

        # Set up environment
        setup_environment(args, config)

        logger = logging.getLogger(__name__)
        logger.info("Starting hierarchical forecasting training")
        logger.info(f"Configuration: {args.config}")

        # Initialize trainer
        trainer = HierarchicalForecastTrainer(
            config=config,
            data_path=args.data_path,
            experiment_name=args.experiment_name
        )

        # Perform cross-validation if requested
        if args.cross_validate:
            logger.info("Performing cross-validation...")
            cv_results = trainer.cross_validate(n_folds=args.cv_folds)
            logger.info(f"Cross-validation results: {cv_results}")

        # Train model
        logger.info("Training model...")
        model = trainer.train(
            optimize_hyperparameters=args.optimize_hyperparameters,
            n_trials=args.n_trials
        )

        # Save model if requested
        if args.save_model:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            model_path = output_dir / "hierarchical_ensemble_model.pkl"
            trainer.save_model(str(model_path))
            logger.info(f"Model saved to: {model_path}")

        logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        logger = logging.getLogger(__name__)
        logger.info("Training interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Training failed: {e}")
        if args.log_level == "DEBUG":
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()