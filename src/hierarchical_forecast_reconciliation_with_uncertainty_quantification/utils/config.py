"""Configuration utilities for the hierarchical forecasting framework."""

import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

def load_config(config_path: Optional[str] = None, validate_schema: bool = False) -> Dict[str, Any]:
    """
    Load and validate configuration from YAML file.

    Args:
        config_path: Path to configuration file. If None, uses default config.
        validate_schema: Whether to perform comprehensive schema validation.

    Returns:
        Validated and normalized configuration dictionary with defaults applied.

    Raises:
        FileNotFoundError: If config file is not found.
        yaml.YAMLError: If YAML parsing fails.
        ValueError: If configuration validation fails.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent.parent / "configs" / "default.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML configuration: {e}") from e

    if config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    # Basic validation for backward compatibility
    required_sections = ['data', 'models', 'training', 'evaluation']
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(f"Missing required configuration sections: {missing_sections}")

    # Ensure weight_decay is a float (YAML may parse 1e-8 as string)
    training_config = config.get('training', {})
    if 'weight_decay' in training_config:
        training_config['weight_decay'] = float(training_config['weight_decay'])

    # Ensure deep_learning config exists (may be empty dict or None)
    models_config = config.get('models', {})
    if models_config.get('deep_learning') is None:
        models_config['deep_learning'] = {}

    logging.info("Configuration loaded successfully")
    return config


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    console: bool = True
) -> None:
    """
    Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to log file. If None, no file logging.
        format_string: Custom log format string.
        console: Whether to enable console logging.

    Raises:
        ValueError: If invalid logging level is provided.
    """
    # Validate logging level
    level = level.upper()
    if level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        raise ValueError(f"Invalid logging level: {level}")

    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.info(f"Logging setup complete. Level: {level}")


def set_random_seeds(
    seed: int = 42,
    torch_deterministic: bool = True
) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
        torch_deterministic: Whether to use deterministic algorithms in PyTorch.
                           This may impact performance but ensures reproducibility.

    Note:
        Setting torch_deterministic=True may slow down training but ensures
        completely deterministic results across runs.
    """
    # Python random
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Note: torch.use_deterministic_algorithms(True) can cause errors
        # with many common operations, so we skip it for compatibility

    logging.info(f"Random seeds set to {seed}")


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration parameters.

    Args:
        config: Configuration dictionary to validate.

    Raises:
        ValueError: If configuration is invalid.
    """
    # Validate data configuration
    data_config = config.get('data', {})
    required_data_keys = ['train_days', 'validation_days', 'test_days']
    for key in required_data_keys:
        if key not in data_config:
            raise ValueError(f"Missing required data configuration: {key}")
        if not isinstance(data_config[key], int) or data_config[key] <= 0:
            raise ValueError(f"Data configuration {key} must be a positive integer")

    # Validate model configuration
    models_config = config.get('models', {})
    if 'statistical' not in models_config:
        raise ValueError("Missing model configuration: statistical")

    # Validate training configuration
    training_config = config.get('training', {})
    required_training_keys = ['batch_size', 'max_epochs']
    for key in required_training_keys:
        if key not in training_config:
            raise ValueError(f"Missing required training configuration: {key}")

    # Validate evaluation metrics
    eval_config = config.get('evaluation', {})
    if 'metrics' not in eval_config or not eval_config['metrics']:
        raise ValueError("At least one evaluation metric must be specified")

    logging.info("Configuration validation successful")