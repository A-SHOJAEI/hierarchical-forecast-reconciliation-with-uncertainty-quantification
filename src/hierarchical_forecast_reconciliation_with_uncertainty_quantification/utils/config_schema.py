"""Configuration schema validation for the hierarchical forecasting framework.

This module provides comprehensive schema validation for configuration files,
ensuring all required parameters are present and have valid types and values.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class ConfigSchema:
    """Configuration schema validator with comprehensive validation rules."""

    @staticmethod
    def get_base_schema() -> Dict[str, Any]:
        """
        Get the base configuration schema definition.

        Returns:
            Dictionary defining the complete configuration schema with validation rules.
        """
        return {
            'data': {
                'required': True,
                'type': dict,
                'schema': {
                    'path': {'required': True, 'type': str, 'validator': 'validate_path'},
                    'train_days': {'required': True, 'type': int, 'min': 1, 'max': 10000},
                    'validation_days': {'required': True, 'type': int, 'min': 1, 'max': 5000},
                    'test_days': {'required': True, 'type': int, 'min': 1, 'max': 5000},
                    'aggregation_levels': {
                        'required': True,
                        'type': list,
                        'allowed': ['total', 'state', 'store', 'cat', 'dept', 'state_cat',
                                  'state_dept', 'store_cat', 'store_dept', 'item', 'item_store'],
                        'minlength': 1
                    },
                    'min_nonzero_ratio': {
                        'required': False,
                        'type': float,
                        'min': 0.0,
                        'max': 1.0,
                        'default': 0.1
                    },
                    'preprocessing': {
                        'required': False,
                        'type': dict,
                        'default': {},
                        'schema': {
                            'handle_missing': {
                                'required': False,
                                'type': str,
                                'allowed': ['forward_fill', 'backward_fill', 'interpolate', 'zero'],
                                'default': 'zero'
                            },
                            'outlier_detection': {
                                'required': False,
                                'type': dict,
                                'default': {},
                                'schema': {
                                    'method': {
                                        'required': False,
                                        'type': str,
                                        'allowed': ['iqr', 'zscore', 'isolation_forest', 'none'],
                                        'default': 'iqr'
                                    },
                                    'threshold': {'required': False, 'type': float, 'min': 0.1, 'max': 10.0, 'default': 3.0}
                                }
                            },
                            'scaling': {
                                'required': False,
                                'type': str,
                                'allowed': ['standard', 'minmax', 'robust', 'none'],
                                'default': 'standard'
                            }
                        }
                    }
                }
            },
            'models': {
                'required': True,
                'type': dict,
                'schema': {
                    'statistical': {
                        'required': True,
                        'type': dict,
                        'schema': {
                            'ets': {
                                'required': False,
                                'type': dict,
                                'default': {},
                                'schema': {
                                    'trend': {
                                        'required': False,
                                        'type': str,
                                        'allowed': ['add', 'mul', None],
                                        'default': 'add'
                                    },
                                    'seasonal': {
                                        'required': False,
                                        'type': str,
                                        'allowed': ['add', 'mul', None],
                                        'default': 'add'
                                    },
                                    'seasonal_periods': {'required': False, 'type': int, 'min': 1, 'max': 365, 'default': 7}
                                }
                            },
                            'arima': {
                                'required': False,
                                'type': dict,
                                'default': {},
                                'schema': {
                                    'order': {
                                        'required': False,
                                        'type': list,
                                        'minlength': 3,
                                        'maxlength': 3,
                                        'default': [1, 1, 1],
                                        'validator': 'validate_arima_order'
                                    },
                                    'seasonal_order': {
                                        'required': False,
                                        'type': list,
                                        'minlength': 4,
                                        'maxlength': 4,
                                        'default': [1, 1, 1, 7],
                                        'validator': 'validate_seasonal_arima_order'
                                    },
                                    'auto_arima': {'required': False, 'type': bool, 'default': True}
                                }
                            }
                        }
                    },
                    'deep_learning': {
                        'required': True,
                        'type': dict,
                        'schema': {
                            'tft': {
                                'required': False,
                                'type': dict,
                                'default': {},
                                'schema': {
                                    'hidden_size': {'required': False, 'type': int, 'min': 8, 'max': 1024, 'default': 64},
                                    'attention_head_size': {'required': False, 'type': int, 'min': 1, 'max': 16, 'default': 4},
                                    'dropout': {'required': False, 'type': float, 'min': 0.0, 'max': 0.9, 'default': 0.3},
                                    'learning_rate': {'required': False, 'type': float, 'min': 1e-6, 'max': 1.0, 'default': 0.001},
                                    'quantiles': {
                                        'required': False,
                                        'type': list,
                                        'default': [0.1, 0.5, 0.9],
                                        'validator': 'validate_quantiles'
                                    }
                                }
                            },
                            'nbeats': {
                                'required': False,
                                'type': dict,
                                'default': {},
                                'schema': {
                                    'num_stacks': {'required': False, 'type': int, 'min': 1, 'max': 20, 'default': 30},
                                    'num_blocks': {'required': False, 'type': int, 'min': 1, 'max': 10, 'default': 1},
                                    'num_layers': {'required': False, 'type': int, 'min': 1, 'max': 10, 'default': 4},
                                    'layer_widths': {'required': False, 'type': int, 'min': 32, 'max': 1024, 'default': 256},
                                    'learning_rate': {'required': False, 'type': float, 'min': 1e-6, 'max': 1.0, 'default': 0.001}
                                }
                            }
                        }
                    }
                }
            },
            'reconciliation': {
                'required': True,
                'type': dict,
                'schema': {
                    'method': {
                        'required': True,
                        'type': str,
                        'allowed': ['probabilistic_mint', 'ols', 'wls', 'structural']
                    },
                    'weights': {
                        'required': False,
                        'type': str,
                        'allowed': ['wls', 'ols', 'var', 'identity'],
                        'default': 'wls'
                    },
                    'regularization': {'required': False, 'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.01}
                }
            },
            'ensemble': {
                'required': False,
                'type': dict,
                'default': {},
                'schema': {
                    'weights': {
                        'required': False,
                        'type': dict,
                        'default': {'ets': 0.25, 'arima': 0.25, 'tft': 0.35, 'nbeats': 0.15},
                        'validator': 'validate_ensemble_weights'
                    },
                    'combination_method': {
                        'required': False,
                        'type': str,
                        'allowed': ['weighted_average', 'stacking', 'voting'],
                        'default': 'weighted_average'
                    }
                }
            },
            'training': {
                'required': True,
                'type': dict,
                'schema': {
                    'batch_size': {'required': True, 'type': int, 'min': 1, 'max': 1024},
                    'max_epochs': {'required': True, 'type': int, 'min': 1, 'max': 1000},
                    'early_stopping': {
                        'required': False,
                        'type': dict,
                        'default': {},
                        'schema': {
                            'patience': {'required': False, 'type': int, 'min': 1, 'max': 100, 'default': 10},
                            'min_delta': {'required': False, 'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.001}
                        }
                    },
                    'validation_strategy': {
                        'required': False,
                        'type': dict,
                        'default': {},
                        'schema': {
                            'method': {
                                'required': False,
                                'type': str,
                                'allowed': ['time_series_split', 'expanding_window', 'sliding_window'],
                                'default': 'time_series_split'
                            },
                            'n_splits': {'required': False, 'type': int, 'min': 2, 'max': 20, 'default': 5}
                        }
                    }
                }
            },
            'evaluation': {
                'required': True,
                'type': dict,
                'schema': {
                    'metrics': {
                        'required': True,
                        'type': list,
                        'allowed': ['wrmsse', 'mase', 'smape', 'rmse', 'mae', 'mape', 'crps', 'msis'],
                        'minlength': 1
                    },
                    'statistical_tests': {
                        'required': False,
                        'type': dict,
                        'default': {},
                        'schema': {
                            'diebold_mariano': {'required': False, 'type': bool, 'default': True},
                            'model_confidence_set': {'required': False, 'type': bool, 'default': True}
                        }
                    },
                    'confidence_levels': {
                        'required': False,
                        'type': list,
                        'default': [0.8, 0.9, 0.95],
                        'validator': 'validate_confidence_levels'
                    }
                }
            },
            'optimization': {
                'required': False,
                'type': dict,
                'default': {},
                'schema': {
                    'enabled': {'required': False, 'type': bool, 'default': False},
                    'n_trials': {'required': False, 'type': int, 'min': 1, 'max': 1000, 'default': 50},
                    'timeout': {'required': False, 'type': int, 'min': 60, 'max': 86400, 'default': 3600},
                    'direction': {
                        'required': False,
                        'type': str,
                        'allowed': ['minimize', 'maximize'],
                        'default': 'minimize'
                    }
                }
            },
            'logging': {
                'required': False,
                'type': dict,
                'default': {},
                'schema': {
                    'level': {
                        'required': False,
                        'type': str,
                        'allowed': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        'default': 'INFO'
                    },
                    'file': {
                        'required': False,
                        'type': str,
                        'nullable': True,
                        'default': None,
                        'validator': 'validate_log_file_path'
                    },
                    'format': {
                        'required': False,
                        'type': str,
                        'default': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    }
                }
            },
            'reproducibility': {
                'required': False,
                'type': dict,
                'default': {},
                'schema': {
                    'seed': {'required': False, 'type': int, 'min': 0, 'max': 2**32-1, 'default': 42},
                    'deterministic': {'required': False, 'type': bool, 'default': True}
                }
            }
        }

    @staticmethod
    def validate_path(field: str, value: Any) -> bool:
        """
        Validate file system path.

        Args:
            field: Field name being validated.
            value: Path value to validate.

        Returns:
            True if path is valid and accessible.

        Raises:
            ValueError: If path is invalid or inaccessible.
        """
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a string path")

        path = Path(value)

        # Check if path exists and is accessible
        if not path.exists():
            # Allow paths that don't exist yet if parent directory exists and is writable
            if not path.parent.exists():
                raise ValueError(f"{field} parent directory does not exist: {path.parent}")
            if not os.access(path.parent, os.W_OK):
                raise ValueError(f"{field} parent directory is not writable: {path.parent}")
        elif not os.access(path, os.R_OK):
            raise ValueError(f"{field} path is not readable: {path}")

        return True

    @staticmethod
    def validate_arima_order(field: str, value: Any) -> bool:
        """
        Validate ARIMA order parameter.

        Args:
            field: Field name being validated.
            value: ARIMA order tuple (p, d, q).

        Returns:
            True if order is valid.

        Raises:
            ValueError: If order is invalid.
        """
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"{field} must be a list/tuple of 3 integers (p, d, q)")

        p, d, q = value
        if not all(isinstance(x, int) and x >= 0 for x in [p, d, q]):
            raise ValueError(f"{field} values must be non-negative integers")

        if p > 10 or d > 2 or q > 10:
            raise ValueError(f"{field} values are too large (max: p=10, d=2, q=10)")

        return True

    @staticmethod
    def validate_seasonal_arima_order(field: str, value: Any) -> bool:
        """
        Validate seasonal ARIMA order parameter.

        Args:
            field: Field name being validated.
            value: Seasonal ARIMA order tuple (P, D, Q, s).

        Returns:
            True if order is valid.

        Raises:
            ValueError: If order is invalid.
        """
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError(f"{field} must be a list/tuple of 4 integers (P, D, Q, s)")

        P, D, Q, s = value
        if not all(isinstance(x, int) and x >= 0 for x in [P, D, Q]):
            raise ValueError(f"{field} first 3 values must be non-negative integers")

        if not isinstance(s, int) or s < 1:
            raise ValueError(f"{field} seasonal period must be a positive integer")

        if P > 5 or D > 2 or Q > 5:
            raise ValueError(f"{field} values are too large (max: P=5, D=2, Q=5)")

        if s > 365:
            raise ValueError(f"{field} seasonal period is too large (max: 365)")

        return True

    @staticmethod
    def validate_quantiles(field: str, value: Any) -> bool:
        """
        Validate quantile list for prediction intervals.

        Args:
            field: Field name being validated.
            value: List of quantiles.

        Returns:
            True if quantiles are valid.

        Raises:
            ValueError: If quantiles are invalid.
        """
        if not isinstance(value, list):
            raise ValueError(f"{field} must be a list of quantiles")

        if len(value) == 0:
            raise ValueError(f"{field} cannot be empty")

        for i, q in enumerate(value):
            if not isinstance(q, (int, float)):
                raise ValueError(f"{field}[{i}] must be a number")
            if not (0 < q < 1):
                raise ValueError(f"{field}[{i}] must be between 0 and 1, got {q}")

        # Check for duplicates
        if len(set(value)) != len(value):
            raise ValueError(f"{field} contains duplicate quantiles")

        # Ensure median (0.5) is included
        if 0.5 not in value:
            logger.warning(f"{field} does not include median (0.5), adding it")
            value.append(0.5)

        return True

    @staticmethod
    def validate_ensemble_weights(field: str, value: Any) -> bool:
        """
        Validate ensemble model weights.

        Args:
            field: Field name being validated.
            value: Dictionary of model weights.

        Returns:
            True if weights are valid.

        Raises:
            ValueError: If weights are invalid.
        """
        if not isinstance(value, dict):
            raise ValueError(f"{field} must be a dictionary")

        valid_models = {'ets', 'arima', 'tft', 'nbeats'}

        for model, weight in value.items():
            if model not in valid_models:
                raise ValueError(f"{field} contains unknown model: {model}")
            if not isinstance(weight, (int, float)):
                raise ValueError(f"{field}[{model}] must be a number")
            if weight < 0:
                raise ValueError(f"{field}[{model}] must be non-negative")

        total_weight = sum(value.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"{field} weights must sum to 1.0, got {total_weight}")

        return True

    @staticmethod
    def validate_confidence_levels(field: str, value: Any) -> bool:
        """
        Validate confidence levels list.

        Args:
            field: Field name being validated.
            value: List of confidence levels.

        Returns:
            True if confidence levels are valid.

        Raises:
            ValueError: If confidence levels are invalid.
        """
        if not isinstance(value, list):
            raise ValueError(f"{field} must be a list")

        for i, level in enumerate(value):
            if not isinstance(level, (int, float)):
                raise ValueError(f"{field}[{i}] must be a number")
            if not (0 < level < 1):
                raise ValueError(f"{field}[{i}] must be between 0 and 1")

        return True

    @staticmethod
    def validate_log_file_path(field: str, value: Any) -> bool:
        """
        Validate log file path.

        Args:
            field: Field name being validated.
            value: Log file path.

        Returns:
            True if path is valid.

        Raises:
            ValueError: If path is invalid.
        """
        if value is None:
            return True

        if not isinstance(value, str):
            raise ValueError(f"{field} must be a string or null")

        path = Path(value)

        # Check parent directory exists and is writable
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if not os.access(path.parent, os.W_OK):
            raise ValueError(f"{field} parent directory is not writable: {path.parent}")

        return True


class ConfigValidator:
    """Configuration validator using the defined schema."""

    def __init__(self):
        """Initialize the configuration validator."""
        self.schema = ConfigSchema.get_base_schema()
        self.logger = logging.getLogger(__name__)

    def validate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize configuration.

        Args:
            config: Configuration dictionary to validate.

        Returns:
            Validated and normalized configuration with defaults applied.

        Raises:
            ValueError: If configuration is invalid.
        """
        try:
            validated_config = self._validate_recursive(config, self.schema, "root")
            self.logger.info("Configuration validation successful")
            return validated_config

        except Exception as e:
            error_msg = f"Configuration validation failed: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg) from e

    def _validate_recursive(self, config: Dict[str, Any], schema: Dict[str, Any], path: str) -> Dict[str, Any]:
        """
        Recursively validate configuration against schema.

        Args:
            config: Configuration section to validate.
            schema: Schema definition for this section.
            path: Current validation path for error messages.

        Returns:
            Validated configuration section.
        """
        validated = {}

        # Check required fields
        for field, field_schema in schema.items():
            field_path = f"{path}.{field}"

            if field_schema.get('required', False) and field not in config:
                raise ValueError(f"Required field missing: {field_path}")

            # Apply defaults
            if field not in config and 'default' in field_schema:
                config[field] = field_schema['default']

            # Validate field if present
            if field in config:
                validated[field] = self._validate_field(
                    config[field], field_schema, field_path
                )

        # Check for unknown fields
        unknown_fields = set(config.keys()) - set(schema.keys())
        if unknown_fields:
            self.logger.warning(f"Unknown configuration fields in {path}: {unknown_fields}")

        return validated

    def _validate_field(self, value: Any, field_schema: Dict[str, Any], path: str) -> Any:
        """
        Validate a single configuration field.

        Args:
            value: Value to validate.
            field_schema: Schema definition for the field.
            path: Field path for error messages.

        Returns:
            Validated value.
        """
        # Handle nullable fields
        if value is None and field_schema.get('nullable', False):
            return None

        # Type validation
        expected_type = field_schema.get('type')
        if expected_type and not isinstance(value, expected_type):
            raise ValueError(f"{path} must be of type {expected_type.__name__}, got {type(value).__name__}")

        # Range validation for numbers
        if isinstance(value, (int, float)):
            min_val = field_schema.get('min')
            max_val = field_schema.get('max')

            if min_val is not None and value < min_val:
                raise ValueError(f"{path} must be >= {min_val}, got {value}")
            if max_val is not None and value > max_val:
                raise ValueError(f"{path} must be <= {max_val}, got {value}")

        # Length validation for lists/strings
        if isinstance(value, (list, str)):
            minlength = field_schema.get('minlength')
            maxlength = field_schema.get('maxlength')

            if minlength is not None and len(value) < minlength:
                raise ValueError(f"{path} must have length >= {minlength}")
            if maxlength is not None and len(value) > maxlength:
                raise ValueError(f"{path} must have length <= {maxlength}")

        # Allowed values validation
        allowed = field_schema.get('allowed')
        if allowed is not None:
            if isinstance(value, list):
                invalid_items = [item for item in value if item not in allowed]
                if invalid_items:
                    raise ValueError(f"{path} contains invalid items {invalid_items}, allowed: {allowed}")
            elif value not in allowed:
                raise ValueError(f"{path} must be one of {allowed}, got {value}")

        # Custom validator
        validator_name = field_schema.get('validator')
        if validator_name:
            validator_func = getattr(ConfigSchema, validator_name, None)
            if validator_func:
                validator_func(path, value)
            else:
                self.logger.warning(f"Unknown validator: {validator_name}")

        # Nested schema validation
        nested_schema = field_schema.get('schema')
        if nested_schema and isinstance(value, dict):
            return self._validate_recursive(value, nested_schema, path)

        return value