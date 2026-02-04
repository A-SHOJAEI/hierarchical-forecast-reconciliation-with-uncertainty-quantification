"""Runtime type validation utilities for the hierarchical forecasting framework.

This module provides decorators and functions for runtime type checking
and validation of function parameters and return values.
"""

import functools
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_type_hints
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class TypeValidationError(TypeError):
    """Custom exception for type validation errors."""
    pass


def validate_type(value: Any, expected_type: Type, param_name: str) -> None:
    """
    Validate that a value matches the expected type.

    Args:
        value: Value to validate.
        expected_type: Expected type or type annotation.
        param_name: Parameter name for error messages.

    Raises:
        TypeValidationError: If type validation fails.
    """
    try:
        # Handle Union types
        if hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
            if any(isinstance(value, arg) for arg in expected_type.__args__ if arg is not type(None)):
                return
            if type(None) in expected_type.__args__ and value is None:
                return
            raise TypeValidationError(
                f"Parameter '{param_name}' must be one of {expected_type.__args__}, got {type(value)}"
            )

        # Handle Optional types (Union[Type, None])
        if hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
            args = expected_type.__args__
            if len(args) == 2 and type(None) in args:
                if value is None:
                    return
                non_none_type = args[0] if args[1] is type(None) else args[1]
                expected_type = non_none_type

        # Handle List types
        if hasattr(expected_type, '__origin__') and expected_type.__origin__ is list:
            if not isinstance(value, list):
                raise TypeValidationError(f"Parameter '{param_name}' must be a list, got {type(value)}")

            # Check element types if specified
            if hasattr(expected_type, '__args__') and expected_type.__args__:
                element_type = expected_type.__args__[0]
                for i, item in enumerate(value):
                    if not isinstance(item, element_type):
                        raise TypeValidationError(
                            f"Parameter '{param_name}[{i}]' must be {element_type}, got {type(item)}"
                        )
            return

        # Handle Dict types
        if hasattr(expected_type, '__origin__') and expected_type.__origin__ is dict:
            if not isinstance(value, dict):
                raise TypeValidationError(f"Parameter '{param_name}' must be a dict, got {type(value)}")

            # Check key and value types if specified
            if hasattr(expected_type, '__args__') and len(expected_type.__args__) == 2:
                key_type, value_type = expected_type.__args__
                for k, v in value.items():
                    if not isinstance(k, key_type):
                        raise TypeValidationError(
                            f"Parameter '{param_name}' key '{k}' must be {key_type}, got {type(k)}"
                        )
                    if not isinstance(v, value_type):
                        raise TypeValidationError(
                            f"Parameter '{param_name}[{k}]' must be {value_type}, got {type(v)}"
                        )
            return

        # Standard type checking
        if not isinstance(value, expected_type):
            raise TypeValidationError(
                f"Parameter '{param_name}' must be {expected_type}, got {type(value)}"
            )

    except Exception as e:
        if isinstance(e, TypeValidationError):
            raise
        # For complex type annotations that we can't validate, log warning and continue
        logger.warning(f"Could not validate type for parameter '{param_name}': {e}")


def validate_numeric_range(
    value: Union[int, float],
    param_name: str,
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True
) -> None:
    """
    Validate that a numeric value is within the specified range.

    Args:
        value: Numeric value to validate.
        param_name: Parameter name for error messages.
        min_val: Minimum allowed value (optional).
        max_val: Maximum allowed value (optional).
        min_inclusive: Whether minimum is inclusive.
        max_inclusive: Whether maximum is inclusive.

    Raises:
        ValueError: If value is outside the valid range.
    """
    if min_val is not None:
        if min_inclusive and value < min_val:
            raise ValueError(f"Parameter '{param_name}' must be >= {min_val}, got {value}")
        elif not min_inclusive and value <= min_val:
            raise ValueError(f"Parameter '{param_name}' must be > {min_val}, got {value}")

    if max_val is not None:
        if max_inclusive and value > max_val:
            raise ValueError(f"Parameter '{param_name}' must be <= {max_val}, got {value}")
        elif not max_inclusive and value >= max_val:
            raise ValueError(f"Parameter '{param_name}' must be < {max_val}, got {value}")


def validate_dataframe_structure(
    df: pd.DataFrame,
    param_name: str,
    required_columns: Optional[List[str]] = None,
    min_rows: Optional[int] = None,
    max_rows: Optional[int] = None,
    numeric_columns: Optional[List[str]] = None
) -> None:
    """
    Validate DataFrame structure and content.

    Args:
        df: DataFrame to validate.
        param_name: Parameter name for error messages.
        required_columns: List of required column names.
        min_rows: Minimum number of rows.
        max_rows: Maximum number of rows.
        numeric_columns: Columns that must be numeric.

    Raises:
        ValueError: If DataFrame validation fails.
    """
    if df.empty:
        raise ValueError(f"Parameter '{param_name}' cannot be an empty DataFrame")

    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Parameter '{param_name}' missing required columns: {missing_cols}"
            )

    if min_rows is not None and len(df) < min_rows:
        raise ValueError(
            f"Parameter '{param_name}' must have at least {min_rows} rows, got {len(df)}"
        )

    if max_rows is not None and len(df) > max_rows:
        raise ValueError(
            f"Parameter '{param_name}' must have at most {max_rows} rows, got {len(df)}"
        )

    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(
                    f"Parameter '{param_name}' column '{col}' must be numeric, got {df[col].dtype}"
                )


def validate_array_structure(
    arr: np.ndarray,
    param_name: str,
    expected_shape: Optional[tuple] = None,
    min_dims: Optional[int] = None,
    max_dims: Optional[int] = None,
    dtype: Optional[np.dtype] = None
) -> None:
    """
    Validate NumPy array structure.

    Args:
        arr: Array to validate.
        param_name: Parameter name for error messages.
        expected_shape: Expected exact shape.
        min_dims: Minimum number of dimensions.
        max_dims: Maximum number of dimensions.
        dtype: Expected data type.

    Raises:
        ValueError: If array validation fails.
    """
    if arr.size == 0:
        raise ValueError(f"Parameter '{param_name}' cannot be an empty array")

    if expected_shape is not None and arr.shape != expected_shape:
        raise ValueError(
            f"Parameter '{param_name}' must have shape {expected_shape}, got {arr.shape}"
        )

    if min_dims is not None and arr.ndim < min_dims:
        raise ValueError(
            f"Parameter '{param_name}' must have at least {min_dims} dimensions, got {arr.ndim}"
        )

    if max_dims is not None and arr.ndim > max_dims:
        raise ValueError(
            f"Parameter '{param_name}' must have at most {max_dims} dimensions, got {arr.ndim}"
        )

    if dtype is not None and arr.dtype != dtype:
        raise ValueError(
            f"Parameter '{param_name}' must have dtype {dtype}, got {arr.dtype}"
        )


def typed(
    validate_inputs: bool = True,
    validate_outputs: bool = False,
    strict: bool = True
):
    """
    Decorator for runtime type validation of function parameters and return values.

    Args:
        validate_inputs: Whether to validate input parameters.
        validate_outputs: Whether to validate return values.
        strict: Whether to raise exceptions for type mismatches (vs. warnings).

    Returns:
        Decorated function with type validation.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if validate_inputs:
                try:
                    # Get type hints
                    type_hints = get_type_hints(func)

                    # Get function signature
                    sig = inspect.signature(func)

                    # Bind arguments to parameters
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()

                    # Validate each parameter
                    for param_name, value in bound_args.arguments.items():
                        if param_name in type_hints:
                            expected_type = type_hints[param_name]
                            try:
                                validate_type(value, expected_type, param_name)
                            except TypeValidationError as e:
                                if strict:
                                    raise
                                else:
                                    logger.warning(f"Type validation warning in {func.__name__}: {e}")

                except Exception as e:
                    if strict:
                        raise TypeValidationError(f"Type validation failed for {func.__name__}: {e}")
                    else:
                        logger.warning(f"Type validation error in {func.__name__}: {e}")

            # Call the original function
            result = func(*args, **kwargs)

            if validate_outputs:
                try:
                    type_hints = get_type_hints(func)
                    if 'return' in type_hints:
                        expected_return_type = type_hints['return']
                        try:
                            validate_type(result, expected_return_type, 'return_value')
                        except TypeValidationError as e:
                            if strict:
                                raise
                            else:
                                logger.warning(f"Return type validation warning in {func.__name__}: {e}")

                except Exception as e:
                    if strict:
                        raise TypeValidationError(f"Return type validation failed for {func.__name__}: {e}")
                    else:
                        logger.warning(f"Return type validation error in {func.__name__}: {e}")

            return result

        return wrapper
    return decorator


def validate_forecasting_inputs(
    data: pd.DataFrame,
    target_col: str = "sales",
    horizon: Optional[int] = None,
    confidence_levels: Optional[List[float]] = None
) -> None:
    """
    Specialized validation for forecasting function inputs.

    Args:
        data: Training or prediction data.
        target_col: Target column name.
        horizon: Forecast horizon (optional).
        confidence_levels: Confidence levels for intervals (optional).

    Raises:
        ValueError: If validation fails.
    """
    # Validate DataFrame
    validate_dataframe_structure(
        data, 'data',
        required_columns=[target_col],
        min_rows=1,
        numeric_columns=[target_col]
    )

    # Validate target column
    if data[target_col].isna().all():
        raise ValueError(f"Target column '{target_col}' contains only missing values")

    # Validate horizon
    if horizon is not None:
        validate_numeric_range(horizon, 'horizon', min_val=1, max_val=1000)

    # Validate confidence levels
    if confidence_levels is not None:
        if not isinstance(confidence_levels, list):
            raise ValueError("confidence_levels must be a list")

        for i, level in enumerate(confidence_levels):
            validate_numeric_range(
                level, f'confidence_levels[{i}]',
                min_val=0.0, max_val=1.0,
                min_inclusive=False, max_inclusive=False
            )


class ValidationMixin:
    """Mixin class providing validation methods for forecasting models."""

    def _validate_fit_inputs(
        self,
        data: pd.DataFrame,
        target_col: str = "sales"
    ) -> None:
        """Validate inputs for fit method."""
        validate_forecasting_inputs(data, target_col)

        # Additional checks for fitting
        if len(data) < 10:
            logger.warning(f"Very small training dataset ({len(data)} rows), may lead to poor model performance")

    def _validate_predict_inputs(
        self,
        horizon: int,
        confidence_levels: Optional[List[float]] = None
    ) -> None:
        """Validate inputs for predict method."""
        if not hasattr(self, 'is_fitted') or not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        validate_numeric_range(horizon, 'horizon', min_val=1, max_val=1000)

        if confidence_levels is not None:
            if not isinstance(confidence_levels, list):
                raise ValueError("confidence_levels must be a list")

            for i, level in enumerate(confidence_levels):
                validate_numeric_range(
                    level, f'confidence_levels[{i}]',
                    min_val=0.0, max_val=1.0,
                    min_inclusive=False, max_inclusive=False
                )

    def _validate_prediction_outputs(
        self,
        predictions: Dict[str, np.ndarray],
        expected_shape: tuple
    ) -> None:
        """Validate prediction outputs structure."""
        if not isinstance(predictions, dict):
            raise ValueError("Predictions must be returned as a dictionary")

        if 'predictions' not in predictions:
            raise ValueError("Predictions dictionary must contain 'predictions' key")

        pred_array = predictions['predictions']
        if not isinstance(pred_array, np.ndarray):
            raise ValueError("Predictions must be a NumPy array")

        if pred_array.shape != expected_shape:
            raise ValueError(f"Predictions shape {pred_array.shape} does not match expected {expected_shape}")

        # Check for invalid values
        if np.any(np.isnan(pred_array)):
            logger.warning("Predictions contain NaN values")

        if np.any(np.isinf(pred_array)):
            raise ValueError("Predictions contain infinite values")


# Example usage
if __name__ == "__main__":
    # Example of using the typed decorator
    @typed(validate_inputs=True, validate_outputs=True)
    def example_function(
        data: pd.DataFrame,
        horizon: int,
        confidence_levels: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """Example function with type validation."""
        return {
            'predictions': np.random.randn(len(data), horizon)
        }

    # Example of manual validation
    sample_data = pd.DataFrame({'sales': [1, 2, 3, 4, 5]})
    validate_forecasting_inputs(sample_data, 'sales', horizon=10, confidence_levels=[0.8, 0.95])