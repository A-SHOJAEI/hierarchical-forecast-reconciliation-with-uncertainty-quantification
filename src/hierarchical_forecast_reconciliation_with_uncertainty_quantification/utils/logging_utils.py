"""Advanced logging utilities for the hierarchical forecasting framework.

This module provides comprehensive logging functionality including performance
monitoring, structured logging, and context-aware error reporting.
"""

import logging
import time
import traceback
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd


class StructuredLogger:
    """Enhanced logger with structured logging capabilities."""

    def __init__(self, name: str, extra_fields: Optional[Dict[str, Any]] = None):
        """
        Initialize structured logger.

        Args:
            name: Logger name.
            extra_fields: Additional fields to include in all log messages.
        """
        self.logger = logging.getLogger(name)
        self.extra_fields = extra_fields or {}

    def _format_message(self, message: str, extra: Optional[Dict[str, Any]] = None) -> str:
        """Format message with structured fields."""
        fields = {**self.extra_fields}
        if extra:
            fields.update(extra)

        if fields:
            structured_info = " | ".join([f"{k}={v}" for k, v in fields.items()])
            return f"{message} | {structured_info}"
        return message

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message with structured fields."""
        self.logger.debug(self._format_message(message, extra))

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log info message with structured fields."""
        self.logger.info(self._format_message(message, extra))

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message with structured fields."""
        self.logger.warning(self._format_message(message, extra))

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log error message with structured fields."""
        self.logger.error(self._format_message(message, extra))

    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log critical message with structured fields."""
        self.logger.critical(self._format_message(message, extra))


class PerformanceLogger:
    """Performance monitoring and logging utility."""

    def __init__(self, logger: logging.Logger):
        """
        Initialize performance logger.

        Args:
            logger: Base logger instance.
        """
        self.logger = logger
        self.timers: Dict[str, float] = {}
        self.counters: Dict[str, int] = {}
        self.metrics: Dict[str, list] = {}

    @contextmanager
    def timer(self, operation: str, log_level: str = 'INFO'):
        """
        Context manager for timing operations.

        Args:
            operation: Name of the operation being timed.
            log_level: Logging level for the timing result.

        Yields:
            Timer context.
        """
        start_time = time.time()
        self.logger.log(getattr(logging, log_level.upper()), f"Starting operation: {operation}")

        try:
            yield
            duration = time.time() - start_time
            self.timers[operation] = duration

            # Store metrics for analysis
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(duration)

            self.logger.log(
                getattr(logging, log_level.upper()),
                f"Completed operation: {operation} in {duration:.4f} seconds"
            )

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Failed operation: {operation} after {duration:.4f} seconds - {e}")
            raise

    def count(self, event: str) -> None:
        """
        Increment event counter.

        Args:
            event: Event name to count.
        """
        self.counters[event] = self.counters.get(event, 0) + 1

    def log_data_stats(self, data: Union[pd.DataFrame, np.ndarray], name: str) -> None:
        """
        Log statistics about data structures.

        Args:
            data: Data to analyze.
            name: Name of the data for logging.
        """
        if isinstance(data, pd.DataFrame):
            stats = {
                'shape': data.shape,
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
                'null_count': data.isnull().sum().sum(),
                'dtypes': dict(data.dtypes.value_counts())
            }

            # Numeric column statistics
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                numeric_stats = data[numeric_cols].describe()
                stats['numeric_summary'] = {
                    'mean_of_means': numeric_stats.loc['mean'].mean(),
                    'std_of_stds': numeric_stats.loc['std'].mean(),
                    'min_of_mins': numeric_stats.loc['min'].min(),
                    'max_of_maxs': numeric_stats.loc['max'].max()
                }

        elif isinstance(data, np.ndarray):
            stats = {
                'shape': data.shape,
                'dtype': str(data.dtype),
                'memory_usage_mb': data.nbytes / 1024 / 1024,
                'null_count': np.isnan(data).sum() if data.dtype.kind in ['f', 'c'] else 0
            }

            if data.dtype.kind in ['i', 'f', 'c']:  # Numeric types
                stats['numeric_summary'] = {
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data))
                }

        else:
            stats = {
                'type': type(data).__name__,
                'size': len(data) if hasattr(data, '__len__') else 'unknown'
            }

        self.logger.info(f"Data statistics for {name}: {json.dumps(stats, indent=2)}")

    def log_model_performance(self, model_name: str, metrics: Dict[str, float]) -> None:
        """
        Log model performance metrics.

        Args:
            model_name: Name of the model.
            metrics: Performance metrics dictionary.
        """
        formatted_metrics = {k: f"{v:.6f}" for k, v in metrics.items()}
        self.logger.info(f"Model {model_name} performance: {json.dumps(formatted_metrics, indent=2)}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of all performance metrics.

        Returns:
            Dictionary containing performance summary.
        """
        summary = {
            'total_operations': len(self.timers),
            'total_time': sum(self.timers.values()),
            'average_operation_time': np.mean(list(self.timers.values())) if self.timers else 0,
            'slowest_operation': max(self.timers.items(), key=lambda x: x[1]) if self.timers else None,
            'fastest_operation': min(self.timers.items(), key=lambda x: x[1]) if self.timers else None,
            'event_counts': dict(self.counters)
        }

        # Add operation statistics
        if self.metrics:
            operation_stats = {}
            for op, times in self.metrics.items():
                operation_stats[op] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'avg_time': np.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'std_time': np.std(times)
                }
            summary['operation_statistics'] = operation_stats

        return summary

    def log_performance_summary(self) -> None:
        """Log complete performance summary."""
        summary = self.get_performance_summary()
        self.logger.info(f"Performance Summary:\n{json.dumps(summary, indent=2)}")


def log_function_call(
    logger: Optional[logging.Logger] = None,
    log_args: bool = False,
    log_result: bool = False,
    log_timing: bool = True,
    level: str = 'INFO'
):
    """
    Decorator to log function calls with optional argument and result logging.

    Args:
        logger: Logger instance. If None, uses function's module logger.
        log_args: Whether to log function arguments.
        log_result: Whether to log function result.
        log_timing: Whether to log execution time.
        level: Logging level for the messages.

    Returns:
        Decorated function.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = logger or logging.getLogger(func.__module__)
            func_name = f"{func.__module__}.{func.__qualname__}"

            # Log function call start
            log_msg = f"Calling {func_name}"
            if log_args:
                args_str = ", ".join([repr(arg) for arg in args])
                kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
                all_args = ", ".join(filter(None, [args_str, kwargs_str]))
                log_msg += f" with args: ({all_args})"

            start_time = time.time()
            func_logger.log(getattr(logging, level.upper()), log_msg)

            try:
                result = func(*args, **kwargs)

                # Log successful completion
                duration = time.time() - start_time
                completion_msg = f"Completed {func_name}"
                if log_timing:
                    completion_msg += f" in {duration:.4f} seconds"
                if log_result:
                    completion_msg += f" with result: {repr(result)}"

                func_logger.log(getattr(logging, level.upper()), completion_msg)
                return result

            except Exception as e:
                duration = time.time() - start_time
                error_msg = f"Failed {func_name} after {duration:.4f} seconds: {e}"
                func_logger.error(error_msg)
                func_logger.debug(f"Full traceback for {func_name}:\n{traceback.format_exc()}")
                raise

        return wrapper
    return decorator


def log_exception_context(
    logger: logging.Logger,
    context: str,
    include_traceback: bool = True,
    extra_info: Optional[Dict[str, Any]] = None
):
    """
    Decorator to log exceptions with context information.

    Args:
        logger: Logger instance.
        context: Context description for the exception.
        include_traceback: Whether to include full traceback.
        extra_info: Additional information to log.

    Returns:
        Decorated function.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"Exception in {context}: {e}"

                if extra_info:
                    error_msg += f" | Extra info: {extra_info}"

                logger.error(error_msg)

                if include_traceback:
                    logger.debug(f"Full traceback for {context}:\n{traceback.format_exc()}")

                raise
        return wrapper
    return decorator


class MLflowLogger:
    """MLflow integration for experiment logging."""

    def __init__(self, experiment_name: str, run_name: Optional[str] = None):
        """
        Initialize MLflow logger.

        Args:
            experiment_name: Name of the MLflow experiment.
            run_name: Optional name for the run.
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.logger = logging.getLogger(__name__)

        try:
            import mlflow
            self.mlflow = mlflow
            self.mlflow_available = True
        except ImportError:
            self.logger.warning("MLflow not available, experiment logging disabled")
            self.mlflow_available = False

    @contextmanager
    def mlflow_run(self):
        """Context manager for MLflow runs."""
        if not self.mlflow_available:
            yield None
            return

        try:
            self.mlflow.set_experiment(self.experiment_name)
            with self.mlflow.start_run(run_name=self.run_name) as run:
                self.logger.info(f"Started MLflow run: {run.info.run_id}")
                yield run
                self.logger.info(f"Completed MLflow run: {run.info.run_id}")
        except Exception as e:
            self.logger.error(f"MLflow logging error: {e}")
            yield None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        if not self.mlflow_available:
            return

        try:
            self.mlflow.log_params(params)
            self.logger.debug(f"Logged parameters to MLflow: {params}")
        except Exception as e:
            self.logger.warning(f"Failed to log parameters to MLflow: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow."""
        if not self.mlflow_available:
            return

        try:
            self.mlflow.log_metrics(metrics, step=step)
            self.logger.debug(f"Logged metrics to MLflow: {metrics}")
        except Exception as e:
            self.logger.warning(f"Failed to log metrics to MLflow: {e}")

    def log_artifact(self, artifact_path: Union[str, Path]) -> None:
        """Log artifact to MLflow."""
        if not self.mlflow_available:
            return

        try:
            self.mlflow.log_artifact(str(artifact_path))
            self.logger.debug(f"Logged artifact to MLflow: {artifact_path}")
        except Exception as e:
            self.logger.warning(f"Failed to log artifact to MLflow: {e}")


def setup_comprehensive_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    include_performance: bool = True,
    include_mlflow: bool = False,
    experiment_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Set up comprehensive logging system.

    Args:
        level: Logging level.
        log_file: Optional log file path.
        include_performance: Whether to include performance monitoring.
        include_mlflow: Whether to include MLflow logging.
        experiment_name: MLflow experiment name.

    Returns:
        Dictionary with logger instances and utilities.
    """
    # Base logging setup
    from .config import setup_logging
    setup_logging(level=level, log_file=log_file)

    logger = logging.getLogger("hierarchical_forecast")
    structured_logger = StructuredLogger("hierarchical_forecast", {"framework": "hierarchical_forecast"})

    loggers = {
        'base': logger,
        'structured': structured_logger
    }

    if include_performance:
        performance_logger = PerformanceLogger(logger)
        loggers['performance'] = performance_logger

    if include_mlflow and experiment_name:
        mlflow_logger = MLflowLogger(experiment_name)
        loggers['mlflow'] = mlflow_logger

    logger.info("Comprehensive logging system initialized")
    return loggers


# Example usage functions for documentation
def example_performance_logging():
    """Example of performance logging usage."""
    logger = logging.getLogger(__name__)
    perf_logger = PerformanceLogger(logger)

    with perf_logger.timer("data_loading"):
        # Simulate data loading
        time.sleep(0.1)

    perf_logger.count("model_training_started")

    # Log data statistics
    sample_data = pd.DataFrame({'sales': [1, 2, 3, 4, 5]})
    perf_logger.log_data_stats(sample_data, "training_data")

    # Log performance summary
    perf_logger.log_performance_summary()


@log_function_call(log_args=True, log_result=True)
def example_function_logging(x: int, y: int) -> int:
    """Example of function call logging."""
    return x + y


def example_structured_logging():
    """Example of structured logging usage."""
    structured_logger = StructuredLogger("test_module", {"component": "forecasting"})

    structured_logger.info(
        "Model training completed",
        extra={"model_type": "ets", "accuracy": 0.95, "duration": 120.5}
    )

    structured_logger.error(
        "Model validation failed",
        extra={"model_type": "arima", "error_code": "INSUFFICIENT_DATA"}
    )