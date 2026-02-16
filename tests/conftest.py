"""Pytest configuration and fixtures for testing."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import tempfile
import shutil

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide a sample configuration for testing."""
    return {
        "data": {
            "train_days": 100,
            "validation_days": 28,
            "test_days": 28,
            "min_nonzero_ratio": 0.1,
            "aggregation_levels": [
                "total", "state", "store", "cat", "item_store"
            ]
        },
        "models": {
            "statistical": {
                "ets": {
                    "seasonal_periods": 7,
                    "error": "add",
                    "trend": "add",
                    "seasonal": "add"
                },
                "arima": {
                    "seasonal_order": [1, 1, 1, 7]
                }
            },
            "deep_learning": {
                "tft": {
                    "max_epochs": 5,
                    "learning_rate": 0.03,
                    "hidden_size": 8,
                    "input_chunk_length": 14,
                    "output_chunk_length": 14
                },
                "nbeats": {
                    "max_epochs": 5,
                    "input_chunk_length": 14,
                    "output_chunk_length": 14,
                    "num_stacks": 5,
                    "layer_widths": 64
                }
            }
        },
        "ensemble": {
            "weights": {
                "ets": 0.3,
                "arima": 0.2,
                "tft": 0.3,
                "nbeats": 0.2
            }
        },
        "reconciliation": {
            "method": "probabilistic_mint",
            "weights": "ols",
            "lambda_reg": 0.01,
            "preserve_uncertainty": True
        },
        "training": {
            "experiment_name": "test_experiment",
            "batch_size": 16,
            "max_epochs": 5
        },
        "evaluation": {
            "metrics": ["WRMSSE", "MASE", "coverage_90"]
        },
        "random_seed": 42
    }


@pytest.fixture
def sample_time_series_data() -> pd.DataFrame:
    """Generate sample time series data for testing."""
    np.random.seed(42)

    # Create sample hierarchical data
    dates = pd.date_range("2022-01-01", periods=156, freq="D")

    # Generate base patterns
    trend = np.linspace(100, 120, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    noise = np.random.normal(0, 5, len(dates))

    base_sales = np.maximum(0, trend + seasonal + noise)

    # Create hierarchical structure
    data_rows = []

    for state in ["CA", "TX"]:
        for store in ["store_1", "store_2"]:
            for item in ["item_1", "item_2", "item_3"]:
                series_id = f"{state}_{store}_{item}"

                # Add some variation for each series
                variation = np.random.normal(1, 0.2, len(dates))
                series_sales = np.maximum(0, base_sales * variation)

                for i, (date, sales) in enumerate(zip(dates, series_sales)):
                    data_rows.append({
                        "id": series_id,
                        "date": date,
                        "sales": sales,
                        "state_id": state,
                        "store_id": store,
                        "item_id": item,
                        "cat_id": f"cat_{item[-1]}",
                        "dept_id": f"dept_{item[-1]}"
                    })

    return pd.DataFrame(data_rows)


@pytest.fixture
def sample_predictions() -> Dict[str, np.ndarray]:
    """Generate sample predictions for testing."""
    np.random.seed(42)

    predictions = {}

    # Create predictions for sample series
    for state in ["CA", "TX"]:
        for store in ["store_1", "store_2"]:
            for item in ["item_1", "item_2", "item_3"]:
                series_id = f"{state}_{store}_{item}"
                predictions[series_id] = np.random.uniform(50, 150, 28)

    # Add aggregated predictions
    predictions["total"] = np.random.uniform(1000, 2000, 28)
    predictions["CA"] = np.random.uniform(400, 800, 28)
    predictions["TX"] = np.random.uniform(400, 800, 28)

    return predictions


@pytest.fixture
def sample_actuals() -> Dict[str, np.ndarray]:
    """Generate sample actual values for testing."""
    np.random.seed(43)  # Different seed for actuals

    actuals = {}

    # Create actuals for sample series
    for state in ["CA", "TX"]:
        for store in ["store_1", "store_2"]:
            for item in ["item_1", "item_2", "item_3"]:
                series_id = f"{state}_{store}_{item}"
                actuals[series_id] = np.random.uniform(45, 155, 28)

    # Add aggregated actuals
    actuals["total"] = np.random.uniform(950, 2100, 28)
    actuals["CA"] = np.random.uniform(380, 820, 28)
    actuals["TX"] = np.random.uniform(380, 820, 28)

    return actuals


@pytest.fixture
def sample_intervals() -> Dict[str, Dict[str, np.ndarray]]:
    """Generate sample prediction intervals for testing."""
    np.random.seed(44)

    intervals = {
        "lower_90": {},
        "upper_90": {},
        "lower_95": {},
        "upper_95": {}
    }

    # Create intervals for sample series
    for state in ["CA", "TX"]:
        for store in ["store_1", "store_2"]:
            for item in ["item_1", "item_2", "item_3"]:
                series_id = f"{state}_{store}_{item}"

                # Base prediction
                base_pred = np.random.uniform(50, 150, 28)

                # Create intervals around base prediction
                intervals["lower_90"][series_id] = base_pred - np.random.uniform(5, 15, 28)
                intervals["upper_90"][series_id] = base_pred + np.random.uniform(5, 15, 28)
                intervals["lower_95"][series_id] = base_pred - np.random.uniform(8, 20, 28)
                intervals["upper_95"][series_id] = base_pred + np.random.uniform(8, 20, 28)

    return intervals


@pytest.fixture
def sample_aggregation_matrix() -> np.ndarray:
    """Generate sample aggregation matrix for testing."""
    # Simple 3-level hierarchy for testing
    # Total -> 2 states -> 4 stores per state -> 3 items per store
    # Bottom level: 24 series (2 states × 2 stores × 3 items)

    n_bottom = 12  # CA_store_1_item_1, CA_store_1_item_2, ..., TX_store_2_item_3
    n_total = 1 + 2 + 4 + n_bottom  # total + states + stores + bottom

    S = np.zeros((n_total, n_bottom))

    # Bottom level (identity)
    S[-n_bottom:, :] = np.eye(n_bottom)

    # Store level aggregation
    for i in range(4):  # 4 stores
        start_idx = i * 3  # 3 items per store
        end_idx = start_idx + 3
        S[3 + i, start_idx:end_idx] = 1

    # State level aggregation
    S[1, :6] = 1  # CA (first 6 items)
    S[2, 6:] = 1  # TX (last 6 items)

    # Total aggregation
    S[0, :] = 1

    return S


@pytest.fixture
def temp_directory():
    """Provide a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_m5_files(temp_directory):
    """Create sample M5 CSV files for testing."""
    # Create sample calendar data
    calendar_data = pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=156),
        'd': [f'd_{i+1}' for i in range(156)],
        'wm_yr_wk': [202201 + i//7 for i in range(156)],
        'weekday': [i % 7 + 1 for i in range(156)],
        'wday': [i % 7 + 1 for i in range(156)],
        'month': [((i//30) % 12) + 1 for i in range(156)],
        'year': [2022] * 156
    })

    # Create sample sales data
    sales_data = []
    for i in range(10):  # 10 sample items
        row = {
            'id': f'FOODS_1_001_CA_1_validation',
            'item_id': f'FOODS_1_00{i+1}',
            'dept_id': 'FOODS_1',
            'cat_id': 'FOODS',
            'store_id': 'CA_1',
            'state_id': 'CA'
        }

        # Add daily sales columns
        np.random.seed(42 + i)
        daily_sales = np.random.poisson(5, 156)
        for day in range(156):
            row[f'd_{day+1}'] = daily_sales[day]

        sales_data.append(row)

    sales_df = pd.DataFrame(sales_data)

    # Create sample prices data
    prices_data = []
    for i in range(10):
        for week in range(156//7):
            prices_data.append({
                'store_id': 'CA_1',
                'item_id': f'FOODS_1_00{i+1}',
                'wm_yr_wk': 202201 + week,
                'sell_price': np.random.uniform(1.0, 5.0)
            })

    prices_df = pd.DataFrame(prices_data)

    # Save files
    calendar_data.to_csv(temp_directory / 'calendar.csv', index=False)
    sales_df.to_csv(temp_directory / 'sales_train_evaluation.csv', index=False)
    prices_df.to_csv(temp_directory / 'sell_prices.csv', index=False)

    return temp_directory


class MockMLflowRun:
    """Mock MLflow run for testing."""

    def __init__(self):
        self.active = False
        self.logged_params = {}
        self.logged_metrics = {}
        self.logged_artifacts = []

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.active = False


@pytest.fixture
def mock_mlflow(monkeypatch):
    """Mock MLflow functions for testing."""
    mock_run = MockMLflowRun()

    def mock_start_run(*args, **kwargs):
        return mock_run

    def mock_log_params(params):
        mock_run.logged_params.update(params)

    def mock_log_metrics(metrics):
        mock_run.logged_metrics.update(metrics)

    def mock_log_artifact(artifact_path):
        mock_run.logged_artifacts.append(artifact_path)

    def mock_set_experiment(name):
        pass

    def mock_set_tracking_uri(uri):
        pass

    # Apply mocks
    import mlflow
    monkeypatch.setattr(mlflow, "start_run", mock_start_run)
    monkeypatch.setattr(mlflow, "log_params", mock_log_params)
    monkeypatch.setattr(mlflow, "log_metrics", mock_log_metrics)
    monkeypatch.setattr(mlflow, "log_artifact", mock_log_artifact)
    monkeypatch.setattr(mlflow, "set_experiment", mock_set_experiment)
    monkeypatch.setattr(mlflow, "set_tracking_uri", mock_set_tracking_uri)

    return mock_run