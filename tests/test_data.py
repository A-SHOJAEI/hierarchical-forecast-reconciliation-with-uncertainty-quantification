"""Tests for data loading and preprocessing modules."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from hierarchical_forecast_reconciliation_with_uncertainty_quantification.data.loader import (
    M5DataLoader, HierarchicalDataBuilder
)
from hierarchical_forecast_reconciliation_with_uncertainty_quantification.data.preprocessing import (
    M5Preprocessor, HierarchyBuilder
)


class TestM5DataLoader:
    """Test M5 data loading functionality."""

    def test_init_valid_path(self, sample_m5_files):
        """Test initialization with valid data path."""
        loader = M5DataLoader(str(sample_m5_files))
        assert loader.data_path == sample_m5_files
        assert loader.sales_data is None
        assert loader.calendar_data is None
        assert loader.prices_data is None

    def test_init_invalid_path(self):
        """Test initialization with invalid path."""
        with pytest.raises(FileNotFoundError):
            M5DataLoader("/invalid/path")

    def test_load_data_success(self, sample_m5_files):
        """Test successful data loading."""
        loader = M5DataLoader(str(sample_m5_files))
        sales_data, calendar_data, prices_data = loader.load_data()

        assert isinstance(sales_data, pd.DataFrame)
        assert isinstance(calendar_data, pd.DataFrame)
        assert isinstance(prices_data, pd.DataFrame)

        assert not sales_data.empty
        assert not calendar_data.empty
        assert not prices_data.empty

        # Check required columns
        required_sales_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        assert all(col in sales_data.columns for col in required_sales_cols)

        required_calendar_cols = ['date', 'wm_yr_wk', 'weekday', 'd']
        assert all(col in calendar_data.columns for col in required_calendar_cols)

        required_prices_cols = ['store_id', 'item_id', 'wm_yr_wk', 'sell_price']
        assert all(col in prices_data.columns for col in required_prices_cols)

    def test_load_data_missing_files(self, temp_directory):
        """Test loading data with missing files."""
        loader = M5DataLoader(str(temp_directory))

        with pytest.raises(FileNotFoundError):
            loader.load_data()

    def test_get_hierarchy_info(self, sample_m5_files):
        """Test hierarchy information extraction."""
        loader = M5DataLoader(str(sample_m5_files))
        loader.load_data()

        hierarchy_info = loader.get_hierarchy_info()

        assert isinstance(hierarchy_info, dict)
        required_levels = ['state_id', 'store_id', 'cat_id', 'dept_id', 'item_id']
        assert all(level in hierarchy_info for level in required_levels)

        # Check that we have the expected values
        assert len(hierarchy_info['state_id']) > 0
        assert len(hierarchy_info['item_id']) > 0

    def test_prepare_time_series_data(self, sample_m5_files):
        """Test time series data preparation."""
        loader = M5DataLoader(str(sample_m5_files))
        loader.load_data()

        ts_data = loader.prepare_time_series_data(
            start_day=1,
            end_day=50,
            min_nonzero_ratio=0.0  # Include all series for testing
        )

        assert isinstance(ts_data, pd.DataFrame)
        assert not ts_data.empty

        # Check required columns
        required_cols = ['id', 'date', 'sales', 'item_id', 'store_id', 'state_id']
        assert all(col in ts_data.columns for col in required_cols)

        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(ts_data['date'])
        assert pd.api.types.is_numeric_dtype(ts_data['sales'])

    def test_add_calendar_features(self, sample_m5_files):
        """Test adding calendar features."""
        loader = M5DataLoader(str(sample_m5_files))
        loader.load_data()

        # Create sample data with 'd' column
        sample_data = pd.DataFrame({
            'd': ['d_1', 'd_2', 'd_3'],
            'sales': [10, 20, 30]
        })

        enriched_data = loader.add_calendar_features(sample_data)

        assert 'date' in enriched_data.columns
        assert 'weekday' in enriched_data.columns
        assert len(enriched_data) == len(sample_data)

    def test_add_price_features(self, sample_m5_files):
        """Test adding price features."""
        loader = M5DataLoader(str(sample_m5_files))
        loader.load_data()

        # Create sample data
        sample_data = pd.DataFrame({
            'store_id': ['CA_1', 'CA_1'],
            'item_id': ['FOODS_1_001', 'FOODS_1_001'],
            'wm_yr_wk': [202201, 202202],
            'sales': [10, 20]
        })

        enriched_data = loader.add_price_features(sample_data)

        assert 'sell_price' in enriched_data.columns
        assert len(enriched_data) >= len(sample_data)


class TestHierarchicalDataBuilder:
    """Test hierarchical data building functionality."""

    def test_init(self):
        """Test initialization."""
        levels = ["total", "state", "store", "item_store"]
        builder = HierarchicalDataBuilder(levels)
        assert builder.aggregation_levels == levels

    def test_build_hierarchy(self, sample_time_series_data):
        """Test building hierarchy."""
        levels = ["total", "state", "store", "item_store"]
        builder = HierarchicalDataBuilder(levels)

        hierarchy_data = builder.build_hierarchy(sample_time_series_data)

        assert isinstance(hierarchy_data, dict)
        assert len(hierarchy_data) == len(levels)

        # Check each level
        for level in levels:
            assert level in hierarchy_data
            assert isinstance(hierarchy_data[level], pd.DataFrame)
            assert not hierarchy_data[level].empty

        # Check total aggregation
        total_data = hierarchy_data["total"]
        assert len(total_data) > 0
        assert 'sales' in total_data.columns
        assert total_data['id'].iloc[0] == 'total'

    def test_build_aggregation_levels(self, sample_time_series_data):
        """Test individual aggregation level building."""
        levels = ["state", "item_store"]
        builder = HierarchicalDataBuilder(levels)

        # Test state level
        state_data = builder._build_aggregation_level(sample_time_series_data, "state")
        assert not state_data.empty
        assert 'state_id' in state_data.columns
        assert state_data['level'].iloc[0] == 'state'

        # Test bottom level (item_store)
        item_store_data = builder._build_aggregation_level(sample_time_series_data, "item_store")
        assert len(item_store_data) == len(sample_time_series_data)

    def test_invalid_aggregation_level(self, sample_time_series_data):
        """Test handling of invalid aggregation level."""
        builder = HierarchicalDataBuilder(["invalid_level"])

        with pytest.raises(ValueError, match="Unknown aggregation level"):
            builder._build_aggregation_level(sample_time_series_data, "invalid_level")


class TestM5Preprocessor:
    """Test M5 data preprocessing functionality."""

    def test_init(self):
        """Test initialization."""
        preprocessor = M5Preprocessor(
            scaling_method="standard",
            handle_zeros="log1p",
            outlier_threshold=3.0
        )

        assert preprocessor.scaling_method == "standard"
        assert preprocessor.handle_zeros == "log1p"
        assert preprocessor.outlier_threshold == 3.0
        assert not preprocessor.is_fitted

    def test_fit_transform(self, sample_time_series_data):
        """Test fitting and transforming data."""
        preprocessor = M5Preprocessor()
        processed_data = preprocessor.fit_transform(sample_time_series_data)

        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) >= len(sample_time_series_data)
        assert preprocessor.is_fitted

        # Check that new features are added
        assert 'sales_lag_1' in processed_data.columns
        assert 'sales_mean_7' in processed_data.columns

    def test_transform_without_fit(self, sample_time_series_data):
        """Test transforming without fitting."""
        preprocessor = M5Preprocessor()

        with pytest.raises(ValueError, match="Preprocessor not fitted"):
            preprocessor.transform(sample_time_series_data)

    def test_inverse_transform_dataframe(self, sample_time_series_data):
        """Test inverse transform on DataFrame."""
        preprocessor = M5Preprocessor(handle_zeros="log1p")
        processed_data = preprocessor.fit_transform(sample_time_series_data)

        # Inverse transform
        inverse_data = preprocessor.inverse_transform(processed_data)

        assert isinstance(inverse_data, pd.DataFrame)
        assert len(inverse_data) == len(processed_data)

    def test_inverse_transform_array(self):
        """Test inverse transform on numpy array."""
        preprocessor = M5Preprocessor(handle_zeros="log1p")

        # Create simple data for fitting
        simple_data = pd.DataFrame({
            'id': ['series_1'] * 10,
            'date': pd.date_range('2022-01-01', periods=10),
            'sales': np.random.uniform(1, 100, 10)
        })

        preprocessor.fit_transform(simple_data)

        # Test inverse transform on array
        test_array = np.array([1.0, 2.0, 3.0])
        inverse_array = preprocessor.inverse_transform(test_array, 'sales')

        assert isinstance(inverse_array, np.ndarray)
        assert len(inverse_array) == len(test_array)

    def test_create_train_test_split(self, sample_time_series_data):
        """Test train/test split creation."""
        preprocessor = M5Preprocessor()
        processed_data = preprocessor.fit_transform(sample_time_series_data)

        train_data, val_data, test_data = preprocessor.create_train_test_split(
            processed_data,
            train_days=100,
            validation_days=28,
            test_days=28
        )

        assert isinstance(train_data, pd.DataFrame)
        assert isinstance(val_data, pd.DataFrame)
        assert isinstance(test_data, pd.DataFrame)

        # Check temporal ordering
        train_max_date = train_data['date'].max()
        val_min_date = val_data['date'].min()
        val_max_date = val_data['date'].max()
        test_min_date = test_data['date'].min()

        assert train_max_date < val_min_date
        assert val_max_date < test_min_date

    def test_insufficient_data_split(self, sample_time_series_data):
        """Test split with insufficient data."""
        preprocessor = M5Preprocessor()
        processed_data = preprocessor.fit_transform(sample_time_series_data)

        with pytest.raises(ValueError, match="Insufficient data"):
            preprocessor.create_train_test_split(
                processed_data,
                train_days=1000,  # Too many days
                validation_days=28,
                test_days=28
            )

    def test_handle_missing_values(self, sample_time_series_data):
        """Test missing value handling."""
        # Introduce missing values
        data_with_missing = sample_time_series_data.copy()
        data_with_missing.loc[0:5, 'sales'] = np.nan

        preprocessor = M5Preprocessor()
        processed_data = preprocessor._handle_missing_values(data_with_missing)

        # Check that missing values are handled
        assert not processed_data['sales'].isna().any()

    def test_different_scaling_methods(self, sample_time_series_data):
        """Test different scaling methods."""
        for method in ["standard", "minmax", "none"]:
            preprocessor = M5Preprocessor(scaling_method=method)
            processed_data = preprocessor.fit_transform(sample_time_series_data)

            assert isinstance(processed_data, pd.DataFrame)
            assert preprocessor.is_fitted


class TestHierarchyBuilder:
    """Test hierarchy matrix building functionality."""

    def test_init(self):
        """Test initialization."""
        levels = ["total", "state", "store", "item_store"]
        builder = HierarchyBuilder(levels)
        assert builder.hierarchy_levels == levels

    def test_build_aggregation_matrix(self, sample_time_series_data):
        """Test aggregation matrix building."""
        # Build hierarchy data first
        levels = ["total", "state", "item_store"]
        hierarchy_builder = HierarchicalDataBuilder(levels)
        hierarchy_data = hierarchy_builder.build_hierarchy(sample_time_series_data)

        # Build aggregation matrix
        matrix_builder = HierarchyBuilder(levels)
        aggregation_matrix = matrix_builder.build_aggregation_matrix(hierarchy_data)

        assert aggregation_matrix is not None
        assert aggregation_matrix.shape[0] > 0  # Has rows
        assert aggregation_matrix.shape[1] > 0  # Has columns

        # Matrix should have correct structure for hierarchy
        n_bottom = len(hierarchy_data["item_store"]['id'].unique())
        assert aggregation_matrix.shape[1] == n_bottom

    def test_missing_bottom_level(self, sample_time_series_data):
        """Test handling missing bottom level."""
        levels = ["total", "state"]  # Missing item_store
        hierarchy_builder = HierarchicalDataBuilder(levels)
        hierarchy_data = hierarchy_builder.build_hierarchy(sample_time_series_data)

        matrix_builder = HierarchyBuilder(levels)

        with pytest.raises(ValueError, match="Bottom level"):
            matrix_builder.build_aggregation_matrix(hierarchy_data)

    def test_get_hierarchy_structure(self, sample_time_series_data):
        """Test hierarchy structure extraction."""
        levels = ["total", "state", "item_store"]
        hierarchy_builder = HierarchicalDataBuilder(levels)
        hierarchy_data = hierarchy_builder.build_hierarchy(sample_time_series_data)

        matrix_builder = HierarchyBuilder(levels)
        structure = matrix_builder.get_hierarchy_structure(hierarchy_data)

        assert isinstance(structure, dict)
        for level in levels:
            assert level in structure
            assert 'n_series' in structure[level]
            assert 'total_observations' in structure[level]
            assert structure[level]['n_series'] > 0
            assert structure[level]['total_observations'] > 0