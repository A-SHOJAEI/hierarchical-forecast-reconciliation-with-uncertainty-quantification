"""Tests for enhanced error handling and validation."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock

from hierarchical_forecast_reconciliation_with_uncertainty_quantification.data.loader import (
    M5DataLoader, HierarchicalDataBuilder
)


class TestEnhancedM5DataLoaderErrorHandling:
    """Test enhanced error handling in M5DataLoader."""

    def test_load_data_invalid_csv_extension(self, sample_m5_files):
        """Test error handling for non-CSV files."""
        # Create a non-CSV file
        txt_file = sample_m5_files / "invalid_file.txt"
        txt_file.write_text("some text content")

        loader = M5DataLoader(str(sample_m5_files))

        with pytest.raises(ValueError, match="must be CSV format"):
            loader.load_data(sales_file="invalid_file.txt")

    def test_load_data_empty_csv_file(self, sample_m5_files):
        """Test error handling for empty CSV files."""
        # Create empty CSV file
        empty_csv = sample_m5_files / "empty_sales.csv"
        empty_csv.write_text("")

        loader = M5DataLoader(str(sample_m5_files))

        with pytest.raises(ValueError, match="empty"):
            loader.load_data(sales_file="empty_sales.csv")

    def test_load_data_permission_error(self, sample_m5_files):
        """Test error handling for permission denied."""
        loader = M5DataLoader(str(sample_m5_files))

        with patch('pandas.read_csv', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError, match="Permission denied"):
                loader.load_data()

    def test_load_data_csv_parser_error(self, sample_m5_files):
        """Test error handling for CSV parsing errors."""
        loader = M5DataLoader(str(sample_m5_files))

        with patch('pandas.read_csv', side_effect=pd.errors.ParserError("Parse error")):
            with pytest.raises(pd.errors.ParserError, match="Failed to parse CSV"):
                loader.load_data()

    def test_prepare_time_series_invalid_start_day(self, sample_m5_files):
        """Test error handling for invalid start_day parameter."""
        loader = M5DataLoader(str(sample_m5_files))
        loader.load_data()

        with pytest.raises(ValueError, match="start_day must be a positive integer"):
            loader.prepare_time_series_data(start_day=0)

        with pytest.raises(ValueError, match="start_day must be a positive integer"):
            loader.prepare_time_series_data(start_day=-1)

        with pytest.raises(ValueError, match="start_day must be a positive integer"):
            loader.prepare_time_series_data(start_day="invalid")

    def test_prepare_time_series_invalid_end_day(self, sample_m5_files):
        """Test error handling for invalid end_day parameter."""
        loader = M5DataLoader(str(sample_m5_files))
        loader.load_data()

        with pytest.raises(ValueError, match="end_day must be an integer >= start_day"):
            loader.prepare_time_series_data(start_day=5, end_day=3)

    def test_prepare_time_series_invalid_nonzero_ratio(self, sample_m5_files):
        """Test error handling for invalid min_nonzero_ratio parameter."""
        loader = M5DataLoader(str(sample_m5_files))
        loader.load_data()

        with pytest.raises(ValueError, match="min_nonzero_ratio must be between"):
            loader.prepare_time_series_data(min_nonzero_ratio=-0.1)

        with pytest.raises(ValueError, match="min_nonzero_ratio must be between"):
            loader.prepare_time_series_data(min_nonzero_ratio=1.5)

    def test_prepare_time_series_no_day_columns(self, sample_m5_files):
        """Test error handling when no day columns are found."""
        loader = M5DataLoader(str(sample_m5_files))
        loader.load_data()

        # Remove day columns from sales data
        day_cols = [col for col in loader.sales_data.columns if col.startswith('d_')]
        loader.sales_data = loader.sales_data.drop(columns=day_cols)

        with pytest.raises(ValueError, match="No day columns"):
            loader.prepare_time_series_data()

    def test_prepare_time_series_start_day_exceeds_available(self, sample_m5_files):
        """Test error handling when start_day exceeds available days."""
        loader = M5DataLoader(str(sample_m5_files))
        loader.load_data()

        day_cols = [col for col in loader.sales_data.columns if col.startswith('d_')]
        max_days = len(day_cols)

        with pytest.raises(ValueError, match="start_day .* exceeds available days"):
            loader.prepare_time_series_data(start_day=max_days + 1)

    def test_prepare_time_series_no_valid_series(self, sample_m5_files):
        """Test error handling when no series meet minimum non-zero ratio."""
        loader = M5DataLoader(str(sample_m5_files))
        loader.load_data()

        # Set all sales to zero
        day_cols = [col for col in loader.sales_data.columns if col.startswith('d_')]
        loader.sales_data[day_cols] = 0

        with pytest.raises(ValueError, match="No time series meet the minimum"):
            loader.prepare_time_series_data(min_nonzero_ratio=0.5)

    def test_add_calendar_features_empty_data(self, sample_m5_files):
        """Test error handling for empty input data."""
        loader = M5DataLoader(str(sample_m5_files))
        loader.load_data()

        empty_data = pd.DataFrame()

        with pytest.raises(ValueError, match="Input data is empty"):
            loader.add_calendar_features(empty_data)

    def test_add_calendar_features_missing_d_column(self, sample_m5_files):
        """Test error handling for missing 'd' column."""
        loader = M5DataLoader(str(sample_m5_files))
        loader.load_data()

        data_without_d = pd.DataFrame({'sales': [1, 2, 3]})

        with pytest.raises(ValueError, match="must contain 'd' column"):
            loader.add_calendar_features(data_without_d)

    def test_add_calendar_features_empty_calendar_data(self, sample_m5_files):
        """Test error handling for empty calendar data."""
        loader = M5DataLoader(str(sample_m5_files))
        loader.load_data()

        # Make calendar data empty
        loader.calendar_data = pd.DataFrame()

        data_with_d = pd.DataFrame({'d': ['d_1', 'd_2'], 'sales': [1, 2]})

        with pytest.raises(ValueError, match="Calendar data is empty"):
            loader.add_calendar_features(data_with_d)


class TestEnhancedHierarchicalDataBuilderErrorHandling:
    """Test enhanced error handling in HierarchicalDataBuilder."""

    def test_init_empty_aggregation_levels(self):
        """Test error handling for empty aggregation levels."""
        with pytest.raises(ValueError, match="aggregation_levels cannot be empty"):
            HierarchicalDataBuilder([])

    def test_init_invalid_aggregation_levels(self):
        """Test error handling for invalid aggregation levels."""
        with pytest.raises(ValueError, match="Invalid aggregation levels"):
            HierarchicalDataBuilder(['invalid_level'])

    def test_build_hierarchy_empty_data(self):
        """Test error handling for empty input data."""
        builder = HierarchicalDataBuilder(['total'])
        empty_data = pd.DataFrame()

        with pytest.raises(ValueError, match="Input data is empty"):
            builder.build_hierarchy(empty_data)

    def test_build_hierarchy_missing_sales_column(self):
        """Test error handling for missing sales column."""
        builder = HierarchicalDataBuilder(['total'])
        data_without_sales = pd.DataFrame({'date': ['2020-01-01', '2020-01-02']})

        with pytest.raises(ValueError, match="Missing required base columns"):
            builder.build_hierarchy(data_without_sales)

    def test_build_hierarchy_non_numeric_sales(self):
        """Test error handling for non-numeric sales data."""
        builder = HierarchicalDataBuilder(['total'])
        data_with_string_sales = pd.DataFrame({
            'sales': ['not_a_number', 'also_not_number'],
            'date': ['2020-01-01', '2020-01-02']
        })

        with pytest.raises(ValueError, match="Sales column must contain numeric data"):
            builder.build_hierarchy(data_with_string_sales)

    def test_build_hierarchy_missing_required_columns_for_level(self):
        """Test error handling for missing required columns for specific levels."""
        builder = HierarchicalDataBuilder(['state'])
        data_without_state = pd.DataFrame({
            'sales': [100, 200],
            'date': ['2020-01-01', '2020-01-02']
        })

        result = builder.build_hierarchy(data_without_state)

        # Should fail to build state level but continue with other processing
        assert 'state' not in result

    def test_build_aggregation_level_unknown_level(self):
        """Test error handling for unknown aggregation level."""
        builder = HierarchicalDataBuilder(['total'])
        data = pd.DataFrame({'sales': [100, 200], 'date': ['2020-01-01', '2020-01-02']})

        with pytest.raises(ValueError, match="Unknown aggregation level"):
            builder._build_aggregation_level(data, 'unknown_level')

    def test_build_hierarchy_all_levels_fail(self):
        """Test error handling when all aggregation levels fail."""
        builder = HierarchicalDataBuilder(['state', 'store'])
        data_minimal = pd.DataFrame({'sales': [100, 200]})  # Missing required columns

        with pytest.raises(RuntimeError, match="Failed to build any aggregation levels"):
            builder.build_hierarchy(data_minimal)

    def test_get_required_columns(self):
        """Test the _get_required_columns helper method."""
        builder = HierarchicalDataBuilder(['state', 'total'])

        total_cols = builder._get_required_columns('total')
        assert 'sales' in total_cols

        state_cols = builder._get_required_columns('state')
        assert 'sales' in state_cols
        assert 'state_id' in state_cols


class TestConfigurationValidation:
    """Test configuration validation and error handling."""

    def test_invalid_logging_level(self):
        """Test error handling for invalid logging levels."""
        from hierarchical_forecast_reconciliation_with_uncertainty_quantification.utils.config import setup_logging

        with pytest.raises(ValueError, match="Invalid logging level"):
            setup_logging(level="INVALID_LEVEL")

    def test_yaml_parsing_error(self, temp_directory):
        """Test error handling for malformed YAML files."""
        from hierarchical_forecast_reconciliation_with_uncertainty_quantification.utils.config import load_config

        # Create malformed YAML file
        malformed_yaml = temp_directory / "malformed.yaml"
        malformed_yaml.write_text("invalid: yaml: content: [")

        with pytest.raises(Exception):  # YAML parsing error
            load_config(str(malformed_yaml))

    def test_missing_configuration_sections(self, temp_directory):
        """Test error handling for missing required configuration sections."""
        from hierarchical_forecast_reconciliation_with_uncertainty_quantification.utils.config import load_config

        # Create incomplete config file
        incomplete_yaml = temp_directory / "incomplete.yaml"
        incomplete_yaml.write_text("data:\n  some_setting: value\n")

        with pytest.raises(ValueError, match="Missing required configuration sections"):
            load_config(str(incomplete_yaml))

    def test_configuration_validation_missing_keys(self):
        """Test configuration validation with missing required keys."""
        from hierarchical_forecast_reconciliation_with_uncertainty_quantification.utils.config import validate_config

        invalid_config = {
            'data': {},  # Missing required keys
            'models': {'statistical': {}, 'deep_learning': {}},
            'training': {},  # Missing required keys
            'evaluation': {'metrics': ['mse']}
        }

        with pytest.raises(ValueError, match="Missing required"):
            validate_config(invalid_config)

    def test_configuration_validation_invalid_data_types(self):
        """Test configuration validation with invalid data types."""
        from hierarchical_forecast_reconciliation_with_uncertainty_quantification.utils.config import validate_config

        invalid_config = {
            'data': {
                'train_days': 'not_an_integer',  # Should be int
                'validation_days': 10,
                'test_days': 5
            },
            'models': {'statistical': {}, 'deep_learning': {}},
            'training': {'batch_size': 32, 'max_epochs': 100},
            'evaluation': {'metrics': ['mse']}
        }

        with pytest.raises(ValueError, match="must be a positive integer"):
            validate_config(invalid_config)