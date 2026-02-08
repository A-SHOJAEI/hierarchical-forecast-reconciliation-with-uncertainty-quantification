"""Data loading utilities for M5 forecasting competition dataset."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class M5DataLoader:
    """
    Loads and manages M5 forecasting competition data.

    The M5 dataset contains hierarchical sales data from Walmart with multiple
    aggregation levels including stores, categories, departments, and items.
    """

    def __init__(self, data_path: str) -> None:
        """
        Initialize M5 data loader.

        Args:
            data_path: Path to directory containing M5 data files.

        Raises:
            FileNotFoundError: If data directory or required files are not found.
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")

        self.logger = logging.getLogger(__name__)
        self.sales_data: Optional[pd.DataFrame] = None
        self.calendar_data: Optional[pd.DataFrame] = None
        self.prices_data: Optional[pd.DataFrame] = None
        self._label_encoders: Dict[str, LabelEncoder] = {}

    def load_data(
        self,
        calendar_file: str = "calendar.csv",
        prices_file: str = "sell_prices.csv",
        sales_file: str = "sales_train_evaluation.csv"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all M5 data files with comprehensive error handling.

        Args:
            calendar_file: Filename for calendar data. Must be a valid CSV file.
            prices_file: Filename for prices data. Must be a valid CSV file.
            sales_file: Filename for sales data. Must be a valid CSV file.

        Returns:
            Tuple of (sales_data, calendar_data, prices_data) as pandas DataFrames.

        Raises:
            FileNotFoundError: If any data file is not found.
            PermissionError: If insufficient permissions to read files.
            pd.errors.EmptyDataError: If any CSV file is empty.
            pd.errors.ParserError: If CSV parsing fails.
            ValueError: If loaded data has invalid structure.
        """
        self.logger.info("Loading M5 competition data...")

        try:
            # Load sales data
            sales_path = self.data_path / sales_file
            if not sales_path.exists():
                raise FileNotFoundError(f"Sales file not found: {sales_path}")
            if not sales_path.suffix.lower() == '.csv':
                raise ValueError(f"Sales file must be CSV format, got: {sales_path.suffix}")

            self.sales_data = pd.read_csv(sales_path)
            if self.sales_data.empty:
                raise ValueError(f"Sales data file is empty: {sales_path}")
            self.logger.info(f"Loaded sales data: {self.sales_data.shape}")

            # Load calendar data
            calendar_path = self.data_path / calendar_file
            if not calendar_path.exists():
                raise FileNotFoundError(f"Calendar file not found: {calendar_path}")
            if not calendar_path.suffix.lower() == '.csv':
                raise ValueError(f"Calendar file must be CSV format, got: {calendar_path.suffix}")

            self.calendar_data = pd.read_csv(calendar_path)
            if self.calendar_data.empty:
                raise ValueError(f"Calendar data file is empty: {calendar_path}")
            self.logger.info(f"Loaded calendar data: {self.calendar_data.shape}")

            # Load prices data
            prices_path = self.data_path / prices_file
            if not prices_path.exists():
                raise FileNotFoundError(f"Prices file not found: {prices_path}")
            if not prices_path.suffix.lower() == '.csv':
                raise ValueError(f"Prices file must be CSV format, got: {prices_path.suffix}")

            self.prices_data = pd.read_csv(prices_path)
            if self.prices_data.empty:
                raise ValueError(f"Prices data file is empty: {prices_path}")
            self.logger.info(f"Loaded prices data: {self.prices_data.shape}")

            self._validate_data()
            return self.sales_data, self.calendar_data, self.prices_data

        except PermissionError as e:
            error_msg = f"Permission denied accessing data files: {e}"
            self.logger.error(error_msg)
            raise PermissionError(error_msg) from e
        except pd.errors.EmptyDataError as e:
            error_msg = f"Empty CSV file encountered: {e}"
            self.logger.error(error_msg)
            raise pd.errors.EmptyDataError(error_msg) from e
        except pd.errors.ParserError as e:
            error_msg = f"Failed to parse CSV file: {e}"
            self.logger.error(error_msg)
            raise pd.errors.ParserError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error loading data: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _validate_data(self) -> None:
        """Validate loaded data for consistency and completeness."""
        if self.sales_data is None or self.calendar_data is None or self.prices_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Check sales data structure
        required_sales_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        missing_sales_cols = [col for col in required_sales_cols if col not in self.sales_data.columns]
        if missing_sales_cols:
            raise ValueError(f"Missing required sales columns: {missing_sales_cols}")

        # Check calendar data structure
        required_calendar_cols = ['date', 'wm_yr_wk', 'weekday', 'd']
        missing_calendar_cols = [col for col in required_calendar_cols if col not in self.calendar_data.columns]
        if missing_calendar_cols:
            raise ValueError(f"Missing required calendar columns: {missing_calendar_cols}")

        # Check prices data structure
        required_prices_cols = ['store_id', 'item_id', 'wm_yr_wk', 'sell_price']
        missing_prices_cols = [col for col in required_prices_cols if col not in self.prices_data.columns]
        if missing_prices_cols:
            raise ValueError(f"Missing required prices columns: {missing_prices_cols}")

        self.logger.info("Data validation successful")

    def get_hierarchy_info(self) -> Dict[str, List[str]]:
        """
        Extract hierarchy information from sales data.

        Returns:
            Dictionary mapping hierarchy levels to their unique values.
        """
        if self.sales_data is None:
            raise ValueError("Sales data not loaded. Call load_data() first.")

        hierarchy_info = {
            'state_id': sorted(self.sales_data['state_id'].unique()),
            'store_id': sorted(self.sales_data['store_id'].unique()),
            'cat_id': sorted(self.sales_data['cat_id'].unique()),
            'dept_id': sorted(self.sales_data['dept_id'].unique()),
            'item_id': sorted(self.sales_data['item_id'].unique()),
        }

        self.logger.info(f"Hierarchy levels: {[(k, len(v)) for k, v in hierarchy_info.items()]}")
        return hierarchy_info

    def prepare_time_series_data(
        self,
        start_day: int = 1,
        end_day: Optional[int] = None,
        min_nonzero_ratio: float = 0.1
    ) -> pd.DataFrame:
        """
        Prepare time series data in long format with comprehensive validation.

        Args:
            start_day: First day to include (1-based, must be >= 1).
            end_day: Last day to include. If None, uses all available days.
            min_nonzero_ratio: Minimum ratio of non-zero values for series inclusion (0.0-1.0).

        Returns:
            DataFrame with columns: ['id', 'date', 'sales', 'store_id', 'item_id', etc.]

        Raises:
            ValueError: If sales data is not loaded, or if parameters are invalid.
            RuntimeError: If data preparation fails unexpectedly.
        """
        if self.sales_data is None:
            raise ValueError("Sales data not loaded. Call load_data() first.")

        # Validate input parameters
        if not isinstance(start_day, int) or start_day < 1:
            raise ValueError(f"start_day must be a positive integer >= 1, got: {start_day}")

        if end_day is not None and (not isinstance(end_day, int) or end_day < start_day):
            raise ValueError(f"end_day must be an integer >= start_day ({start_day}), got: {end_day}")

        if not isinstance(min_nonzero_ratio, (int, float)) or not (0.0 <= min_nonzero_ratio <= 1.0):
            raise ValueError(f"min_nonzero_ratio must be between 0.0 and 1.0, got: {min_nonzero_ratio}")

        self.logger.info("Preparing time series data...")

        try:
            # Get day columns
            day_cols = [col for col in self.sales_data.columns if col.startswith('d_')]
            if not day_cols:
                raise ValueError("No day columns (d_*) found in sales data")

            if end_day is None:
                end_day = len(day_cols)
            elif end_day > len(day_cols):
                self.logger.warning(f"end_day ({end_day}) exceeds available days ({len(day_cols)}), using maximum")
                end_day = len(day_cols)

            # Validate day range
            if start_day > len(day_cols):
                raise ValueError(f"start_day ({start_day}) exceeds available days ({len(day_cols)})")

            # Select day range
            selected_days = day_cols[start_day-1:end_day]
            if not selected_days:
                raise ValueError(f"No days selected for range d_{start_day} to d_{end_day}")

            self.logger.info(f"Selected {len(selected_days)} days from d_{start_day} to d_{end_day}")

            # Filter series with sufficient non-zero values
            try:
                sales_values = self.sales_data[selected_days].values
                if sales_values.size == 0:
                    raise ValueError("Selected sales data is empty")

                nonzero_ratios = (sales_values > 0).mean(axis=1)
                valid_series_mask = nonzero_ratios >= min_nonzero_ratio

                if not valid_series_mask.any():
                    raise ValueError(f"No time series meet the minimum non-zero ratio threshold of {min_nonzero_ratio}")

                filtered_sales = self.sales_data[valid_series_mask].copy()
                self.logger.info(f"Filtered to {len(filtered_sales)} series with ≥{min_nonzero_ratio*100}% non-zero values")

            except Exception as e:
                raise RuntimeError(f"Error filtering time series data: {e}") from e

            # Validate required columns for melting
            id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
            missing_cols = [col for col in id_cols if col not in filtered_sales.columns]
            if missing_cols:
                raise ValueError(f"Missing required ID columns: {missing_cols}")

            # Melt to long format
            try:
                melted_data = pd.melt(
                    filtered_sales,
                    id_vars=id_cols,
                    value_vars=selected_days,
                    var_name='d',
                    value_name='sales'
                )

                if melted_data.empty:
                    raise ValueError("Melted data is empty")

            except Exception as e:
                raise RuntimeError(f"Error melting data to long format: {e}") from e

            # Add date information
            try:
                if self.calendar_data is not None:
                    if 'd' not in self.calendar_data.columns or 'date' not in self.calendar_data.columns:
                        self.logger.warning("Calendar data missing 'd' or 'date' columns, skipping date mapping")
                    else:
                        date_mapping = self.calendar_data[['d', 'date']].set_index('d')['date'].to_dict()
                        melted_data['date'] = melted_data['d'].map(date_mapping)

                        # Validate date mapping
                        missing_dates = melted_data['date'].isna().sum()
                        if missing_dates > 0:
                            self.logger.warning(f"Unable to map {missing_dates} dates from calendar data")

                        try:
                            melted_data['date'] = pd.to_datetime(melted_data['date'], errors='coerce')
                        except Exception as e:
                            self.logger.warning(f"Error converting dates to datetime: {e}")

            except Exception as e:
                self.logger.warning(f"Error adding date information: {e}")

            # Sort by id and date
            try:
                sort_cols = ['id']
                if 'date' in melted_data.columns:
                    sort_cols.append('date')
                melted_data = melted_data.sort_values(sort_cols).reset_index(drop=True)

            except Exception as e:
                self.logger.warning(f"Error sorting data: {e}")

            self.logger.info(f"Prepared time series data: {melted_data.shape}")
            return melted_data

        except Exception as e:
            error_msg = f"Unexpected error preparing time series data: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def add_calendar_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add calendar-based features to the data with comprehensive validation.

        Args:
            data: DataFrame with 'd' column for merging with calendar data.

        Returns:
            DataFrame with additional calendar features including derived temporal features.

        Raises:
            ValueError: If calendar data is not loaded or required columns are missing.
            RuntimeError: If feature addition fails unexpectedly.
        """
        if self.calendar_data is None:
            raise ValueError("Calendar data not loaded. Call load_data() first.")

        if data.empty:
            raise ValueError("Input data is empty")

        if 'd' not in data.columns:
            raise ValueError("Input data must contain 'd' column for calendar feature merging")

        self.logger.info("Adding calendar features...")

        try:
            # Validate calendar data structure
            if self.calendar_data.empty:
                raise ValueError("Calendar data is empty")

            if 'd' not in self.calendar_data.columns:
                raise ValueError("Calendar data must contain 'd' column")

            # Merge with calendar data, avoiding duplicate columns
            calendar_features = ['d', 'date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year']
            available_features = [col for col in calendar_features if col in self.calendar_data.columns]

            # Remove features already present in data (except 'd' which is the merge key)
            existing_cols = set(data.columns)
            merge_features = ['d'] + [col for col in available_features if col != 'd' and col not in existing_cols]

            if len(merge_features) <= 1:
                self.logger.warning("No new calendar features to add (all already present)")
                enriched_data = data.copy()
            else:
                enriched_data = data.merge(
                    self.calendar_data[merge_features].drop_duplicates(subset=['d']),
                    on='d',
                    how='left'
                )

            # Check merge results
            if enriched_data.empty:
                raise ValueError("Calendar feature merge resulted in empty data")

            # Add derived features if date column is available
            if 'date' in self.calendar_data.columns:
                try:
                    date_col = pd.to_datetime(self.calendar_data['date'], errors='coerce')
                    invalid_dates = date_col.isna().sum()
                    if invalid_dates > 0:
                        self.logger.warning(f"{invalid_dates} invalid dates found in calendar data")

                    calendar_derived = pd.DataFrame()
                    calendar_derived['d'] = self.calendar_data['d']
                    calendar_derived['day_of_month'] = date_col.dt.day
                    calendar_derived['quarter'] = date_col.dt.quarter
                    calendar_derived['is_weekend'] = date_col.dt.weekday >= 5

                    enriched_data = enriched_data.merge(calendar_derived, on='d', how='left')

                except Exception as e:
                    self.logger.warning(f"Error adding derived calendar features: {e}")

            # Add event information if available
            event_cols = [col for col in self.calendar_data.columns if 'event' in col.lower()]
            if event_cols:
                try:
                    event_data = self.calendar_data[['d'] + event_cols]
                    enriched_data = enriched_data.merge(event_data, on='d', how='left')
                    self.logger.info(f"Added {len(event_cols)} event columns")

                except Exception as e:
                    self.logger.warning(f"Error adding event features: {e}")

            # Validate final result
            if enriched_data.shape[0] != data.shape[0]:
                self.logger.warning(f"Row count changed after adding calendar features: {data.shape[0]} -> {enriched_data.shape[0]}")

            self.logger.info(f"Added calendar features: {enriched_data.shape}")
            return enriched_data

        except Exception as e:
            error_msg = f"Unexpected error adding calendar features: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features to the data.

        Args:
            data: DataFrame with 'store_id', 'item_id', and 'wm_yr_wk' columns.

        Returns:
            DataFrame with additional price features.

        Raises:
            ValueError: If prices data is not loaded.
        """
        if self.prices_data is None:
            raise ValueError("Prices data not loaded. Call load_data() first.")

        self.logger.info("Adding price features...")

        # Merge with price data
        enriched_data = data.merge(
            self.prices_data[['store_id', 'item_id', 'wm_yr_wk', 'sell_price']],
            on=['store_id', 'item_id', 'wm_yr_wk'],
            how='left'
        )

        # Add price-derived features
        grouped = enriched_data.groupby(['store_id', 'item_id'])['sell_price']
        enriched_data['price_mean'] = grouped.transform('mean')
        enriched_data['price_std'] = grouped.transform('std').fillna(0)
        enriched_data['price_min'] = grouped.transform('min')
        enriched_data['price_max'] = grouped.transform('max')

        # Add price change indicators
        enriched_data['price_change'] = (
            enriched_data.groupby(['store_id', 'item_id'])['sell_price']
            .pct_change()
            .fillna(0)
        )

        self.logger.info(f"Added price features: {enriched_data.shape}")
        return enriched_data


class HierarchicalDataBuilder:
    """
    Builds hierarchical aggregations of M5 data for reconciliation.

    This class creates multiple aggregation levels of the M5 dataset to support
    hierarchical forecasting and reconciliation. It supports various levels
    including total, state, store, category, department, and item levels.
    """

    def __init__(self, aggregation_levels: List[str]) -> None:
        """
        Initialize hierarchical data builder with validation.

        Args:
            aggregation_levels: List of aggregation levels to build. Valid levels include:
                'total', 'state', 'store', 'cat', 'dept', 'state_cat', 'state_dept',
                'store_cat', 'store_dept', 'item', 'item_store'.

        Raises:
            ValueError: If aggregation_levels is empty or contains invalid levels.
        """
        if not aggregation_levels:
            raise ValueError("aggregation_levels cannot be empty")

        valid_levels = {
            'total', 'state', 'store', 'cat', 'dept',
            'state_cat', 'state_dept', 'store_cat', 'store_dept',
            'item', 'item_store'
        }

        invalid_levels = set(aggregation_levels) - valid_levels
        if invalid_levels:
            raise ValueError(f"Invalid aggregation levels: {invalid_levels}. Valid levels: {valid_levels}")

        self.aggregation_levels = aggregation_levels
        self.logger = logging.getLogger(__name__)

    def build_hierarchy(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Build all hierarchical aggregations with comprehensive validation.

        Args:
            data: Base time series data containing required columns for aggregation.
                Must include columns like 'sales', 'date', 'state_id', 'store_id', etc.

        Returns:
            Dictionary mapping aggregation levels to their corresponding aggregated DataFrames.
            Each DataFrame contains aggregated sales data with appropriate grouping columns.

        Raises:
            ValueError: If input data is invalid or missing required columns.
            RuntimeError: If aggregation fails unexpectedly.
        """
        if data.empty:
            raise ValueError("Input data is empty")

        # Validate required columns for basic aggregation
        required_base_cols = ['sales']
        missing_base_cols = [col for col in required_base_cols if col not in data.columns]
        if missing_base_cols:
            raise ValueError(f"Missing required base columns: {missing_base_cols}")

        # Validate numeric sales data
        if not pd.api.types.is_numeric_dtype(data['sales']):
            raise ValueError("Sales column must contain numeric data")

        self.logger.info("Building hierarchical aggregations...")

        hierarchy_data = {}
        failed_levels = []

        for level in self.aggregation_levels:
            try:
                self.logger.info(f"Building aggregation level: {level}")

                # Validate required columns for this level
                required_cols = self._get_required_columns(level)
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    error_msg = f"Missing required columns for level '{level}': {missing_cols}"
                    self.logger.error(error_msg)
                    failed_levels.append(level)
                    continue

                aggregated_data = self._build_aggregation_level(data, level)

                # Validate aggregation result
                if aggregated_data.empty:
                    self.logger.warning(f"Aggregation for level '{level}' resulted in empty data")
                    failed_levels.append(level)
                    continue

                hierarchy_data[level] = aggregated_data
                self.logger.info(f"Successfully built aggregation level '{level}': {aggregated_data.shape}")

            except Exception as e:
                error_msg = f"Failed to build aggregation level '{level}': {e}"
                self.logger.error(error_msg)
                failed_levels.append(level)

        if failed_levels:
            self.logger.warning(f"Failed to build {len(failed_levels)} aggregation levels: {failed_levels}")

        if not hierarchy_data:
            raise RuntimeError("Failed to build any aggregation levels")

        self.logger.info(f"Successfully built {len(hierarchy_data)} aggregation levels")
        return hierarchy_data

    def _get_required_columns(self, level: str) -> List[str]:
        """
        Get required columns for a specific aggregation level.

        Args:
            level: Aggregation level name.

        Returns:
            List of required column names for the aggregation level.
        """
        base_cols = ['sales']

        level_column_map = {
            'total': [],
            'state': ['state_id'],
            'store': ['store_id'],
            'cat': ['cat_id'],
            'dept': ['dept_id'],
            'state_cat': ['state_id', 'cat_id'],
            'state_dept': ['state_id', 'dept_id'],
            'store_cat': ['store_id', 'cat_id'],
            'store_dept': ['store_id', 'dept_id'],
            'item': ['item_id'],
            'item_store': ['item_id', 'store_id']
        }

        if 'date' in base_cols:
            base_cols.append('date')

        return base_cols + level_column_map.get(level, [])

    def _build_aggregation_level(self, data: pd.DataFrame, level: str) -> pd.DataFrame:
        """
        Build specific aggregation level with validation and error handling.

        Args:
            data: Base time series data containing required columns for the level.
            level: Aggregation level name (must be one of the valid levels).

        Returns:
            Aggregated data for the specified level with appropriate grouping and metadata.

        Raises:
            ValueError: If aggregation level is unknown or data is insufficient.
            RuntimeError: If aggregation fails due to data issues.
        """
        try:
            aggregation_methods = {
                "total": self._build_total_aggregation,
                "state": self._build_state_aggregation,
                "store": self._build_store_aggregation,
                "cat": self._build_category_aggregation,
                "dept": self._build_department_aggregation,
                "state_cat": self._build_state_category_aggregation,
                "state_dept": self._build_state_department_aggregation,
                "store_cat": self._build_store_category_aggregation,
                "store_dept": self._build_store_department_aggregation,
                "item": self._build_item_aggregation,
                "item_store": lambda x: x.copy()  # Bottom level - no aggregation
            }

            if level not in aggregation_methods:
                raise ValueError(f"Unknown aggregation level: {level}")

            result = aggregation_methods[level](data)

            # Validate aggregation result
            if result.empty:
                raise RuntimeError(f"Aggregation for level '{level}' produced empty result")

            if 'sales' not in result.columns:
                raise RuntimeError(f"Aggregation for level '{level}' missing sales column")

            # Check for NaN values in sales
            nan_sales = result['sales'].isna().sum()
            if nan_sales > 0:
                self.logger.warning(f"Level '{level}' has {nan_sales} NaN sales values")

            return result

        except Exception as e:
            error_msg = f"Error building aggregation level '{level}': {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _build_total_aggregation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build total aggregation (all sales combined)."""
        return (
            data.groupby(['date'])
            .agg({'sales': 'sum'})
            .reset_index()
            .assign(id='total', level='total')
        )

    def _build_state_aggregation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build state-level aggregation."""
        return (
            data.groupby(['state_id', 'date'])
            .agg({'sales': 'sum'})
            .reset_index()
            .assign(id=lambda x: x['state_id'], level='state')
        )

    def _build_store_aggregation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build store-level aggregation."""
        return (
            data.groupby(['store_id', 'date'])
            .agg({'sales': 'sum'})
            .reset_index()
            .assign(id=lambda x: x['store_id'], level='store')
        )

    def _build_category_aggregation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build category-level aggregation."""
        return (
            data.groupby(['cat_id', 'date'])
            .agg({'sales': 'sum'})
            .reset_index()
            .assign(id=lambda x: x['cat_id'], level='cat')
        )

    def _build_department_aggregation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build department-level aggregation."""
        return (
            data.groupby(['dept_id', 'date'])
            .agg({'sales': 'sum'})
            .reset_index()
            .assign(id=lambda x: x['dept_id'], level='dept')
        )

    def _build_state_category_aggregation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build state-category aggregation."""
        return (
            data.groupby(['state_id', 'cat_id', 'date'])
            .agg({'sales': 'sum'})
            .reset_index()
            .assign(id=lambda x: x['state_id'] + '_' + x['cat_id'], level='state_cat')
        )

    def _build_state_department_aggregation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build state-department aggregation."""
        return (
            data.groupby(['state_id', 'dept_id', 'date'])
            .agg({'sales': 'sum'})
            .reset_index()
            .assign(id=lambda x: x['state_id'] + '_' + x['dept_id'], level='state_dept')
        )

    def _build_store_category_aggregation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build store-category aggregation."""
        return (
            data.groupby(['store_id', 'cat_id', 'date'])
            .agg({'sales': 'sum'})
            .reset_index()
            .assign(id=lambda x: x['store_id'] + '_' + x['cat_id'], level='store_cat')
        )

    def _build_store_department_aggregation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build store-department aggregation."""
        return (
            data.groupby(['store_id', 'dept_id', 'date'])
            .agg({'sales': 'sum'})
            .reset_index()
            .assign(id=lambda x: x['store_id'] + '_' + x['dept_id'], level='store_dept')
        )

    def _build_item_aggregation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build item-level aggregation."""
        return (
            data.groupby(['item_id', 'date'])
            .agg({'sales': 'sum'})
            .reset_index()
            .assign(id=lambda x: x['item_id'], level='item')
        )