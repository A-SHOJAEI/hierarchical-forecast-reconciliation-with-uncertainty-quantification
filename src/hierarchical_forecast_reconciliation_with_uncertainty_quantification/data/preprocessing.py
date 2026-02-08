"""Data preprocessing utilities for hierarchical forecasting."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class M5Preprocessor:
    """
    Preprocesses M5 data for hierarchical forecasting models.

    Handles data cleaning, feature engineering, scaling, and train/test splits
    while maintaining the hierarchical structure.
    """

    def __init__(
        self,
        scaling_method: str = "standard",
        handle_zeros: str = "log1p",
        outlier_threshold: float = 3.0
    ) -> None:
        """
        Initialize M5 preprocessor.

        Args:
            scaling_method: Scaling method ("standard", "minmax", "none").
            handle_zeros: How to handle zero values ("log1p", "none").
            outlier_threshold: Z-score threshold for outlier detection.
        """
        self.scaling_method = scaling_method
        self.handle_zeros = handle_zeros
        self.outlier_threshold = outlier_threshold
        self.logger = logging.getLogger(__name__)

        # Fitted scalers
        self.scalers: Dict[str, Union[StandardScaler, MinMaxScaler]] = {}
        self.is_fitted = False

    def fit_transform(
        self,
        data: pd.DataFrame,
        target_col: str = "sales"
    ) -> pd.DataFrame:
        """
        Fit preprocessor and transform data.

        Args:
            data: Input data with hierarchical time series.
            target_col: Name of target variable column.

        Returns:
            Transformed data.
        """
        self.logger.info("Fitting preprocessor and transforming data...")

        # Make copy to avoid modifying original
        processed_data = data.copy()

        # Handle missing values
        processed_data = self._handle_missing_values(processed_data)

        # Handle zero values
        if self.handle_zeros == "log1p":
            processed_data[target_col] = np.log1p(processed_data[target_col])
            self.logger.info("Applied log1p transformation to target variable")

        # Detect and handle outliers
        processed_data = self._handle_outliers(processed_data, target_col)

        # Feature engineering
        processed_data = self._engineer_features(processed_data)

        # Fit and apply scaling
        if self.scaling_method != "none":
            processed_data = self._fit_transform_scaling(processed_data, target_col)

        self.is_fitted = True
        self.logger.info(f"Preprocessing complete: {processed_data.shape}")

        return processed_data

    def transform(self, data: pd.DataFrame, target_col: str = "sales") -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.

        Args:
            data: Input data to transform.
            target_col: Name of target variable column.

        Returns:
            Transformed data.

        Raises:
            ValueError: If preprocessor is not fitted.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform() first.")

        self.logger.info("Transforming data...")

        # Make copy to avoid modifying original
        processed_data = data.copy()

        # Handle missing values
        processed_data = self._handle_missing_values(processed_data)

        # Handle zero values
        if self.handle_zeros == "log1p":
            processed_data[target_col] = np.log1p(processed_data[target_col])

        # Feature engineering (same features as training)
        processed_data = self._engineer_features(processed_data)

        # Apply scaling
        if self.scaling_method != "none":
            processed_data = self._transform_scaling(processed_data, target_col)

        return processed_data

    def inverse_transform(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        target_col: str = "sales"
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Inverse transform predictions back to original scale.

        Args:
            data: Transformed data or predictions.
            target_col: Name of target variable column.

        Returns:
            Data in original scale.
        """
        if isinstance(data, pd.DataFrame):
            result = data.copy()

            # Inverse scaling
            if self.scaling_method != "none" and target_col in self.scalers:
                result[target_col] = self.scalers[target_col].inverse_transform(
                    result[[target_col]]
                ).flatten()

            # Inverse zero handling
            if self.handle_zeros == "log1p":
                result[target_col] = np.expm1(result[target_col])

            return result
        else:
            # Handle numpy array
            result = data.copy()

            # Inverse scaling
            if self.scaling_method != "none" and target_col in self.scalers:
                result = self.scalers[target_col].inverse_transform(
                    result.reshape(-1, 1)
                ).flatten()

            # Inverse zero handling
            if self.handle_zeros == "log1p":
                result = np.expm1(result)

            return result

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data."""
        self.logger.info("Handling missing values...")

        # Forward fill for time series continuity
        data = data.sort_values(['id', 'date'])
        data['sales'] = data.groupby('id')['sales'].transform(
            lambda x: x.ffill()
        )

        # Backward fill for remaining missing values
        data['sales'] = data.groupby('id')['sales'].transform(
            lambda x: x.bfill()
        )

        # Fill remaining with 0
        data['sales'] = data['sales'].fillna(0)

        # Handle missing categorical features
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'id' and col != 'date':
                data[col] = data[col].fillna('unknown')

        # Handle missing numerical features
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col != 'sales':
                median_val = data[col].median()
                data[col] = data[col].fillna(median_val if pd.notna(median_val) else 0)

        return data

    def _handle_outliers(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Detect and handle outliers in the target variable."""
        self.logger.info("Handling outliers...")

        # Calculate Z-scores by series
        z_scores = data.groupby('id')[target_col].transform(
            lambda x: np.abs((x - x.mean()) / x.std())
        )

        # Identify outliers
        outlier_mask = z_scores > self.outlier_threshold
        n_outliers = outlier_mask.sum()

        if n_outliers > 0:
            self.logger.info(f"Found {n_outliers} outliers")

            # Cap outliers at threshold percentile
            for series_id in data['id'].unique():
                series_mask = data['id'] == series_id
                series_outliers = outlier_mask & series_mask

                if series_outliers.sum() > 0:
                    # Cap at 99th percentile of the series
                    cap_value = data.loc[series_mask, target_col].quantile(0.99)
                    data.loc[series_outliers, target_col] = cap_value

        return data

    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features for forecasting."""
        self.logger.info("Engineering features...")

        # Sort by id and date
        data = data.sort_values(['id', 'date'])

        # Lag features
        for lag in [1, 7, 14, 28]:
            data[f'sales_lag_{lag}'] = (
                data.groupby('id')['sales']
                .shift(lag)
                .fillna(0)
            )

        # Rolling statistics
        for window in [7, 14, 28]:
            data[f'sales_mean_{window}'] = (
                data.groupby('id')['sales']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )

            data[f'sales_std_{window}'] = (
                data.groupby('id')['sales']
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(0, drop=True)
                .fillna(0)
            )

        # Trend features
        data['sales_trend_7'] = (
            data.groupby('id')['sales']
            .pct_change(periods=7)
            .fillna(0)
        )

        # Seasonal features from date
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data['day_of_week'] = data['date'].dt.dayofweek.astype(int)
            data['day_of_month'] = data['date'].dt.day.astype(int)
            data['week_of_year'] = data['date'].dt.isocalendar().week.astype(int).values
            data['month'] = data['date'].dt.month.astype(int)
            data['quarter'] = data['date'].dt.quarter.astype(int)

        return data

    def _fit_transform_scaling(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Fit scalers and transform numerical features."""
        self.logger.info(f"Fitting {self.scaling_method} scaling...")

        # Replace inf values with NaN first, then fill
        data = data.replace([np.inf, -np.inf], np.nan)

        # Select numerical columns for scaling
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        # Remove id-like columns and date columns
        exclude_cols = ['id', 'date', 'wm_yr_wk']
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

        # Fit and transform each column
        for col in numerical_cols:
            # Replace remaining NaN with 0 for this column
            data[col] = data[col].fillna(0)

            if self.scaling_method == "standard":
                scaler = StandardScaler()
            elif self.scaling_method == "minmax":
                scaler = MinMaxScaler()
            else:
                continue

            # Check for constant columns
            col_std = data[col].std()
            if col_std == 0 or np.isnan(col_std):
                continue

            try:
                scaler.fit(data[[col]])
                data[col] = scaler.transform(data[[col]]).flatten()
                self.scalers[col] = scaler
            except Exception as e:
                self.logger.warning(f"Scaling failed for column {col}: {e}")

        return data

    def _transform_scaling(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Apply fitted scalers to transform data."""
        # Apply scaling using fitted scalers
        for col, scaler in self.scalers.items():
            if col in data.columns:
                data[col] = scaler.transform(data[[col]]).flatten()

        return data

    def create_train_test_split(
        self,
        data: pd.DataFrame,
        train_days: int,
        validation_days: int,
        test_days: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits maintaining temporal order.

        Args:
            data: Preprocessed time series data.
            train_days: Number of training days.
            validation_days: Number of validation days.
            test_days: Number of test days.

        Returns:
            Tuple of (train_data, validation_data, test_data).
        """
        self.logger.info("Creating train/validation/test splits...")

        # Ensure date column is datetime
        data = data.copy()
        data['date'] = pd.to_datetime(data['date'])

        # Sort by date
        data = data.sort_values('date')
        unique_dates = sorted(data['date'].unique())

        total_needed = train_days + validation_days + test_days
        available = len(unique_dates)

        if available < total_needed:
            # Adjust splits proportionally
            ratio = available / total_needed
            train_days = max(1, int(train_days * ratio))
            validation_days = max(1, int(validation_days * ratio))
            test_days = max(1, min(test_days, available - train_days - validation_days))
            self.logger.warning(
                f"Adjusted splits: train={train_days}, val={validation_days}, "
                f"test={test_days} (available={available})"
            )

        train_end_idx = train_days
        val_end_idx = train_days + validation_days

        # Split dates
        train_dates = set(unique_dates[:train_end_idx])
        val_dates = set(unique_dates[train_end_idx:val_end_idx])
        test_dates = set(unique_dates[val_end_idx:val_end_idx + test_days])

        # Create splits
        train_data = data[data['date'].isin(train_dates)].copy()
        val_data = data[data['date'].isin(val_dates)].copy()
        test_data = data[data['date'].isin(test_dates)].copy()

        self.logger.info(f"Train split: {train_data.shape} ({len(train_dates)} days)")
        self.logger.info(f"Validation split: {val_data.shape} ({len(val_dates)} days)")
        self.logger.info(f"Test split: {test_data.shape} ({len(test_dates)} days)")

        return train_data, val_data, test_data


class HierarchyBuilder:
    """Builds hierarchical aggregation matrices for reconciliation."""

    def __init__(self, hierarchy_levels: List[str]) -> None:
        """
        Initialize hierarchy builder.

        Args:
            hierarchy_levels: List of aggregation levels in the hierarchy.
        """
        self.hierarchy_levels = hierarchy_levels
        self.logger = logging.getLogger(__name__)

    def build_aggregation_matrix(
        self,
        hierarchy_data: Dict[str, pd.DataFrame]
    ) -> sparse.csr_matrix:
        """
        Build aggregation matrix S for hierarchical reconciliation.

        The aggregation matrix S defines how bottom-level forecasts aggregate
        to upper levels in the hierarchy.

        Args:
            hierarchy_data: Dictionary mapping levels to aggregated data.

        Returns:
            Sparse aggregation matrix S where S @ bottom_forecasts = all_forecasts.
        """
        self.logger.info("Building aggregation matrix...")

        # Get bottom level (item_store)
        bottom_level = "item_store"
        if bottom_level not in hierarchy_data:
            raise ValueError(f"Bottom level '{bottom_level}' not found in hierarchy data")

        bottom_data = hierarchy_data[bottom_level]
        bottom_series = sorted(bottom_data['id'].unique())
        n_bottom = len(bottom_series)

        self.logger.info(f"Bottom level has {n_bottom} series")

        # Initialize matrix rows
        matrix_rows = []
        all_series = []

        # Add each hierarchy level
        for level in self.hierarchy_levels:
            if level not in hierarchy_data:
                continue

            level_data = hierarchy_data[level]
            level_series = sorted(level_data['id'].unique())

            self.logger.info(f"Level {level}: {len(level_series)} series")

            for series_id in level_series:
                row = self._build_aggregation_row(
                    series_id, level, bottom_series, hierarchy_data
                )
                matrix_rows.append(row)
                all_series.append(series_id)

        # Convert to sparse matrix
        S = sparse.csr_matrix(np.array(matrix_rows))

        self.logger.info(f"Built aggregation matrix: {S.shape}")
        return S

    def _build_aggregation_row(
        self,
        series_id: str,
        level: str,
        bottom_series: List[str],
        hierarchy_data: Dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """
        Build a single row of the aggregation matrix.

        Args:
            series_id: ID of the series for this row.
            level: Hierarchy level of the series.
            bottom_series: List of bottom-level series IDs.
            hierarchy_data: Dictionary mapping levels to data.

        Returns:
            Binary row vector indicating which bottom series contribute to this aggregate.
        """
        row = np.zeros(len(bottom_series), dtype=float)

        if level == "item_store":
            # Bottom level - identity mapping
            if series_id in bottom_series:
                idx = bottom_series.index(series_id)
                row[idx] = 1.0
        else:
            # Upper level - find contributing bottom series
            contributing_series = self._find_contributing_series(
                series_id, level, hierarchy_data["item_store"]
            )

            for bottom_id in contributing_series:
                if bottom_id in bottom_series:
                    idx = bottom_series.index(bottom_id)
                    row[idx] = 1.0

        return row

    def _find_contributing_series(
        self,
        aggregate_id: str,
        level: str,
        bottom_data: pd.DataFrame
    ) -> List[str]:
        """
        Find bottom-level series that contribute to an aggregate.

        Args:
            aggregate_id: ID of the aggregate series.
            level: Hierarchy level.
            bottom_data: Bottom-level data.

        Returns:
            List of contributing bottom-level series IDs.
        """
        if level == "total":
            return bottom_data['id'].unique().tolist()
        elif level == "state":
            if 'state_id' in bottom_data.columns:
                return bottom_data[bottom_data['state_id'] == aggregate_id]['id'].unique().tolist()
            return []
        elif level == "store":
            if 'store_id' in bottom_data.columns:
                return bottom_data[bottom_data['store_id'] == aggregate_id]['id'].unique().tolist()
            return []
        elif level == "cat":
            if 'cat_id' in bottom_data.columns:
                return bottom_data[bottom_data['cat_id'] == aggregate_id]['id'].unique().tolist()
            return []
        elif level == "dept":
            if 'dept_id' in bottom_data.columns:
                return bottom_data[bottom_data['dept_id'] == aggregate_id]['id'].unique().tolist()
            return []
        elif level == "state_cat":
            # aggregate_id format: "CA_HOBBIES" - state is first part before underscore
            parts = aggregate_id.split('_', 1)
            if len(parts) == 2 and 'state_id' in bottom_data.columns and 'cat_id' in bottom_data.columns:
                return bottom_data[
                    (bottom_data['state_id'] == parts[0]) &
                    (bottom_data['cat_id'] == parts[1])
                ]['id'].unique().tolist()
            return []
        elif level == "state_dept":
            parts = aggregate_id.split('_', 1)
            if len(parts) == 2 and 'state_id' in bottom_data.columns and 'dept_id' in bottom_data.columns:
                return bottom_data[
                    (bottom_data['state_id'] == parts[0]) &
                    (bottom_data['dept_id'] == parts[1])
                ]['id'].unique().tolist()
            return []
        elif level == "store_cat":
            # aggregate_id format: "CA_1_HOBBIES" - store is "XX_N", cat is the rest
            parts = aggregate_id.split('_', 2)
            if len(parts) >= 3 and 'store_id' in bottom_data.columns and 'cat_id' in bottom_data.columns:
                store_id = f"{parts[0]}_{parts[1]}"
                cat_id = parts[2]
                return bottom_data[
                    (bottom_data['store_id'] == store_id) &
                    (bottom_data['cat_id'] == cat_id)
                ]['id'].unique().tolist()
            return []
        elif level == "store_dept":
            parts = aggregate_id.split('_', 2)
            if len(parts) >= 3 and 'store_id' in bottom_data.columns and 'dept_id' in bottom_data.columns:
                store_id = f"{parts[0]}_{parts[1]}"
                dept_id = '_'.join(parts[2:])
                return bottom_data[
                    (bottom_data['store_id'] == store_id) &
                    (bottom_data['dept_id'] == dept_id)
                ]['id'].unique().tolist()
            return []
        elif level == "item":
            if 'item_id' in bottom_data.columns:
                return bottom_data[bottom_data['item_id'] == aggregate_id]['id'].unique().tolist()
            return []
        else:
            self.logger.warning(f"Unknown hierarchy level: {level}")
            return []

    def get_hierarchy_structure(
        self,
        hierarchy_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, int]]:
        """
        Get the structure of the hierarchy.

        Args:
            hierarchy_data: Dictionary mapping levels to data.

        Returns:
            Dictionary with hierarchy structure information.
        """
        structure = {}

        for level in self.hierarchy_levels:
            if level in hierarchy_data:
                data = hierarchy_data[level]
                structure[level] = {
                    'n_series': data['id'].nunique(),
                    'total_observations': len(data)
                }

        return structure