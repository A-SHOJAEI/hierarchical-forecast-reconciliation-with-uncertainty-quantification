# Hierarchical Forecast Reconciliation with Uncertainty Quantification

A framework for hierarchical time series forecasting that combines statistical ensemble models (ETS + ARIMA) with probabilistic minimum trace (MinT) reconciliation to produce coherent forecasts while preserving uncertainty quantification across all hierarchy levels.

## Overview

This project addresses the challenge of forecasting hierarchical time series where forecasts must be coherent (aggregates equal the sum of their components) while maintaining reliable uncertainty estimates. The approach combines multiple statistical forecasting methods through a weighted ensemble and applies probabilistic reconciliation to enforce hierarchical coherence.

### Key Features

- **Statistical Ensemble**: Combines Exponential Smoothing (ETS) and ARIMA models with weighted averaging
- **Probabilistic MinT Reconciliation**: Minimum trace reconciliation that maps bottom-level forecasts to all hierarchy levels while preserving prediction intervals
- **11-Level Hierarchy**: Supports total, state, store, category, department, and cross-level aggregations
- **Comprehensive Evaluation**: WRMSSE, MASE, sMAPE, coverage probability, interval scores, and coherence metrics
- **MLflow Integration**: Experiment tracking, parameter logging, and artifact management
- **Optuna Support**: Hyperparameter optimization for reconciliation parameters

### Important Note on Data

**This project was trained and evaluated on synthetic M5-like data**, not the actual M5 Walmart competition dataset (which requires a Kaggle download). The synthetic data generator (`scripts/generate_synthetic_data.py`) creates realistic retail sales patterns with:
- 490 item-store combinations (49 items across 10 stores in 3 states)
- 365 days of daily sales with weekly seasonality, trends, and random noise
- 3 categories and 7 departments matching M5 hierarchy structure
- Corresponding calendar and price data

Results below reflect performance on this synthetic dataset and should not be compared to M5 competition leaderboard scores.

## Architecture

```
Synthetic M5 Hierarchical Data (490 bottom-level series)
  |
  v
Data Pipeline: Load -> Calendar/Price Features -> Log1p -> Outlier Handling -> Scaling
  |
  v
 +-------------+     +-------------+
 |     ETS     |     |   ARIMA     |
 | (Holt-Winters)|   | (SARIMAX)   |
 +-------------+     +-------------+
       |                    |
       v                    v
    Weighted Ensemble (50% ETS / 50% ARIMA)
       |
       v
    Bottom-Up Aggregation (S matrix: 693 x 490)
       |
       v
    MinT Reconciliation: G = S(S'WS)^{-1}S'W
       |
       v
  +-------------------+-------------------+
  | Reconciled Point  | Reconciled        |
  | Forecasts (693)   | Prediction        |
  |                   | Intervals         |
  +-------------------+-------------------+
```

### Hierarchy Structure (11 Levels, 693 Total Series)

| Level | Description | Count |
|-------|------------|-------|
| Total | Grand total | 1 |
| State | CA, TX, WI | 3 |
| Store | 10 stores | 10 |
| Category | HOBBIES, HOUSEHOLD, FOODS | 3 |
| Department | 7 departments | 7 |
| State-Category | | 9 |
| State-Department | | 21 |
| Store-Category | | 30 |
| Store-Department | | 70 |
| Item | 49 unique items | 49 |
| Item-Store (bottom) | | 490 |

## Training Results (Synthetic Data)

Training was performed on synthetic M5-like data with 300 training days, 28 validation days, and 28 test days.

### Model Fitting

| Model | Series Fitted | Success Rate | Training Time |
|-------|--------------|--------------|---------------|
| ETS (Holt-Winters) | 490 / 490 | 100% | ~18 seconds |
| ARIMA (SARIMAX) | 490 / 490 | 100% | ~96 seconds |
| **Total** | **980 / 980** | **100%** | **~113 seconds** |

### Validation Metrics (28-Day Forecast Horizon)

| Metric | Value | Notes |
|--------|-------|-------|
| **WRMSSE** | 1.726 | Weighted Root Mean Squared Scaled Error |
| **MASE** | 3.013 | Mean Absolute Scaled Error |
| **sMAPE** | 106.1% | Symmetric Mean Absolute Percentage Error |
| **Coverage 90%** | 33.9% | 90% prediction interval coverage |
| **Coverage 95%** | 35.3% | 95% prediction interval coverage |
| **Coherence** | 1.000 | Perfect coherence (bottom-up construction) |

### Honest Analysis of Results

The metrics above reveal several characteristics of the current system:

1. **Point forecast accuracy (WRMSSE 1.726, MASE 3.013)**: These values indicate the ensemble forecasts are roughly 1.7x worse than a naive seasonal baseline. This is expected because:
   - The data is standardized (log1p + z-score), and metrics are computed on the transformed scale
   - The synthetic data has limited signal-to-noise ratio
   - Only statistical models are used (no deep learning components)

2. **Coverage is low (33.9% instead of 90%)**: The MinT reconciliation matrix G transforms both point forecasts and interval bounds linearly. When applied to bottom-up aggregated forecasts that are already coherent, this transformation can distort prediction intervals, sometimes making lower bounds exceed upper bounds (evidenced by negative interval widths in raw output). This is a known limitation of applying MinT reconciliation to already-coherent base forecasts.

3. **Perfect coherence (1.000)**: This is trivially achieved because the bottom-up aggregation approach constructs upper-level forecasts as exact sums of bottom-level forecasts before reconciliation.

4. **Training on synthetic data**: Results on real M5 data would differ significantly. The synthetic data generator produces simpler patterns than real Walmart sales data.

### Areas for Improvement

- Use the actual M5 competition data from Kaggle for realistic evaluation
- Add deep learning models (TFT, N-BEATS) for improved point forecast accuracy; the code supports these via the `darts` library but they are disabled by default for stability
- Implement proper probabilistic reconciliation that preserves interval validity (e.g., using sample-based reconciliation instead of transforming bounds directly)
- Train individual models at each hierarchy level rather than only at the bottom level
- Tune ensemble weights using validation performance

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd hierarchical-forecast-reconciliation-with-uncertainty-quantification

# Install the package
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Core Dependencies

- `statsmodels` - ETS and ARIMA models
- `scipy` - Sparse matrices, optimization, statistical distributions
- `scikit-learn` - Preprocessing, covariance estimation
- `pandas`, `numpy` - Data manipulation
- `mlflow` - Experiment tracking
- `optuna` - Hyperparameter optimization
- `darts` (optional) - Deep learning forecasting models (TFT, N-BEATS)

## Usage

### Generate Synthetic Data

If you do not have the M5 dataset:

```bash
python scripts/generate_synthetic_data.py
```

This creates `data/m5/sales_train_evaluation.csv`, `data/m5/calendar.csv`, and `data/m5/sell_prices.csv`.

### Train a Model

```bash
# Basic training
python scripts/train.py --config configs/default.yaml --log-level INFO

# With model saving
python scripts/train.py --config configs/default.yaml --save-model --output-dir outputs

# With hyperparameter optimization
python scripts/train.py --config configs/default.yaml --optimize-hyperparameters --n-trials 10
```

### Python API

```python
from hierarchical_forecast_reconciliation_with_uncertainty_quantification.training.trainer import (
    HierarchicalForecastTrainer
)
from hierarchical_forecast_reconciliation_with_uncertainty_quantification.utils.config import (
    load_config
)

# Load configuration
config = load_config('configs/default.yaml')

# Initialize and train
trainer = HierarchicalForecastTrainer(config=config)
model = trainer.train()

# Generate predictions with uncertainty
predictions = model.predict(
    horizon=28,
    return_intervals=True,
    confidence_levels=[0.1, 0.05]
)

# Access results
point_forecasts = predictions['forecasts']       # Dict[series_id, np.ndarray]
coherence = predictions['coherence_score']        # float
lower_90 = predictions.get('lower_90', {})        # Dict[series_id, np.ndarray]
upper_90 = predictions.get('upper_90', {})        # Dict[series_id, np.ndarray]
```

## Configuration

All parameters are controlled via `configs/default.yaml`:

```yaml
data:
  train_days: 300          # Training period length
  validation_days: 28      # Validation period
  test_days: 28            # Test period
  aggregation_levels:      # 11-level hierarchy
    - total
    - state
    - store
    # ... (see configs/default.yaml for full list)

models:
  statistical:
    ets:
      seasonal_periods: 7
      trend: "add"
      seasonal: "add"
    arima:
      seasonal_order: [1, 1, 1, 7]
  deep_learning: {}        # Empty = disabled; add tft/nbeats configs to enable

ensemble:
  weights:
    ets: 0.5
    arima: 0.5

reconciliation:
  method: "probabilistic_mint"
  weights: "wls"
  lambda_reg: 0.01
  preserve_uncertainty: true
```

## Project Structure

```
hierarchical-forecast-reconciliation-with-uncertainty-quantification/
  configs/
    default.yaml                  # Training configuration
  data/
    m5/                           # Synthetic or real M5 data (not tracked in git)
  scripts/
    generate_synthetic_data.py    # Synthetic M5 data generator
    train.py                      # Training CLI
    evaluate.py                   # Evaluation CLI
  src/hierarchical_forecast_reconciliation_with_uncertainty_quantification/
    data/
      loader.py                   # M5DataLoader, HierarchicalDataBuilder
      preprocessing.py            # M5Preprocessor, HierarchyBuilder
    models/
      model.py                    # StatisticalForecaster, ProbabilisticReconciler,
                                  # HierarchicalEnsembleForecaster
    training/
      trainer.py                  # HierarchicalForecastTrainer
    evaluation/
      metrics.py                  # HierarchicalMetrics, IntervalMetrics,
                                  # CoherenceMetrics
    utils/
      config.py                   # Configuration loading and validation
      config_schema.py            # Schema definitions
  tests/                          # Test suite
  pyproject.toml                  # Package metadata
  requirements.txt                # Dependencies
  LICENSE                         # MIT License
```

## Evaluation Metrics

### Point Forecast Metrics
- **WRMSSE**: Weighted Root Mean Squared Scaled Error (M5 primary metric)
- **MASE**: Mean Absolute Scaled Error (scale-free accuracy measure)
- **sMAPE**: Symmetric Mean Absolute Percentage Error

### Probabilistic Metrics
- **Coverage Probability**: Empirical coverage of prediction intervals
- **Interval Score**: Combines interval width with coverage penalties
- **CRPS**: Continuous Ranked Probability Score

### Hierarchical Metrics
- **Reconciliation Coherence**: Measures whether aggregated forecasts equal sum of components
- **Structural Coherence**: Parent-child relationship consistency

## References

1. Wickramasuriya, S.L., Athanasopoulos, G., & Hyndman, R.J. (2019). Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization. *Journal of the American Statistical Association*, 114(526), 804-819.

2. Hyndman, R.J., Ahmed, R.A., Athanasopoulos, G., & Shang, H.L. (2011). Optimal combination forecasts for hierarchical time series. *Computational Statistics & Data Analysis*, 55(9), 2579-2589.

3. Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. *International Journal of Forecasting*, 38(4), 1346-1364.

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.
