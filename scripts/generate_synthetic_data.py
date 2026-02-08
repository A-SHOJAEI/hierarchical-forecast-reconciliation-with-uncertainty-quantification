#!/usr/bin/env python3
"""
Generate synthetic M5-like data for the hierarchical forecast reconciliation project.

This creates realistic synthetic retail sales data mimicking the structure of the
M5 Walmart competition dataset, with weekly seasonality, trends, and random noise.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import datetime

np.random.seed(42)

# Configuration
N_ITEMS = 50  # Reduced for tractable training
N_DAYS = 365
START_DATE = datetime.date(2016, 1, 29)

# M5-like hierarchy
STATES = ["CA", "TX", "WI"]
STORES = {
    "CA": ["CA_1", "CA_2", "CA_3", "CA_4"],
    "TX": ["TX_1", "TX_2", "TX_3"],
    "WI": ["WI_1", "WI_2", "WI_3"],
}
CATEGORIES = ["HOBBIES", "HOUSEHOLD", "FOODS"]
DEPARTMENTS = {
    "HOBBIES": ["HOBBIES_1", "HOBBIES_2"],
    "HOUSEHOLD": ["HOUSEHOLD_1", "HOUSEHOLD_2"],
    "FOODS": ["FOODS_1", "FOODS_2", "FOODS_3"],
}

ALL_STORES = []
for state, stores in STORES.items():
    ALL_STORES.extend(stores)

ALL_DEPTS = []
for cat, depts in DEPARTMENTS.items():
    ALL_DEPTS.extend(depts)


def generate_item_sales(n_days, base_level, trend_slope, weekly_pattern, noise_std):
    """Generate a single item's daily sales with realistic patterns."""
    t = np.arange(n_days, dtype=float)

    # Trend component
    trend = base_level + trend_slope * t / n_days

    # Weekly seasonality (day 0 = Friday in M5)
    weekly = np.array([weekly_pattern[i % 7] for i in range(n_days)])

    # Monthly seasonality (subtle)
    monthly = 0.05 * base_level * np.sin(2 * np.pi * t / 30.4)

    # Random noise
    noise = np.random.normal(0, noise_std, n_days)

    # Combine
    sales = trend * weekly + monthly + noise

    # Ensure non-negative integer sales
    sales = np.maximum(0, np.round(sales)).astype(int)

    return sales


def generate_sales_data():
    """Generate the sales_train_evaluation.csv file."""
    print("Generating sales_train_evaluation.csv...")

    rows = []
    item_counter = 0

    for cat, depts in DEPARTMENTS.items():
        for dept in depts:
            # Determine how many items per department
            items_per_dept = max(2, N_ITEMS // len(ALL_DEPTS))
            for item_idx in range(items_per_dept):
                item_counter += 1
                if item_counter > N_ITEMS:
                    break

                item_id = f"{dept}_{item_counter:03d}"

                # Assign to a random store
                for store in ALL_STORES:
                    state = store.split("_")[0]

                    # Generate sales parameters
                    base_level = np.random.uniform(1, 15)
                    trend_slope = np.random.uniform(-2, 3)
                    noise_std = np.random.uniform(0.5, 3.0)

                    # Weekly pattern: higher sales on weekends
                    weekly_base = np.random.uniform(0.6, 1.0, 7)
                    weekly_base[5] = np.random.uniform(1.1, 1.5)  # Saturday
                    weekly_base[6] = np.random.uniform(1.0, 1.4)  # Sunday
                    weekly_pattern = weekly_base / weekly_base.mean()

                    # Generate sales
                    sales = generate_item_sales(
                        N_DAYS, base_level, trend_slope, weekly_pattern, noise_std
                    )

                    # Create row
                    row_id = f"{item_id}_{store}_evaluation"
                    row = {
                        "id": row_id,
                        "item_id": item_id,
                        "dept_id": dept,
                        "cat_id": cat,
                        "store_id": store,
                        "state_id": state,
                    }
                    # Add daily sales columns
                    for d in range(1, N_DAYS + 1):
                        row[f"d_{d}"] = sales[d - 1]

                    rows.append(row)

            if item_counter > N_ITEMS:
                break
        if item_counter > N_ITEMS:
            break

    df = pd.DataFrame(rows)
    print(f"  Generated {len(df)} item-store combinations, {N_DAYS} days each")
    return df


def generate_calendar():
    """Generate the calendar.csv file."""
    print("Generating calendar.csv...")

    dates = [START_DATE + datetime.timedelta(days=i) for i in range(N_DAYS)]

    calendar_rows = []
    weekday_names = [
        "Saturday",
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
    ]

    # Simple events
    events = {
        (1, 1): ("NewYear", "National"),
        (2, 14): ("ValentinesDay", "Cultural"),
        (5, 30): ("MemorialDay", "National"),
        (7, 4): ("IndependenceDay", "National"),
        (9, 5): ("LaborDay", "National"),
        (11, 24): ("Thanksgiving", "National"),
        (12, 25): ("Christmas", "National"),
    }

    for i, date in enumerate(dates):
        d_label = f"d_{i+1}"
        wday = date.weekday()
        # M5 uses Saturday=1, Sunday=2, etc.
        wday_m5 = (wday + 2) % 7 + 1
        wm_yr_wk = date.isocalendar()[0] * 100 + date.isocalendar()[1]

        event_name = ""
        event_type = ""
        key = (date.month, date.day)
        if key in events:
            event_name, event_type = events[key]

        calendar_rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "wm_yr_wk": wm_yr_wk,
                "weekday": weekday_names[wday],
                "wday": wday_m5,
                "month": date.month,
                "year": date.year,
                "d": d_label,
                "event_name_1": event_name if event_name else np.nan,
                "event_type_1": event_type if event_type else np.nan,
            }
        )

    df = pd.DataFrame(calendar_rows)
    print(f"  Generated {len(df)} calendar rows")
    return df


def generate_prices(sales_df, calendar_df):
    """Generate the sell_prices.csv file."""
    print("Generating sell_prices.csv...")

    # Get unique item-store combinations
    item_store = sales_df[["item_id", "store_id"]].drop_duplicates()

    # Get unique weeks
    weeks = sorted(calendar_df["wm_yr_wk"].unique())

    price_rows = []
    for _, row in item_store.iterrows():
        item_id = row["item_id"]
        store_id = row["store_id"]

        # Base price for this item-store
        base_price = np.random.uniform(0.5, 20.0)

        for wk in weeks:
            # Add small random price variation
            price_variation = np.random.normal(0, 0.02) * base_price
            price = max(0.01, round(base_price + price_variation, 2))

            price_rows.append(
                {
                    "store_id": store_id,
                    "item_id": item_id,
                    "wm_yr_wk": wk,
                    "sell_price": price,
                }
            )

    df = pd.DataFrame(price_rows)
    print(f"  Generated {len(df)} price rows")
    return df


def main():
    """Main function to generate all synthetic data files."""
    data_dir = Path(__file__).parent.parent / "data" / "m5"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    sales_df = generate_sales_data()
    calendar_df = generate_calendar()
    prices_df = generate_prices(sales_df, calendar_df)

    # Save files
    sales_path = data_dir / "sales_train_evaluation.csv"
    calendar_path = data_dir / "calendar.csv"
    prices_path = data_dir / "sell_prices.csv"

    sales_df.to_csv(sales_path, index=False)
    calendar_df.to_csv(calendar_path, index=False)
    prices_df.to_csv(prices_path, index=False)

    print(f"\nSaved files to {data_dir}:")
    print(f"  sales_train_evaluation.csv: {sales_df.shape}")
    print(f"  calendar.csv: {calendar_df.shape}")
    print(f"  sell_prices.csv: {prices_df.shape}")

    # Print summary statistics
    day_cols = [c for c in sales_df.columns if c.startswith("d_")]
    sales_values = sales_df[day_cols].values
    print(f"\nSales statistics:")
    print(f"  Mean daily sales: {sales_values.mean():.2f}")
    print(f"  Std daily sales: {sales_values.std():.2f}")
    print(f"  Max daily sales: {sales_values.max()}")
    print(f"  Zero ratio: {(sales_values == 0).mean():.2%}")
    print(f"  Items: {sales_df['item_id'].nunique()}")
    print(f"  Stores: {sales_df['store_id'].nunique()}")
    print(f"  States: {sales_df['state_id'].nunique()}")
    print(f"  Categories: {sales_df['cat_id'].nunique()}")
    print(f"  Departments: {sales_df['dept_id'].nunique()}")


if __name__ == "__main__":
    main()
