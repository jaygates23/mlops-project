# src/create_drift.py

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from utils import load_config, load_data


def create_simulated_production_data(config=None, drift_strength=0.15):
    """
    Create simulated production data by applying controlled drift
    to selected numeric and categorical features.
    """

    if config is None:
        config = load_config()

    df = load_data(config["data_url"])
    production_df = df.copy()

    np.random.seed(config["random_state"])

    # Numeric drift
    numeric_drift_columns = [
        "monthly_income",
        "daily_rate",
        "hourly_rate",
        "distance_from_home",
        "years_at_company",
    ]

    for col in numeric_drift_columns:
        if col in production_df.columns:
            noise = np.random.normal(
                loc=1 + drift_strength,
                scale=0.05,
                size=len(production_df),
            )
            production_df[col] = production_df[col] * noise

            if pd.api.types.is_integer_dtype(df[col]):
                production_df[col] = production_df[col].round().astype(int)

    # Categorical drift
    if "over_time" in production_df.columns:
        overtime_indices = production_df.sample(
            frac=drift_strength,
            random_state=config["random_state"],
        ).index

        production_df.loc[overtime_indices, "over_time"] = "yes"

    if "business_travel" in production_df.columns:
        travel_indices = production_df.sample(
            frac=drift_strength,
            random_state=config["random_state"] + 1,
        ).index

        production_df.loc[travel_indices, "business_travel"] = "travel_frequently"

    # Simulate slight target drift
    if config["target"] in production_df.columns:
        target_indices = production_df.sample(
            frac=drift_strength / 2,
            random_state=config["random_state"] + 2,
        ).index

        production_df.loc[target_indices, config["target"]] = "yes"

    os.makedirs("data/production", exist_ok=True)

    output_path = "data/production/simulated_production_data.csv"
    production_df.to_csv(output_path, index=False)

    print(f"Simulated production data saved to {output_path}")
    print(f"Rows: {len(production_df)}")
    print(f"Columns: {len(production_df.columns)}")

    return production_df


if __name__ == "__main__":
    create_simulated_production_data()