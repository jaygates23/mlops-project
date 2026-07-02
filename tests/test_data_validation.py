# tests/test_data_validation.py

import os
import sys
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, SRC_PATH)

from utils import load_config, load_data


def test_expected_columns_are_present():
    config = load_config("configs/config.yaml")
    df = load_data(config["t_data_url"])

    expected_columns = (
        config["numeric_columns"]
        + config["categorical_columns"]
    )

    for col in expected_columns:
        assert col in df.columns


def test_target_contains_only_expected_values():
    config = load_config("configs/config.yaml")
    df = load_data(config["t_data_url"])

    target = config["target"]

    expected_values = {"yes", "no"}
    actual_values = set(df[target].unique())

    assert actual_values.issubset(expected_values)


def test_numeric_features_are_within_expected_ranges():
    config = load_config("configs/config.yaml")
    df = load_data(config["t_data_url"])

    numeric_columns = config["numeric_columns"]

    for col in numeric_columns:
        assert df[col].notna().all()
        assert df[col].min() >= 0


def test_dataset_has_rows():
    config = load_config("configs/config.yaml")
    df = load_data(config["t_data_url"])

    assert len(df) > 0