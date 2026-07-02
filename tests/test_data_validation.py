# tests/test_data_validation.py

import sys

sys.path.insert(0, "src")

from utils import load_config, load_data


def test_expected_columns_are_present():
    config = load_config("configs/config.yaml")
    df = load_data(config["data_url"])

    expected_columns = (
        config["numeric_columns"]
        + config["categorical_columns"]
    )

    for col in expected_columns:
        assert col in df.columns


def test_target_contains_only_expected_values():
    config = load_config("configs/config.yaml")
    df = load_data(config["data_url"])

    target = config["target"]

    expected_values = {"yes", "no"}
    actual_values = set(df[target].unique())

    assert actual_values.issubset(expected_values)


def test_numeric_features_are_within_expected_ranges():
    config = load_config("configs/config.yaml")
    df = load_data(config["data_url"])

    numeric_columns = config["numeric_columns"]

    for col in numeric_columns:
        assert df[col].notna().all()
        assert df[col].min() >= 0


def test_dataset_has_rows():
    config = load_config("configs/config.yaml")
    df = load_data(config["data_url"])

    assert len(df) > 0