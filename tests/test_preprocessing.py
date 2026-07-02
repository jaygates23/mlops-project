# tests/test_preprocessing.py

import sys
import pandas as pd
import pytest

sys.path.insert(0, "src")

from preprocessing import (
    validate_dataframe,
    clean_data,
    encode_categoricals,
    check_data_quality,
)
from utils import load_config, load_data


@pytest.fixture
def config():
    return load_config("configs/config.yaml")


@pytest.fixture
def real_data(config):
    return load_data(config["data_url"])


class TestValidateDataframe:

    def test_valid_dataframe_passes(self, real_data, config):
        required_columns = config["numeric_columns"] + config["categorical_columns"]

        result = validate_dataframe(
            real_data,
            required_columns=required_columns,
            target_column=config["target"],
        )

        assert result is True

    def test_missing_column_raises(self, real_data, config):
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_dataframe(
                real_data,
                required_columns=["age", "fake_column"],
                target_column=config["target"],
            )

    def test_missing_target_raises(self, real_data):
        with pytest.raises(ValueError, match="Target column"):
            validate_dataframe(
                real_data,
                required_columns=["age"],
                target_column="fake_target",
            )

    def test_empty_dataframe_raises(self):
        empty_df = pd.DataFrame(columns=["age", "attrition"])

        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(
                empty_df,
                required_columns=["age"],
                target_column="attrition",
            )


class TestCleanData:

    def test_fills_numeric_nulls(self, real_data, config):
        df = real_data.copy()
        df.loc[0, "age"] = None

        result = clean_data(
            df,
            numeric_columns=config["numeric_columns"],
            categorical_columns=[],
        )

        assert result["age"].isna().sum() == 0

    def test_does_not_modify_original(self, real_data, config):
        df = real_data.copy()
        df.loc[0, "age"] = None

        original_nulls = df["age"].isna().sum()

        clean_data(
            df,
            numeric_columns=["age"],
            categorical_columns=[],
        )

        assert df["age"].isna().sum() == original_nulls

    def test_numeric_values_remain_valid(self, real_data, config):
        result = clean_data(
            real_data,
            numeric_columns=config["numeric_columns"],
            categorical_columns=[],
        )

        assert result[config["numeric_columns"]].isna().sum().sum() == 0

    def test_preserves_row_count_after_cleaning(self, real_data, config):
        result = clean_data(
            real_data,
            numeric_columns=config["numeric_columns"],
            categorical_columns=config["categorical_columns"],
        )

        assert len(result) == len(real_data)


class TestEncodeCategoricals:

    def test_creates_dummy_columns(self, real_data):
        result = encode_categoricals(real_data, ["gender"])

        assert "gender" not in result.columns
        assert any("gender" in col for col in result.columns)

    def test_preserves_row_count(self, real_data):
        result = encode_categoricals(
            real_data,
            ["gender", "department", "over_time"],
        )

        assert len(result) == len(real_data)

    def test_target_column_can_be_encoded(self, real_data):
        result = encode_categoricals(real_data, ["attrition"])

        assert "attrition" not in result.columns
        assert any("attrition" in col for col in result.columns)


class TestDataQuality:

    def test_counts_total_rows(self, real_data, config):
        report = check_data_quality(real_data, config["numeric_columns"])

        assert report["total_rows"] == len(real_data)

    def test_counts_total_nulls(self, real_data, config):
        report = check_data_quality(real_data, config["numeric_columns"])

        assert "total_nulls" in report
        assert report["total_nulls"] >= 0

    def test_counts_duplicate_rows(self, real_data, config):
        report = check_data_quality(real_data, config["numeric_columns"])

        assert "duplicate_rows" in report
        assert report["duplicate_rows"] >= 0

    def test_reports_numeric_ranges(self, real_data, config):
        report = check_data_quality(real_data, ["age"])

        assert "age_min" in report
        assert "age_max" in report
        assert report["age_min"] >= 0
        assert report["age_max"] >= report["age_min"]