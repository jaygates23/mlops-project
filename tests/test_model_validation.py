# tests/test_model_validation.py
import os
import sys
import pandas as pd
import pytest
from sklearn.naive_bayes import GaussianNB


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, SRC_PATH)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from preprocessing import clean_data, encode_categoricals
from utils import load_config, load_data, prepare_features


def test_model_predictions_have_correct_shape_and_type():
    config = load_config("configs/config.yaml")
    df = load_data(config["t_data_url"])

    df = clean_data(
        df,
        config["numeric_columns"],
        config["categorical_columns"],
    )
    df[config["target"]] = df[config["target"]].map({"no": 0, "yes": 1})

    df = encode_categoricals(df, config["categorical_columns"])

    X_train, X_test, y_train, y_test = prepare_features(df, config)

    model = GaussianNB()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert len(preds) == len(y_test)
    assert preds.ndim == 1


def test_model_meets_minimum_performance_threshold():
    config = load_config("configs/config.yaml")
    df = load_data(config["t_data_url"])

    df = clean_data(
        df,
        config["numeric_columns"],
        config["categorical_columns"],
    )
    df[config["target"]] = df[config["target"]].map({"no": 0, "yes": 1})

    df = encode_categoricals(df, config["categorical_columns"])

    X_train, X_test, y_train, y_test = prepare_features(df, config)

    model = GaussianNB()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    assert accuracy >= config["performance_thresholds"]["min_accuracy"]
    assert f1 >= config["performance_thresholds"]["min_f1"]