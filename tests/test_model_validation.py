# tests/test_model_validation.py

import sys

sys.path.insert(0, "src")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from preprocessing import clean_data, encode_categoricals
from utils import load_config, load_data, prepare_features


def test_model_predictions_have_correct_shape_and_type():
    config = load_config("configs/config.yaml")
    df = load_data(config["data_url"])

    df = clean_data(
        df,
        config["numeric_columns"],
        config["categorical_columns"],
    )

    df = encode_categoricals(df, config["categorical_columns"])

    X_train, X_test, y_train, y_test = prepare_features(df, config)

    model = RandomForestClassifier(
        n_estimators=10,
        max_depth=5,
        random_state=config["random_state"],
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert len(preds) == len(y_test)
    assert preds.ndim == 1


def test_model_meets_minimum_performance_threshold():
    config = load_config("configs/config.yaml")
    df = load_data(config["data_url"])

    df = clean_data(
        df,
        config["numeric_columns"],
        config["categorical_columns"],
    )

    df = encode_categoricals(df, config["categorical_columns"])

    X_train, X_test, y_train, y_test = prepare_features(df, config)

    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        random_state=config["random_state"],
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    assert accuracy >= config["performance_thresholds"]["min_accuracy"]
    assert f1 >= config["performance_thresholds"]["min_f1"]