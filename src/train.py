# src/train.py

import os
import sys

from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # type: ignore

sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import (
    validate_dataframe,
    clean_data,
    encode_categoricals,
    check_data_quality,
)

from utils import (
    load_config,
    load_data,
    get_required_columns,
    prepare_features,
    save_json,
    save_model,
    check_thresholds,
)


def train_model(config=None):
    if config is None:
        config = load_config()

    df = load_data(config["data_url"])
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    required = get_required_columns(config)

    validate_dataframe(df, required, config["target"])

    quality = check_data_quality(df, config["numeric_columns"])
    print(
        f"Data quality: {quality['total_nulls']} nulls, "
        f"{quality['duplicate_rows']} duplicates"
    )

    df = clean_data(
        df,
        config["numeric_columns"],
        config["categorical_columns"],
    )

    df = encode_categoricals(df, config["categorical_columns"])

    X_train, X_test, y_train, y_test = prepare_features(df, config)

    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        random_state=config["random_state"],
    )

    print("Training random forest...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": X_train.shape[1],
    }

    print("\nTraining Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    save_model(model, "models/model.pkl")
    save_json(metrics, "metrics/results.json")

    print("\nModel saved to models/model.pkl")
    print("Metrics saved to metrics/results.json")

    return metrics


if __name__ == "__main__":
    CONFIG = load_config()
    metrics = train_model(CONFIG)
    check_thresholds(metrics, CONFIG, exit_on_fail=True)

    print("\nAll thresholds passed!")