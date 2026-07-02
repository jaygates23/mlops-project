# src/evaluate.py

import os
import sys

from sklearn.metrics import ( # type: ignore
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
) 

sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import validate_dataframe, clean_data, encode_categoricals

from utils import (
    load_config,
    load_data,
    get_required_columns,
    prepare_features,
    load_model,
    save_json,
    check_thresholds,
)


def evaluate_model(config=None):
    if config is None:
        config = load_config()

    df = load_data(config["data_url"])

    required = get_required_columns(config)

    validate_dataframe(df, required, config["target"])

    df = clean_data(
        df,
        config["numeric_columns"],
        config["categorical_columns"],
    )
    target = config["target"]

    df[target] = df[target].map({"no": 0, "yes": 1})

    df = encode_categoricals(df, config["categorical_columns"])

    _, X_test, _, y_test = prepare_features(df, config)

    model = load_model("models/model.pkl")

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = round(roc_auc_score(y_test, y_proba), 4)

    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred).tolist()

    save_json(metrics, "metrics/evaluation_metrics.json")
    save_json(report, "metrics/classification_report.json")
    save_json(matrix, "metrics/confusion_matrix.json")

    print("\nEvaluation Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print("\nConfusion Matrix:")
    print(matrix)

    check_thresholds(metrics, config, exit_on_fail=True)

    print("\nEvaluation passed all thresholds.")

    return metrics


if __name__ == "__main__":
    evaluate_model()