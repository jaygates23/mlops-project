# src/train.py

import os
import sys

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

    target = config["target"]

    df = load_data(config["data_url"])
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    required = get_required_columns(config)
    validate_dataframe(df, required, target)

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

    df[target] = df[target].map({"no": 0, "yes": 1})

    if df[target].isna().any():
        raise ValueError("Target encoding failed. Expected values: 'yes' and 'no'.")

    df = encode_categoricals(df, config["categorical_columns"])

    X_train, X_test, y_train, y_test = prepare_features(df, config)

    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=config["random_state"],
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=10,
            class_weight="balanced",
            random_state=config["random_state"],
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=config["model"]["n_estimators"],
            max_depth=config["model"]["max_depth"],
            class_weight="balanced",
            random_state=config["random_state"],
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=12,
            class_weight="balanced",
            random_state=config["random_state"],
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=config["random_state"],
        ),
        "adaboost": AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.5,
            random_state=config["random_state"],
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
        ),
        "gaussian_nb": GaussianNB(),
        "svc_rbf": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=config["random_state"],
        ),
    }

    results = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "n_features": X_train.shape[1],
        }

        results[model_name] = {
            "model": model,
            "metrics": metrics,
        }

        print(f"  Accuracy:  {metrics['accuracy']}")
        print(f"  Precision: {metrics['precision']}")
        print(f"  Recall:    {metrics['recall']}")
        print(f"  F1 Score:  {metrics['f1_score']}")

    best_model_name = max(
        results,
        key=lambda name: results[name]["metrics"]["f1_score"],
    )

    best_model = results[best_model_name]["model"]
    best_metrics = results[best_model_name]["metrics"]
    best_metrics["best_model"] = best_model_name

    print(f"\nBest model: {best_model_name}")
    print(f"Best F1 Score: {best_metrics['f1_score']}")

    save_model(best_model, "models/model.pkl")
    save_json(best_metrics, "metrics/results.json")
    save_json(
        {name: result["metrics"] for name, result in results.items()},
        "metrics/all_model_results.json",
    )

    print("\nBest model saved to models/model.pkl")
    print("Best metrics saved to metrics/results.json")
    print("All model results saved to metrics/all_model_results.json")

    return best_metrics


if __name__ == "__main__":
    CONFIG = load_config()
    metrics = train_model(CONFIG)
    check_thresholds(metrics, CONFIG, exit_on_fail=True)

    print("\nAll thresholds passed!")