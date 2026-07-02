# src/compare_experiments.py

import os
import sys
import copy

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import clean_data, encode_categoricals
from utils import load_config, load_data, prepare_features, save_json


def run_single_experiment(config, experiment_name, run_name):
    mlflow.set_experiment(experiment_name)

    df = load_data(config["data_url"])

    df = clean_data(
        df,
        config["numeric_columns"],
        config["categorical_columns"],
    )

    df = encode_categoricals(df, config["categorical_columns"])

    X_train, X_test, y_train, y_test = prepare_features(df, config)

    model_type = config["model"]["type"]

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=config["model"]["n_estimators"],
            max_depth=config["model"]["max_depth"],
            random_state=config["random_state"],
        )

    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=config["model"]["n_estimators"],
            learning_rate=config["model"]["learning_rate"],
            max_depth=config["model"]["max_depth"],
            random_state=config["random_state"],
        )

    elif model_type == "logistic_regression":
        model = LogisticRegression(
            max_iter=config["model"]["max_iter"],
            random_state=config["random_state"],
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("test_size", config["test_size"])
        mlflow.log_param("random_state", config["random_state"])
        mlflow.log_param("data_version", config.get("data_version", "local_dvc_dataset"))

        for key, value in config["model"].items():
            mlflow.log_param(key, value)

        model.fit(X_train, y_train)

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

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        return metrics


def run_five_experiments():
    base_config = load_config("/Users/tjscott23/Documents/MLOps_PL_project/configs/configs/config.yaml")

    experiment_name = "employee_attrition_experiments"

    experiments = [
        {
            "run_name": "random_forest_baseline",
            "model": {
                "type": "random_forest",
                "n_estimators": 100,
                "max_depth": 10,
            },
        },
        {
            "run_name": "random_forest_deeper",
            "model": {
                "type": "random_forest",
                "n_estimators": 200,
                "max_depth": 15,
            },
        },
        {
            "run_name": "random_forest_shallow",
            "model": {
                "type": "random_forest",
                "n_estimators": 50,
                "max_depth": 5,
            },
        },
        {
            "run_name": "gradient_boosting_baseline",
            "model": {
                "type": "gradient_boosting",
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
            },
        },
        {
            "run_name": "logistic_regression_baseline",
            "model": {
                "type": "logistic_regression",
                "max_iter": 1000,
            },
        },
    ]

    results = []

    for exp in experiments:
        config = copy.deepcopy(base_config)
        config["model"] = exp["model"]

        print(f"\nRunning experiment: {exp['run_name']}")
        metrics = run_single_experiment(
            config=config,
            experiment_name=experiment_name,
            run_name=exp["run_name"],
        )

        results.append({
            "run_name": exp["run_name"],
            **metrics,
        })

    results_df = pd.DataFrame(results)
    os.makedirs("metrics", exist_ok=True)
    results_df.to_csv("metrics/experiment_results.csv", index=False)

    return experiment_name, results_df


def find_best_experiment(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_score DESC"],
    )

    if runs.empty:
        raise ValueError("No MLflow runs found.")

    best_run = runs.iloc[0]

    best_summary = {
        "run_id": best_run["run_id"],
        "run_name": best_run.get("tags.mlflow.runName", "unknown"),
        "model_type": best_run.get("params.model_type", "unknown"),
        "accuracy": best_run.get("metrics.accuracy"),
        "precision": best_run.get("metrics.precision"),
        "recall": best_run.get("metrics.recall"),
        "f1_score": best_run.get("metrics.f1_score"),
        "roc_auc": best_run.get("metrics.roc_auc"),
    }

    save_json(best_summary, "metrics/best_experiment.json")

    print("\nBest Experiment")
    print("----------------")
    for key, value in best_summary.items():
        print(f"{key}: {value}")

    return best_summary


if __name__ == "__main__":
    experiment_name, results_df = run_five_experiments()

    print("\nAll Experiment Results")
    print("----------------------")
    print(results_df)

    find_best_experiment(experiment_name)