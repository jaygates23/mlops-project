# src/utils.py

import os
import json
import pickle
import sys

import pandas as pd  # type: ignore
import yaml  # type: ignore

from sklearn.model_selection import train_test_split  # type: ignore


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path=None):
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)

        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(path):
    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"([a-z])([A-Z])", r"\1_\2", regex=True)
        .str.lower()
    )

    for col in df.select_dtypes(include="object").columns:
        df[col] = (
            df[col]
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

    return df


def get_required_columns(config):
    required = config["numeric_columns"] + config["categorical_columns"]

    if config["target"] not in required:
        required.append(config["target"])

    return required


def prepare_features(df, config):
    target = config["target"]

    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_model(model, path="models/model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path="models/model.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError("Model file not found. Run src/train.py first.")

    with open(path, "rb") as f:
        return pickle.load(f)


def check_thresholds(metrics, config, exit_on_fail=False):
    min_accuracy = config["performance_thresholds"]["min_accuracy"]
    min_f1 = config["performance_thresholds"]["min_f1"]

    failed = False

    if metrics["accuracy"] < min_accuracy:
        print(f"\nFAILED: Accuracy {metrics['accuracy']} below threshold {min_accuracy}")
        failed = True

    if metrics["f1_score"] < min_f1:
        print(f"\nFAILED: F1 {metrics['f1_score']} below threshold {min_f1}")
        failed = True

    if failed and exit_on_fail:
        sys.exit(1)

    return not failed