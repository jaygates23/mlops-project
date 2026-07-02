# src/monitor_drift.py

import os
import sys
import subprocess

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from utils import load_config, load_data


from evidently import Report
from evidently.presets import DataDriftPreset


def load_or_create_production_data(config):
    production_path = "data/production/simulated_production_data.csv"

    if not os.path.exists(production_path):
        print("Production data not found. Creating simulated production data...")
        subprocess.run(
            [sys.executable, "src/create_drift.py"],
            check=True,
        )

    return pd.read_csv(production_path)


def run_drift_monitoring(config=None):
    if config is None:
        config = load_config()

    drift_threshold = config.get("drift_threshold", 0.50)

    reference_data = load_data(config["data_url"])
    production_data = load_or_create_production_data(config)

    common_columns = [
        col for col in reference_data.columns
        if col in production_data.columns
    ]

    reference_data = reference_data[common_columns]
    production_data = production_data[common_columns]

    print(f"Reference rows: {len(reference_data)}")
    print(f"Production rows: {len(production_data)}")
    print(f"Columns checked: {len(common_columns)}")

    report = Report([
        DataDriftPreset(),
    ])

    result = report.run(
        reference_data=reference_data,
        current_data=production_data,
    )

    os.makedirs("reports", exist_ok=True)
    report_path = "reports/data_drift_report.html"
    result.save_html(report_path)

    drift_summary = result.dict()

    drifted_features = []
    drift_share = 0.0

    for metric in drift_summary.get("metrics", []):
        metric_result = metric.get("result", {})

        if "drift_share" in metric_result:
            drift_share = metric_result["drift_share"]

        if "drift_by_columns" in metric_result:
            for feature, values in metric_result["drift_by_columns"].items():
                if values.get("drift_detected") is True:
                    drifted_features.append(feature)

    print("\nDrift Monitoring Summary")
    print("------------------------")
    print(f"Overall drift share: {drift_share:.4f}")
    print(f"Drift threshold: {drift_threshold}")
    print(f"Number of drifted features: {len(drifted_features)}")

    if drifted_features:
        print("\nDrifted features:")
        for feature in drifted_features:
            print(f"  - {feature}")
    else:
        print("\nNo drifted features detected.")

    print(f"\nHTML report saved to: {report_path}")

    if drift_share > drift_threshold:
        print("\nFAILED: Drift exceeds configured threshold.")
        sys.exit(1)

    print("\nPASSED: Drift is within acceptable threshold.")

    return {
        "drift_share": drift_share,
        "drift_threshold": drift_threshold,
        "drifted_features": drifted_features,
        "report_path": report_path,
    }


if __name__ == "__main__":
    run_drift_monitoring()