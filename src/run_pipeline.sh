#!/bin/bash

set -e

echo "======================================="
echo "Starting MLOps Pipeline"
echo "======================================="

echo ""
echo "Running unit tests..."
pytest tests/ -v

echo ""
echo "Training model..."
python src/train.py

echo ""
echo "Evaluating model..."
python src/evaluate.py

echo ""
echo "Running MLflow experiments..."
python src/compare_experiments.py

echo ""
echo "Creating simulated production data..."
python src/create_drift.py

echo ""
echo "Running drift monitoring..."
python src/monitor_drift.py

echo ""
echo "======================================="
echo "All tasks completed successfully!"
echo "======================================="