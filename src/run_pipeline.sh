#!/bin/bash

set -e  # Exit immediately if any command fails

echo "======================================="
echo "Starting MLOps Pipeline"
echo "======================================="

echo ""
echo "1. Training model..."
python train.py

echo ""
echo "2. Evaluating model..."
python evaluate.py

echo ""
echo "3. Running MLflow experiments..."
python compare_experiments.py

echo ""
echo "4. Creating simulated production data..."
python create_drift.py

echo ""
echo "5. Monitoring data drift..."
python monitor_drift.py

echo ""
echo "======================================="
echo "Pipeline completed successfully!"
echo "======================================="