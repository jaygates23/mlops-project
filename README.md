# Employee Attrition Prediction MLOps Pipeline

## Project Overview

This project demonstrates an end-to-end **Machine Learning Operations (MLOps)** pipeline for predicting employee attrition using the IBM HR Analytics Employee Attrition dataset from Kaggle. The focus of this project is not on building the most accurate predictive model, but on implementing production-ready machine learning practices including version control, experiment tracking, automated testing, continuous integration, and data drift monitoring.

The pipeline covers the complete machine learning lifecycle from data preprocessing to model evaluation and monitoring while following industry best practices.

---

## Objectives

The primary objective is to build a reproducible and automated machine learning workflow that includes:

- Data preprocessing and validation
- Multiple machine learning model comparisons
- Experiment tracking with MLflow
- Data versioning using DVC
- Automated testing with pytest
- Continuous Integration using GitHub Actions
- Data drift monitoring using Evidently
- Configuration management using YAML
- Reproducible model training and evaluation

---

## Dataset

**IBM HR Analytics Employee Attrition & Performance**

- **Task:** Binary Classification
- **Target Variable:** `attrition`
- **Records:** 1,470 employees
- **Features:** 30 employee demographic, compensation, and performance attributes

Example features include:

- Age
- Business Travel
- Department
- Education
- Job Role
- Monthly Income
- Over Time
- Years at Company
- Work Life Balance

---

# Project Structure

```
mlops-project/
│
├── configs/
│   └── config.yaml
│
├── Data/
│   ├── IBM_HR_Attrition.csv.dvc
│   └── sample_attrition.csv
│
├── src/
│   ├── preprocessing.py
│   ├── utils.py
│   ├── train.py
│   ├── evaluate.py
│   ├── compare_experiments.py
│   ├── create_drift.py
│   └── monitor_drift.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_data_validation.py
│   └── test_model_validation.py
│
├── metrics/
├── models/
├── reports/
├── mlruns/
│
├── requirements.txt
├── run_pipeline.sh
├── README.md
└── .github/
    └── workflows/
        └── mlworkflow.yml
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- MLflow
- DVC
- Evidently
- PyTest
- Git
- GitHub Actions
- YAML

---

# Installation

Clone the repository

```bash
git clone https://github.com/jaygates23/mlops-project.git
```

Move into the project directory

```bash
cd mlops-project
```

Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate the environment

### macOS/Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Download Dataset

If using DVC

```bash
dvc pull
```

or place the dataset inside

```
Data/
```

---

# Configuration

Project settings are stored in

```
configs/config.yaml
```

Configuration includes

- dataset location
- train/test split
- random seed
- model hyperparameters
- performance thresholds
- drift thresholds

---

# Train the Model

Run

```bash
python src/train.py
```

Training will

- load the dataset
- clean the data
- encode categorical variables
- compare multiple machine learning models
- save the best model
- save evaluation metrics

Outputs

```
models/model.pkl

metrics/results.json

metrics/all_model_results.json
```

---

# Evaluate the Model

Run

```bash
python src/evaluate.py
```

Outputs

```
metrics/evaluation_metrics.json

metrics/classification_report.json

metrics/confusion_matrix.json
```

---

# Run MLflow Experiments

Execute

```bash
python src/compare_experiments.py
```

The script

- trains multiple models
- logs hyperparameters
- logs evaluation metrics
- stores trained models
- identifies the best experiment

View experiments

```bash
mlflow ui
```

Open

```
http://127.0.0.1:5000
```

---

# Simulate Production Data

```bash
python src/create_drift.py
```

Creates

```
data/production/simulated_production_data.csv
```

---

# Monitor Data Drift

```bash
python src/monitor_drift.py
```

Outputs

```
reports/data_drift_report.html
```

The monitoring script

- compares reference and production datasets
- detects feature drift
- reports overall drift percentage
- exits with an error if drift exceeds the configured threshold

---

# Run Tests

Execute all tests

```bash
pytest tests/ -v
```

The test suite includes

- preprocessing unit tests
- dataset validation tests
- model validation tests

---

# Run the Complete Pipeline

Execute

```bash
./run_pipeline.sh
```

The pipeline performs

1. Unit testing
2. Model training
3. Model evaluation
4. MLflow experiment tracking
5. Simulated production data generation
6. Drift monitoring

---

# Continuous Integration

GitHub Actions automatically

- installs dependencies
- runs the test suite
- trains the model
- evaluates model performance
- executes MLflow experiments
- runs drift monitoring
- uploads model and metrics as workflow artifacts

The workflow is triggered on

- push to main
- pull requests
- scheduled weekly runs
- manual workflow dispatch

---

# Data Versioning

This project uses **DVC** for dataset versioning.

Dataset files are tracked separately from Git while lightweight `.dvc` pointer files remain under version control.

Retrieve the latest dataset

```bash
dvc pull
```

---

# Experiment Tracking

MLflow tracks

- model type
- hyperparameters
- dataset version
- evaluation metrics
- trained model artifacts

Experiments can be compared using

```bash
python src/compare_experiments.py
```

---

# Model Performance Metrics

Models are evaluated using

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

The highest-performing model is automatically selected and saved.

---
---

# Drift Monitoring Analysis

The project includes an automated drift monitoring pipeline using **Evidently** to compare the training dataset (reference data) against simulated production data. The simulated production dataset introduces controlled changes to several numerical and categorical features to emulate how real-world data distributions may evolve after deployment.

## Which Features Showed Drift and Why?

The simulated production dataset intentionally introduced drift into several employee-related features, including:

- **Monthly Income** – increased using random scaling to simulate salary growth over time.
- **Daily Rate** – adjusted to represent changes in compensation policies.
- **Hourly Rate** – modified to reflect wage adjustments.
- **Distance From Home** – shifted to simulate changes in employee hiring locations or remote work policies.
- **Years at Company** – increased slightly to represent an aging workforce.
- **Over Time** – the proportion of employees working overtime was intentionally increased.
- **Business Travel** – additional employees were assigned frequent travel to simulate changing business requirements.

These changes represent realistic business scenarios that may occur after a model has been deployed into production.

---

## Would This Drift Likely Affect Model Performance?

Yes.

Several of the drifted features are among the most informative predictors of employee attrition. Changes in compensation, tenure, overtime frequency, and travel requirements may alter the relationship between employee characteristics and attrition outcomes.

If these feature distributions continue to shift over time, model predictions may become less reliable because the model was trained on historical data that no longer reflects the current employee population.

The severity of the impact depends on both the magnitude of the drift and the importance of the affected features within the trained model.

---

## Recommended Action

The appropriate response depends on the amount of detected drift.

- **Low drift (< configured threshold):**
  - Continue monitoring.
  - No immediate action is required.

- **Moderate drift:**
  - Investigate the affected features.
  - Determine whether the changes reflect expected business evolution or potential data quality issues.

- **High drift (above configured threshold):**
  - Retrain the model using recent production data.
  - Re-evaluate model performance on updated data.
  - Redeploy the model only after validation confirms acceptable predictive performance.

The automated monitoring script (`src/monitor_drift.py`) exits with a non-zero status code whenever the overall drift share exceeds the configured threshold. This behavior enables CI/CD pipelines to detect significant distribution changes and trigger further investigation or retraining before deploying updated models.

# Future Improvements

Potential enhancements include

- Hyperparameter optimization
- Docker containerization
- FastAPI model deployment
- Kubernetes deployment
- Scheduled model retraining
- Cloud-based MLflow tracking server
- Feature Store integration

---

# Author

**Terrence Scott**

MS Data Science and Artificial Intelligence Candidate

AI | Machine Learning | MLOps | Data Science
