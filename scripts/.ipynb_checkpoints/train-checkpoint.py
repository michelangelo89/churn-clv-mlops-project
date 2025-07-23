#!/usr/bin/env python
# scripts/train.py

import mlflow
import argparse
from xgboost import XGBClassifier

from src.data.load import load_data
from src.features.pipeline import build_pipeline
from src.models.train import train_and_log_model


def main(train_path, test_path):
    # Load data
    X_train, y_train, X_test, y_test = load_data(train_path, test_path)

    # Build preprocessing pipeline
    pipeline = build_pipeline()

    # Define model
    xgb_model = XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.1,
        eval_metric="logloss"
    )

    # Setup MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("churn-pipeline-baseline-models")

    # Train and log model
    train_and_log_model(
        model=xgb_model,
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_name="XGBoost"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/processed/train.csv")
    parser.add_argument("--test_path", type=str, default="data/processed/test.csv")
    args = parser.parse_args()

    main(args.train_path, args.test_path)