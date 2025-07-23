from prefect import flow, task
from xgboost import XGBClassifier

from src.data.load import load_data
from src.features.pipeline import build_pipeline
from src.models.train import train_and_log_model


@task
def load_data_task(train_path, test_path):
    return load_data(train_path, test_path)

@task
def build_pipeline_task():
    return build_pipeline()

@task
def train_model_task(model, pipeline, X_train, y_train, X_test, y_test):
    train_and_log_model(
        model=model,
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_name="XGBoost"
    )

@flow
def churn_training_flow(train_path: str, test_path: str):
    X_train, y_train, X_test, y_test = load_data_task(train_path, test_path)
    pipeline = build_pipeline_task()

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss"
    )

    train_model_task(model, pipeline, X_train, y_train, X_test, y_test)

# To run the flow locally
if __name__ == "__main__":
    churn_training_flow(
        train_path="data/processed/train.csv",
        test_path="data/processed/test.csv"
    )