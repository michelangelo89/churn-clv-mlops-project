import mlflow
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow.sklearn
from mlflow.tracking import MlflowClient  # ✅ keep this import at the top

def train_and_log_model(model, pipeline, X_train, y_train, X_test, y_test, model_name):
    with mlflow.start_run():
        mlflow.set_tag("model", model_name)

        mlflow.log_params(model.get_params())

        final_pipeline = Pipeline([
            ("feature_pipeline", pipeline),
            ("classifier", model)
        ])

        final_pipeline.fit(X_train, y_train)

        y_pred = final_pipeline.predict(X_test)
        y_proba = final_pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.sklearn.log_model(final_pipeline, artifact_path="model")

        # ✅ REGISTER THE MODEL HERE
        client = MlflowClient()
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"

        try:
            client.create_registered_model("churn-model")
        except Exception as e:
            print("ℹ️ Model probably already exists:", e)

        client.create_model_version(
            name="churn-model",
            source=model_uri,
            run_id=run_id
        )

        print(f"✅ {model_name} - Accuracy: {acc:.4f}, ROC AUC: {roc_auc:.4f}")