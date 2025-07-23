import mlflow.pyfunc
import pandas as pd

model_uri = "s3://mlops-churn-analytics-falcon/mlflow-artifacts/2/models/m-7cb7b517788c48a5ac1aa5808135197f/artifacts/"
model = mlflow.pyfunc.load_model(model_uri)

print("✅ Model loaded.")

# Create a dummy input from training
# Replace with a realistic row or load from file if needed
sample_input = pd.DataFrame([{
    "customer_Age": 45,
    "gender": "Male",
    "income": 50000,
    "subscription_length": 12,
    "churned": 0
}])

try:
    proba = model.predict_proba(sample_input)
    print("🎯 YES — `predict_proba` works. Example output:")
    print(proba)
except Exception as e:
    print("❌ `predict_proba` is NOT available or failed:")
    print(e)