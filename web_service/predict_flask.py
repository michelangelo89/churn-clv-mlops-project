import mlflow.sklearn  # if needed later
import pandas as pd
from flask import Flask, request, jsonify
import pickle
import logging
import os
from datetime import datetime

# --- Logging Setup ---
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/predictions_{datetime.today().strftime('%Y-%m-%d')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)


# Load model directly from S3 using sklearn loader
# model_uri = "s3://mlops-churn-analytics-falcon/mlflow-artifacts/2/models/m-7cb7b517788c48a5ac1aa5808135197f/artifacts/"
# model = mlflow.sklearn.load_model(model_uri)

# --- Load Model ---
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

# --- Init Flask App ---
app = Flask("churn-predictor")

@app.route("/predict", methods=["POST"])
def predict():
    customer = request.get_json()
    df = pd.DataFrame([customer])

    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0, 1]

    result = {
        "churn_prediction": int(pred),
        "churn_probability": float(proba)
    }

    # --- Log Request & Result ---
    logging.info(f"Input: {customer}")
    logging.info(f"Prediction: {result}")

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)