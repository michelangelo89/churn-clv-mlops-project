# scripts/batch_predict.py

import pandas as pd
import pickle
import argparse
import os

from sklearn.metrics import classification_report

def load_model(model_dir):
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    return model

def run_batch_prediction(input_path, model_dir, output_path):
    # Load data
    df = pd.read_csv(input_path)

    # Keep a copy of the IDs if needed
    if "CLIENTNUM" in df.columns:
        ids = df["CLIENTNUM"]
    else:
        ids = df.index

    # Load model (contains pipeline + classifier)
    model = load_model(model_dir)

    # Run prediction
    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    # Save to CSV
    output_df = pd.DataFrame({
        "CLIENTNUM": ids,
        "churn_pred": preds,
        "churn_proba": probs
    })

    output_df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="data/raw/new_data.csv")
    parser.add_argument("--model_dir", default="models/churn-randomforest-classifier_v3")
    parser.add_argument("--output_path", default="data/predictions/churn_predictions.csv")
    args = parser.parse_args()

    run_batch_prediction(args.input_path, args.model_dir, args.output_path)