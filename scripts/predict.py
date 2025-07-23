# scripts/predict.py

import sys
import pandas as pd
import pickle


def load_model(model_path):
    with open(f"{model_path}/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


def predict_single(model, input_dict):
    df = pd.DataFrame([input_dict])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0, 1]
    return {
        "churn_prediction": int(pred),
        "churn_probability": float(proba)
    }


if __name__ == "__main__":
    # Example: python scripts/predict.py "data/raw/example_input.csv" "models/churn-randomforest-classifier_v3"
    input_csv = sys.argv[1]  # path to CSV file with 1 row
    model_path = sys.argv[2]

    model = load_model(model_path)
    df = pd.read_csv(input_csv)
    result = predict_single(model, df.iloc[0].to_dict())

    print(result)