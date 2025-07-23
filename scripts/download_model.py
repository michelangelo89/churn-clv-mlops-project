#!/usr/bin/env python
# scripts/download_model.py

import mlflow
import argparse

def download_model(model_name: str, model_version: int, dst_path: str):
    model_uri = f"models:/{model_name}/{model_version}"
    mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=dst_path)
    print(f"✅ Model v{model_version} downloaded to: {dst_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_version", type=int, required=True)
    parser.add_argument("--dst_path", type=str, default="models/")

    args = parser.parse_args()

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    download_model(
        model_name=args.model_name,
        model_version=args.model_version,
        dst_path=f"{args.dst_path}/{args.model_name}_v{args.model_version}"
    )