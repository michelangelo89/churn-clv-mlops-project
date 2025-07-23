# 🧠 Churn Prediction MLOps Pipeline

> A full-stack machine learning project to predict customer churn, designed with production-ready MLOps tools.

---

## 🚀 Project Overview

This project demonstrates how to build, track, and deploy a machine learning pipeline using industry best practices.  
The pipeline predicts whether a customer will churn based on behavior and engagement features.

---

## 🛠 Tech Stack

| Area              | Tool             |
|-------------------|------------------|
| Experimentation   | MLflow           |
| Workflow          | Prefect          |
| Deployment        | Flask + Docker   |
| Monitoring        | Evidently        |
| Containerization  | Docker           |
| Logging           | Python Logging   |
| Unit Testing      | Pytest           |

---

## 📁 Key Folders

churn-clv-mlops-project/
├── data/               → Raw, processed, and prediction datasets
├── scripts/            → Scripts for training, predicting, and monitoring
├── src/                → Modularized code: data, features, models
├── web_service/        → Flask API to serve the model
├── orchestration/      → Prefect flow for training pipeline
├── models/             → MLflow exported model artifacts
├── monitoring/         → Data drift reporting with Evidently
├── tests/              → Unit test(s)

---

## 🧪 How to Use

1. 🏋️‍♀️ **Train the Model**
   ```bash
   make train

	2.	🚀 Serve the API

make docker-build
make docker-run


	3.	📬 Test the API

make test-predict


	4.	🧹 Clean Up

make clean



⸻

## ✅ Features Implemented

- ✅ Data pipeline for preprocessing

- ✅ MLflow tracking + model registry

- ✅ Prefect flow for training orchestration

- ✅ Flask API in Docker container

- ✅ Logging of predictions to file

- ✅ Basic unit test via Pytest

- ✅ Makefile for automation

## 📊 Dataset
Source: BankChurners.csv (public dataset)  
Includes demographics, usage behavior, and product info for 10,000+ customers.

## 📌 Project Status
- ✔️ Churn model deployed locally in Docker

- 🔜 CI/CD via GitHub Actions

- 🔜 Cloud deployment (S3 / ECR / EC2)

- 🔜 Real-time monitoring (Evidently, Prometheus)