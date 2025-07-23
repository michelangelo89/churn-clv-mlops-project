# Churn and Customer Lifetime Value (CLV) Prediction

## 🎯 Goal
Build and deploy a full MLOps pipeline that predicts:
1. Whether a customer will churn
2. The customer's estimated lifetime value (CLV)

This project is based on a real-world dataset and mirrors business use cases I've worked on, now enhanced with proper deployment, monitoring, and orchestration.

## 💡 Why this project?
During my professional experience, I worked on churn and CLV models but never deployed them. This is a chance to connect everything end-to-end and strengthen my MLOps expertise.

## 📦 Dataset
The dataset is a public one: `BankChurners.csv`, which contains customer behavior and usage metrics.

## 🔧 Tech Stack
- **Cloud**: AWS
- **Experiment Tracking**: MLflow
- **Pipeline Orchestration**: Prefect
- **Deployment**: FastAPI (web service)
- **Monitoring**: Evidently + Prometheus/Grafana (optional)
- **CI/CD**: GitHub Actions
- **Containerization**: Docker
- **IaC**: Terraform

## 🧱 Project Structure
(TBD after folders are created)

## ✅ Goals
- Track experiments with MLflow
- Train churn + CLV models
- Serve prediction API
- Monitor model data and performance
- Automate with Prefect and CI/CD