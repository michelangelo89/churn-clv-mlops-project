# Churn Prediction with MLOps Pipeline

## 🌟 Goal

Build and deploy a full MLOps pipeline that predicts:

* ✅ Whether a customer will **churn**

> 🖜️ (Optional future extension): Predicting the customer's estimated **lifetime value (CLV)**

This project mirrors real-world business use cases I worked on, enhanced with proper deployment, monitoring, and orchestration.

## 💡 Why this project?

During my professional experience, I developed churn prediction models but never deployed them end-to-end.
This project allows me to **close that loop** — using production-ready tools to **track, orchestrate, and serve** a model with real-world MLOps best practices.

## 📦 Dataset

The dataset used is public: `BankChurners.csv`, which contains customer usage behavior and product subscription metrics.

## 🔧 Tech Stack

| Task                   | Tool                                               |
| ---------------------- | -------------------------------------------------- |
| Experiment Tracking    | MLflow                                             |
| Workflow Orchestration | Prefect                                            |
| Model Deployment       | FastAPI                                            |
| Containerization       | Docker                                             |
| Cloud Storage          | AWS S3                                             |
| Infrastructure (WIP)   | Terraform (planned)                                |
| Monitoring (WIP)       | Evidently (planned), Prometheus/Grafana (optional) |
| CI/CD (WIP)            | GitHub Actions (planned)                           |

## 🧱 Project Structure

> The repo is modularized under `src/` for data, features, models, etc.
> Prefect workflows live in `orchestration/`, and deployment code is in `service/`.

```
churn-clv-mlops-project/
├── data/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
├── orchestration/
├── service/
├── notebooks/
├── Dockerfile
└── README.md
```

## ✅ Completed So Far

* ✅ Cleaned and preprocessed data
* ✅ Created and tracked churn model experiments using **MLflow**
* ✅ Integrated **Prefect** to orchestrate training as a workflow
* ✅ Registered model in the **MLflow Model Registry**
* ✅ Confirmed model can be served with `predict_proba`
* ✅ Built a FastAPI endpoint for predictions and tested via `curl`

## 🚀 Next Steps

* [ ] Add model signature and input example during MLflow model logging
* [ ] (Optional) Add retraining triggers if performance drifts
* [ ] (Optional) Monitor input data with **Evidently**
* [ ] (Optional) Create CI/CD pipeline with **GitHub Actions**
* [ ] (Optional) Use **Terraform** to provision S3/EC2 infrastructure
