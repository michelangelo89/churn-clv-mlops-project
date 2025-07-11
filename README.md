🧠 Churn Prediction Web Service

This project demonstrates how to train a machine learning pipeline, deploy it as a Flask API, and serve predictions for customer churn risk using HTTP requests.

⸻

📌 Project Overview
	•	Goal: Predict customer churn based on transactional and demographic data.
	•	Model: Random Forest (wrapped in a full sklearn pipeline).
	•	Deployment: REST API via Flask.
	•	Inference modes:
	•	Batch mode (offline, CSV)
	•	Web service mode (real-time, JSON via HTTP)

⸻

✅ Pipeline Training & Saving

python scripts/train.py \
  --train_path data/processed/train.csv \
  --test_path data/processed/test.csv

	•	Trains a model using a pipeline:
	•	Feature engineering
	•	Preprocessing (scaling, imputation, encoding)
	•	Classifier
	•	Logs to MLflow
	•	Saves the model locally via pickle

⸻

🌐 Web Service (Flask API)

🔁 Start the API

python web_service/predict_flask.py

This will:
	•	Load the trained pipeline model
	•	Start a Flask server at http://127.0.0.1:9696

📥 Example Input (JSON)

sample_customer.json

{
  "CLIENTNUM": 1,
  "Customer_Age": 45,
  "Gender": "F",
  "Dependent_count": 2,
  "Education_Level": "Graduate",
  "Marital_Status": "Married",
  "Income_Category": "$60K - $80K",
  "Card_Category": "Blue",
  "Months_on_book": 39,
  "Total_Relationship_Count": 5,
  "Months_Inactive_12_mon": 1,
  "Contacts_Count_12_mon": 3,
  "Credit_Limit": 8555,
  "Total_Revolving_Bal": 300,
  "Avg_Utilization_Ratio": 0.1,
  "Total_Trans_Amt": 4000,
  "Total_Trans_Ct": 50,
  "Total_Amt_Chng_Q4_Q1": 1.2,
  "Total_Ct_Chng_Q4_Q1": 1.1
}

🚀 Run Prediction

curl -X POST \
  -H "Content-Type: application/json" \
  --data @sample_customer.json \
  http://127.0.0.1:9696/predict

📤 Output Example

{
  "churn_prediction": 0,
  "churn_probability": 0.0008
}


⸻

🔐 Why It Works with Extra Columns?

Your trained pipeline uses ColumnTransformer and ignores irrelevant fields like CLIENTNUM — keeping the model robust and production-safe.

⸻

🧩 Folder Structure Summary

churn-clv-mlops-project/
│
├── data/                        # Input, processed and prediction data
├── models/                      # Trained model (pickle format)
├── notebooks/                   # EDA, pipeline dev, experiments
├── scripts/                     # Training, batch prediction scripts
├── src/                         # Modularized data, pipeline, model logic
├── web_service/
│   └── predict_flask.py         # Flask app serving predictions
│
└── sample_customer.json         # Input example for prediction


⸻

