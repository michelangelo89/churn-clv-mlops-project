#!/usr/bin/env python
# coding: utf-8

# In[11]:


# 1. Setup
import pandas as pd
import numpy as np
import pickle
import mlflow
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline


# In[12]:


# 2. Load Train/Test Data

train_df = pd.read_csv("../data/processed/train.csv")
test_df = pd.read_csv("../data/processed/test.csv")

# Target
target = "churn"
X_train = train_df.drop(columns=[target])
y_train = train_df[target]

X_test = test_df.drop(columns=[target])
y_test = test_df[target]


# In[13]:


print(f"✅ Train shape: {train_df.shape}")
print(f"✅ Test shape: {test_df.shape}")

# Optional: show a few rows
train_df.head()


# In[10]:


# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("churn-pipeline-baseline-models")


# In[14]:


# === 1. Define your custom feature engineering logic ===
def add_interaction_features(df):
    df = df.copy()
    df["Avg_Transaction_Amt"] = df["Total_Trans_Amt"] / (df["Total_Trans_Ct"] + 1e-3)
    df["Revolve_to_Limit"] = df["Total_Revolving_Bal"] / (df["Credit_Limit"] + 1e-3)
    df["AmtCt_Chg_Ratio"] = df["Total_Amt_Chng_Q4_Q1"] / (df["Total_Ct_Chng_Q4_Q1"] + 1e-3)
    return df

# Wrap it as a FunctionTransformer
interaction_transformer = FunctionTransformer(add_interaction_features)

# === 2. Separate features ===
categorical = [
    "Gender", "Education_Level", "Marital_Status", 
    "Income_Category", "Card_Category"
]

numerical_to_scale = [
    "Credit_Limit", "Total_Revolving_Bal", 
    "Total_Trans_Amt", "Total_Trans_Ct", 
    "Avg_Utilization_Ratio", "Avg_Transaction_Amt", 
    "Revolve_to_Limit", "AmtCt_Chg_Ratio"
]

numerical_no_scale = [
    "Customer_Age", "Dependent_count", "Months_on_book",
    "Total_Relationship_Count", "Months_Inactive_12_mon",
    "Contacts_Count_12_mon", "Total_Amt_Chng_Q4_Q1", 
    "Total_Ct_Chng_Q4_Q1"
]

# === 3. Define transformers ===

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

num_scale_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

num_noscale_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# === 4. Compose all into one ColumnTransformer ===
preprocessor = ColumnTransformer([
    ("num_scaled", num_scale_transformer, numerical_to_scale),
    ("num_noscale", num_noscale_transformer, numerical_no_scale),
    ("cat", cat_transformer, categorical)
])

# === 5. Final pipeline with feature engineering + preprocessor ===
full_pipeline = Pipeline([
    ("feature_engineering", interaction_transformer),
    ("preprocessor", preprocessor)
])


# In[15]:


def train_and_log_model(model, model_name, extra_params=None):
    #mlflow.set_tracking_uri("http://127.0.0.1:5000")
    #mlflow.set_experiment("churn-pipeline-baseline-models")

    with mlflow.start_run():
        mlflow.set_tag("model", model_name)

        # Automatically extract model parameters
        model_params = model.get_params()

        # Merge with any manually passed params (optional)
        if extra_params:
            model_params.update(extra_params)

        mlflow.log_params(model_params)

        pipeline = Pipeline([
            ("feature_pipeline", full_pipeline),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        print(f"✅ {model_name} - Accuracy: {acc:.4f}, ROC AUC: {roc_auc:.4f}")


# In[16]:


xgb_model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss")

train_and_log_model(xgb_model, model_name="XGBoost")


# In[ ]:




