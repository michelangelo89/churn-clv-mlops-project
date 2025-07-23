from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

# === Feature Engineering Function ===
def add_interaction_features(df):
    df = df.copy()
    df["Avg_Transaction_Amt"] = df["Total_Trans_Amt"] / (df["Total_Trans_Ct"] + 1e-3)
    df["Revolve_to_Limit"] = df["Total_Revolving_Bal"] / (df["Credit_Limit"] + 1e-3)
    df["AmtCt_Chg_Ratio"] = df["Total_Amt_Chng_Q4_Q1"] / (df["Total_Ct_Chng_Q4_Q1"] + 1e-3)
    return df

def build_pipeline():
    interaction_transformer = FunctionTransformer(add_interaction_features)

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

    preprocessor = ColumnTransformer([
        ("num_scaled", num_scale_transformer, numerical_to_scale),
        ("num_noscale", num_noscale_transformer, numerical_no_scale),
        ("cat", cat_transformer, categorical)
    ])

    full_pipeline = Pipeline([
        ("feature_engineering", interaction_transformer),
        ("preprocessor", preprocessor)
    ])

    return full_pipeline