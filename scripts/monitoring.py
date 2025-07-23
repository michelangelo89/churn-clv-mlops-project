import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Load historical (reference) and new (current) data
df_ref = pd.read_csv("data/processed/train.csv")
df_curr = pd.read_csv("data/processed/test.csv")

# Optionally drop the target column
for df in (df_ref, df_curr):
    df.drop(columns=["Churn"], errors="ignore", inplace=True)

# Create and run the data drift report
report = Report([DataDriftPreset()])
evaluation = report.run(reference_data=df_ref, current_data=df_curr)

# Save HTML output
evaluation.save_html("monitoring/drift_report.html")
print("✅ Drift report saved to monitoring/drift_report.html")