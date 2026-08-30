import os
import pandas as pd
import joblib
import mlflow
from sklearn.metrics import accuracy_score

# -------------------------------------------------------------
# Fix MLflow Windows Path & Backend Storage Issues
# -------------------------------------------------------------
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
db_path = os.path.abspath("mlflow.db").replace("\\", "/")
mlflow.set_tracking_uri(f"sqlite:///{db_path}")

# -------------------------------------------------------------
# 1. Load the trained model
# -------------------------------------------------------------
model = joblib.load("telecom_tower_model.pkl")
print("Model loaded successfully!")

# -------------------------------------------------------------
# 2. Load production data (CSV file)
# -------------------------------------------------------------
new_data = pd.read_csv("Telecom_Tower_Failure_Dataset_10000-1.csv")
print("New production data loaded successfully!")

# -------------------------------------------------------------
# 3. Select ML features & target
# -------------------------------------------------------------
features = [
    "Temperature_C",
    "Battery_Voltage",
    "Power_Consumption_W",
    "Signal_Strength_Percent",
    "Fan_Speed_RPM",
    "Humidity_Percent",
    "Traffic_Load",
    "Tower_Age_Years"
]

X_new = new_data[features]
y_new = new_data["Failure_Within_48Hrs"]

# -------------------------------------------------------------
# 4. Make predictions
# -------------------------------------------------------------
predictions = model.predict(X_new)

# -------------------------------------------------------------
# 5. Calculate accuracy
# -------------------------------------------------------------
accuracy = accuracy_score(y_new, predictions)

print("-----------------------------------")
print("MODEL DRIFT MONITORING")
print("-----------------------------------")
print("Production Accuracy:", round(accuracy, 2))

# -------------------------------------------------------------
# 6. Start MLflow experiment & log metrics
# -------------------------------------------------------------
mlflow.set_experiment("Telecom_Tower_Model_Monitoring")

with mlflow.start_run():
    mlflow.log_param("model", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("production_accuracy", accuracy)
    print("Results logged successfully to MLflow")

# -------------------------------------------------------------
# 7. Drift / performance threshold check
# -------------------------------------------------------------
threshold = 0.80
if accuracy < threshold:
    print("\nWARNING: Model performance has degraded!")
    print("Model retraining is recommended.")
else:
    print("\nModel performance is stable.")