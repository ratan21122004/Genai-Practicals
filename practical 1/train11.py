import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import json

print("Loading Dataset...")

df = pd.read_csv("Telecom_Tower_Failure_Dataset_10000-1.csv")

X = df[['Temperature_C',
        'Battery_Voltage',
        'Power_Consumption_W',
        'Signal_Strength_Percent',
        'Fan_Speed_RPM',
        'Humidity_Percent',
        'Traffic_Load',
        'Tower_Age_Years']]

y = df['Failure_Within_48Hrs']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("Training Model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("Accuracy :", accuracy)

joblib.dump(model, "telecom_tower_model.pkl")

metrics = {"accuracy": accuracy}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Training Completed Successfully")
