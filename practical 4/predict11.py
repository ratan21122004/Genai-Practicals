import joblib
import pandas as pd

model = joblib.load("telecom_tower_model.pkl")

new_data = pd.DataFrame({
    "Temperature_C": [55],
    "Battery_Voltage": [45.6],
    "Power_Consumption_W": [3300],
    "Signal_Strength_Percent": [40],
    "Fan_Speed_RPM": [3400],
    "Humidity_Percent": [88],
    "Traffic_Load": [4200],
    "Tower_Age_Years": [10]
})

prediction = model.predict(new_data)

if prediction[0] == 1:
    print("Hardware Failure Predicted")
else:
    print("Tower is Healthy")
