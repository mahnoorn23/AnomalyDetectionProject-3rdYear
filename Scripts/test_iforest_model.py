import joblib

# Load the trained model
model = joblib.load("Models\isolation_forest_model.pk1")

# Using it for predictions
new_data = [[0.35], [2.5], [0.002]] # Example inputs to test
predictions = model.predict(new_data)

for i, val in enumerate(new_data):
    label = "Anomaly" if predictions[i] == -1 else "Normal"
    print(f"Flow = {val[0]} L -> {label}")