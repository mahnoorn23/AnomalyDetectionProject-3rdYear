import warnings
import joblib
import numpy as np

# 1) Suppress the sklearn “feature names” warning
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)

# 2) Load the trained model (forward slash)
model = joblib.load("Models\isolation_forest_model.pk1")

# 3) Prepare test inputs as a NumPy array
new_data = np.array([
    [0.35],
    [2.5],
    [0.002]
])

# 4) Run predictions
predictions = model.predict(new_data)

# 5) Print results
for flow, pred in zip(new_data.flatten(), predictions):
    label = "Anomaly" if pred == -1 else "Normal"
    print(f"Flow = {flow} L -> {label}")