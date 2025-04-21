import joblib
import torch
# from sklearn.ensemble import IsolationForest
import os
import pandas as pd

# Loading the trained models
# iforest: IsolationForest = joblib.load("..\Models\isolation_forest_model.pk1")
# autoencoder = torch.load("..\Models\autoencoder_model_pytorch.pth")
# scaler = joblib.load("..\Models\scaler_pytorch.pk1")

# Computing project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Building full, OS-indepenent paths to each model file
IFOREST_PATH  = os.path.join(MODELS_DIR, "isolation_forest_model.pk1")
AUTOENCODER_PATH = os.path.join(MODELS_DIR, "autoencoder_model_pytorch.pth")
SCALER_PATH      = os.path.join(MODELS_DIR, "scaler.pk1")

# Sanity‐check that the files are in place
for path in (IFOREST_PATH, AUTOENCODER_PATH, SCALER_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required model file missing: {path}")

# Loading the files
iforest     = joblib.load(IFOREST_PATH)
autoencoder = torch.load(AUTOENCODER_PATH, map_location="cpu")
scaler      = joblib.load(SCALER_PATH)

def iforest_predict(flow: float) -> bool:
    # Wrapping into a DataFrame with the same column name it was trained on
    df = pd.DataFrame({"flowQuantity_delta": [flow]})
    """Return True if anomaly."""
    return iforest.predict([[flow]])[0] == -1

def autoencoder_predict (flow: float, threshold: float = None) -> bool:
    """Return True if flow > threshold or reconstruction error > threshold."""
    # Scaling and tensorizing
    x = scaler.transform([[flow]])
    xt = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        recon = autoencoder(xt).numpy()
    error = ((x - recon) ** 2).mean()
    if threshold is None:
        # Default threshold loaded elsewhere
        raise ValueError("Threshold required")
    return error > threshold