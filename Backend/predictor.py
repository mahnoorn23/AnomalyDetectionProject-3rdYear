import joblib
import torch
from sklearn.ensemble import IsolationForest

# Loading the models at import time
iforest = joblib.load("Models\isolation_forest_model.pk1")
autoencoder = torch.load("Models\autoencoder_model_pytorch.pth")
scaler = joblib.load("Models\scaler_pytorch.pk1")

def iforest_predict(flow):
    """Return True if anomaly."""
    return iforest.predict([[flow]])[0] == -1

def autoencoder_predict (flow: float, threshold: float = None) -> bool:
    """Return True if flo > threshold or reconstruction error > threshold."""
    # Scaling and tensorizing
    x = scaler.transform([[flow]])
    xt = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        recon = autoencoder(xt).numpy()
    error = ((x - recon) ** 2).mean()
    if threshold is None:
        # default threshold loaded elsewhere
        raise ValueError("Threshold required")
    return error > threshold