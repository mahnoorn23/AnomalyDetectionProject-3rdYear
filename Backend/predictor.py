import joblib
import torch
# from sklearn.ensemble import IsolationForest
import os

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

# 2) Assert that they actually exist
for name, path in [
    ("IsolationForest", IFOREST_PATH),
    ("Autoencoder", AUTOENCODER_PATH),
    ("Scaler", SCALER_PATH),
]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} file not found at: {path}")

# 3) Load each in its own try/except so we know which one fails
try:
    iforest = joblib.load(IFOREST_PATH)
    print("✅ Loaded IsolationForest from", IFOREST_PATH)
except Exception as e:
    print("❌ Failed to load IsolationForest from", IFOREST_PATH)
    raise

try:
    # map to CPU in case GPU tensors are the culprit
    autoencoder = torch.load(AUTOENCODER_PATH, map_location="cpu")
    print("✅ Loaded Autoencoder from", AUTOENCODER_PATH)
except Exception as e:
    print("❌ Failed to load Autoencoder from", AUTOENCODER_PATH)
    raise

try:
    scaler = joblib.load(SCALER_PATH)
    print("✅ Loaded Scaler from", SCALER_PATH)
except Exception as e:
    print("❌ Failed to load Scaler from", SCALER_PATH)
    raise

def iforest_predict(flow: float) -> bool:
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