from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predictor import iforest_predict, autoencoder_predict
from db import save_reading, get_history, save_threshold, get_threshold
from alerts import send_email, send_sms
from datetime import datetime, timezone
from decouple import config

app = FastAPI(title="Water Anomaly Detection API")

class PredictRequest(BaseModel):
    flow: float
    email: str = None
    phone: str = None

class ThresholdRequest(BaseModel):
    model: str
    threshold: float

# Isolation Forest Endpoint
@app.post("/predict/{iforest}")
def predict_iforest(req: PredictRequest):
    timestamp = datetime.now(timezone.utc).isoformat()
    # Run IF model
    is_anomaly = iforest_predict(req.flow)

    # Save to history
    save_reading("iforest", req.flow, is_anomaly, timestamp)

    # Send alerts
    subject = "[Alert] Isolation Forest detected anomaly"
    body = f"Model: iforest\nFlow: {req.flow}\nTime: {timestamp}"
    if is_anomaly:
        if req.email:
            send_email(req.email, subject, body)
        if req.phone:
            send_sms(req.phone, body)

    return {"model": "iforest", "flow": req.flow, "anomaly": is_anomaly, "timestamp": timestamp}


# Autoencoder Endpoint
@app.post("/predict/{autoencoder}")
def predict_autoencoder(req: PredictRequest):
    timestamp = datetime.now(timezone.utc).isoformat()
    # Load user threshold (must have been set via /threshold)
    thresh = get_threshold("autoencoder")
    if thresh is None:
        raise HTTPException(400, "Autoencoder threshold not set")

    # Run Autoencoder model
    is_anomaly = autoencoder_predict(req.flow, threshold=thresh)

    # Save to history
    save_reading("autoencoder", req.flow, is_anomaly, timestamp)

    # Send alerts
    subject = "[Alert] Autoencoder detected anomaly"
    body = f"Model: autoencoder\nFlow: {req.flow}\nThreshold: {thresh}\nTime: {timestamp}"
    if is_anomaly:
        if req.email:
            send_email(req.email, subject, body)
        if req.phone:
            send_sms(req.phone, body)

    return {"model": "autoencoder", "flow": req.flow, "anomaly": is_anomaly, "threshold": thresh, "timestamp": timestamp}


# History Endpoint
@app.get("/history/{model_name}")
def history(model_name: str):
    if model_name not in ("iforest", "autoencoder"):
        raise HTTPException(404, "Model not found")
    return get_history(model_name)


# Threshold Endpoint
@app.post("/threshold")
def set_threshold(req: ThresholdRequest):
    if req.model not in ("iforest", "autoencoder"):
        raise HTTPException(400, "Invalid model")
    save_threshold(req.model, req.threshold)
    return {"status": "OK", "model": req.model, "threshold": req.threshold}