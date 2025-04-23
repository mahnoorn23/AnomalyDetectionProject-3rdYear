import os
from decimal import Decimal

# If TESTING=1, use in-memory dicts
TESTING = os.getenv("TESTING") == "1"

if not TESTING:
    import boto3
    from boto3.dynamodb.conditions import Key
    from decouple import config

    dynamodb = boto3.resource("dynamodb", region_name=config("AWS_REGION"))
    history_table = dynamodb.Table(config("DDB_HISTORY_TABLE"))
    thresh_table = dynamodb.Table(config("DDB_THRESH_TABLE"))


# In-memory stores
_memory_history = []
_memory_thresholds = {}

def save_reading(model: str, flow: float, anomaly: bool, timestamp: str):
    if TESTING:
        _memory_history.append({
            "model": model,
            "timestamp": timestamp,
            "flow": float(flow),
            "anomaly": anomaly
        })
    else:
        history_table.put_item(Item={
            "model": model,
            "timestamp": timestamp,
            "flow": Decimal(str(flow)),
            "anomaly": anomaly
        })

def get_history(model: str):
    if TESTING:
        return [r for r in _memory_history if r["model"] == model]
    else:
        resp = history_table.query(
            KeyConditionExpression=Key("model").eq(model),
            ScanIndexForward=True
        )
        items = resp.get("Items", [])
        for item in items:
            item["flow"] = float(item["flow"])
        return items

def save_threshold(model: str, threshold: float):
    if TESTING:
        _memory_thresholds[model] = float(threshold)
    else:
        thresh_table.put_item(Item={
            "model": model,
            "threshold": Decimal(str(threshold))
        })

def get_threshold(model: str) -> float:
    if TESTING:
        return _memory_thresholds.get(model)
    else:
        resp = thresh_table.get_item(Key={"model": model})
        item = resp.get("Item")
        return float(item["threshold"]) if item and "threshold" in item else None