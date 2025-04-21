import boto3
from boto3.dynamodb.conditions import Key
from decouple import config
from decimal import Decimal

dynamodb = boto3.resource("dynamodb", region_name=config("AWS_REGION"))
history_table = dynamodb.Table(config("DDB_HISTORY_TABLE"))
thresh_table = dynamodb.Table(config("DDB_THRESH_TABLE"))

def save_reading(model: str, flow: float, anomaly: bool, timestamp: str):
    # Covert flow (float) to Decimal for DynamoDB storage
    history_table.put_item(Item={
        "model": model,
        "timestamp": timestamp,
        "flow": Decimal(str(flow)),
        "anomaly": anomaly
    })

def get_history(model: str):
    resp = history_table.query(
        KeyConditionExpression=Key("model").eq(model),
        ScanIndexForward=True
    )
    items = resp.get("Items", [])
    # Convert the Decimal back to float for the API response
    for item in items:
        item["flow"] = float(item["flow"])
    return items

def save_threshold(model: str, threshold: float):
    # Convert threshold (float) to Decimal
    thresh_table.put_item(Item={
        "model": model, 
        "threshold": Decimal(str(threshold))
    })

def get_threshold(model:str) -> float:
    resp = thresh_table.get_item(Key={"model": model})
    item = resp.get("Item")
    if item and "threshold" in item:
        return float(item["threshold"])
    return None