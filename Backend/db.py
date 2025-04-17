import boto3
from boto3.dynamodb.conditions import Key
from decouple import config

dynamodb = boto3.resource("dynamodb", region_name=config("AWS_REGION"))
history_table = dynamodb.Table(config("DDB_HISTORY_TABLE"))
thresh_table = dynamodb.Table(config("DDB_THRESH_TABLE"))

def save_reading(model: str, flow: float, anomaly: bool, timestamp: str):
    history_table.put_item(Item={
        "model": model,
        "timestamp": timestamp,
        "flow": flow,
        "anomaly": anomaly
    })

def get_history(model: str):
    resp = history_table.query(
        KeyConditionExpression=Key("model").eq(model),
        ScanIndexForward=True
    )
    return resp.get("Items", [])

def save_threshold(model: str, threshold: float):
    thresh_table.put_item(Item={"model": model, "threshold": threshold})

def get_threshold(model:str) -> float:
    resp = thresh_table.get_item(Key={"model": model})
    item = resp.get("Item")
    if item:
        return float(item["threshold"])
    return None