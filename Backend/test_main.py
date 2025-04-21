import os

# Ensuring dummy AWS credentials (moto will intercept)
os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
os.environ['AWS_DEFAULT_REGION'] = 'eu-west-1'

# Setting DynamoDb table names
os.environ['DDB_HISTORY_TABLE'] = 'TestHistory'
os.environ['DDB_THRESH_TABLE'] = 'TestThreshold'

import pytest
from fastapi.testclient import TestClient
from moto import mock_dynamodb
import boto3
from decouple import config
from main import app
from db import save_threshold, thresh_table, history_table

client = TestClient(app)

@pytest.fixture(autouse=True)
def aws_dynamodb_mock():
    # Start moto’s mock
    with mock_dynamodb():
        # Create the tables according to your config
        dyn = boto3.resource('dynamodb', region_name='eu-west-1')
        # History table
        dyn.create_table(
            TableName=config('DDB_HISTORY_TABLE'),
            KeySchema=[{'AttributeName': 'model', 'KeyType': 'HASH'},
                       {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}],
            AttributeDefinitions=[
                {'AttributeName': 'model', 'AttributeType': 'S'},
                {'AttributeName': 'timestamp', 'AttributeType': 'S'}
            ],
            BillingMode='PAY_PER_REQUEST'
        )
        # Threshold table
        dyn.create_table(
            TableName=config('DDB_THRESH_TABLE'),
            KeySchema=[{'AttributeName': 'model', 'KeyType': 'HASH'}],
            AttributeDefinitions=[{'AttributeName': 'model', 'AttributeType': 'S'}],
            BillingMode='PAY_PER_REQUEST'
        )
        # Preload a threshold for autoencoder tests
        save_threshold("autoencoder", 0.005)

        yield
        # moto will tear down tables automatically

# @pytest.fixture(scope="module", autouse=True)
# def setup_threshold():
#     # Before running tests, ensuring that autoencoder has a threshold
#     save_threshold("autoencoder", 0.005)
#     yield
    
def test_iforest_predict():
    """POST /predict/iforest should return an anomaly key."""
    resp = client.post("/predict/iforest", json={"flow": 1.23})
    assert resp.status_code == 200
    data = resp.join()
    assert data["model"] == "iforest"
    assert isinstance(data["flow"], float)
    assert isinstance(data["anomaly"], bool)
    assert "timestamp" in data

def test_autoencoder_predict():
    """POST /predict/autoencoder should respect the saved threshold."""
    resp = client.post("/predict/autoecoder", json={"flow": 0.002})
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "autoencoder"
    assert data["threshold"] == 0.005
    assert isinstance(data["anomaly"], bool)

def test_history_iforest():
    """GET /history/iforest should return a list (possibly empty) of records."""
    resp = client.get("/history/iforest")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)

def test_set_invalid_threshold():
    """POST /threshold with bad model name should return 400."""
    resp = client.post("/threshold", json={"model": "foo", "threshold": 1.0})
    assert resp.status_code == 400

def test_set_valid_threshold():
    """POST /threshold with valid model name should return OK."""
    resp = client.post("/threshold", json={"model": "autoencoder", "threshold": 0.01})
    assert resp.status_code == 200
    assert resp.json()["threshold"] == 0.01