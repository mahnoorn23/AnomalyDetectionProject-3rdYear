import os
from decouple import config

# Detect testing mode
TESTING = os.getenv("TESTING") == "1"

if not TESTING:
    import boto3
    ses = boto3.client("ses", region_name=config("AWS_REGION"))
    sns = boto3.client("sns", region_name=config("AWS_REGION"))
else:
    # stub clients
    ses = None
    sns = None

def send_email(to_addr: str, subject: str, body: str):
    if TESTING:
        # no-op during tests
        return
    ses.send_email(
        Source=config("SES_SENDER"),
        Destination={"ToAddresses": [to_addr]},
        Message={
            "Subject": {"Data": subject},
            "Body": {"Text": {"Data": body}}
        }
    )

def send_sms(phone_number: str, message: str):
    if TESTING:
        # no-op during tests
        return
    sns.publish(PhoneNumber=phone_number, Message=message)