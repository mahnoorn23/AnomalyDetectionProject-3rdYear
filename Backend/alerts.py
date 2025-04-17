import boto3
from decouple import config

ses = boto3.client("ses", region_name=config("AWS_REGION"))
sns = boto3.client("sns", region_name=config("AWS_REGION"))

def send_email(to_addr: str, subject: str, body: str):
    ses.send_email(
        Source=config("SES_SENDER"),
        Destination={"ToAddresses": [to_addr]},
        Message={
            "Subject": {"Data": subject},
            "Body": {"Text": {"Data": body}}
        }
    )
def send_sms(phone_number: str, message: str):
    sns.publish(PhoneNumber=phone_number, Message=message)