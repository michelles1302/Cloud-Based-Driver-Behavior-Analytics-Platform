"""
Cloud Storage module for Driver Behavior Analytics Platform.
Provides optional integration with Amazon S3 using Boto3.
Defaults to 100% Local Mode when AWS credentials are not configured.
"""

import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load optional .env file if present
load_dotenv()

# AWS S3 Pricing and Setup Documentation for Educational / Presentation Reference
AWS_S3_DOCUMENTATION = {
    "service": "Amazon Simple Storage Service (Amazon S3)",
    "purpose": "Secure cloud archive for historical telematics timeseries and driver risk backups.",
    "pricing_summary": (
        "AWS Free Tier includes 5 GB of standard Amazon S3 storage, 20,000 GET requests, "
        "and 2,000 PUT/POST/LIST requests per month for 12 months. Beyond free tier, "
        "Standard S3 storage costs approximately $0.023 per GB/month (US East region). "
        "Tip: Always delete temporary demo buckets or set automated S3 lifecycle rules to avoid unexpected charges."
    ),
    "required_iam_permissions": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket"
    ]
}


def get_s3_config() -> Dict[str, str]:
    """
    Retrieves AWS S3 configuration from environment variables.
    Never hardcodes secrets.
    """
    return {
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID", "").strip(),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", "").strip(),
        "aws_region": os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "us-east-1")).strip(),
        "s3_bucket_name": os.getenv("S3_BUCKET_NAME", "").strip()
    }


def is_s3_configured() -> bool:
    """
    Returns True only if both AWS credentials and a target S3 bucket name are set.
    """
    cfg = get_s3_config()
    return bool(cfg["aws_access_key_id"] and cfg["aws_secret_access_key"] and cfg["s3_bucket_name"])


def get_s3_client():
    """
    Builds a boto3 S3 client using environment configuration.
    Returns None if credentials are not configured.
    """
    if not is_s3_configured():
        return None
    cfg = get_s3_config()
    return boto3.client(
        "s3",
        aws_access_key_id=cfg["aws_access_key_id"],
        aws_secret_access_key=cfg["aws_secret_access_key"],
        region_name=cfg["aws_region"]
    )


def test_s3_connection() -> Dict[str, Any]:
    """
    Validates AWS credentials and checks whether the specified S3 bucket is reachable.
    """
    cfg = get_s3_config()
    if not is_s3_configured():
        missing = []
        if not cfg["aws_access_key_id"]:
            missing.append("AWS_ACCESS_KEY_ID")
        if not cfg["aws_secret_access_key"]:
            missing.append("AWS_SECRET_ACCESS_KEY")
        if not cfg["s3_bucket_name"]:
            missing.append("S3_BUCKET_NAME")
        return {
            "status": "local_mode",
            "connected": False,
            "message": f"Running in Local Mode (Missing configuration: {', '.join(missing)}).",
            "guide": "To enable S3 upload, configure your credentials in a .env file or environment variables."
        }

    try:
        s3 = get_s3_client()
        # Head bucket to verify existence and access permissions
        s3.head_bucket(Bucket=cfg["s3_bucket_name"])
        return {
            "status": "connected",
            "connected": True,
            "bucket": cfg["s3_bucket_name"],
            "region": cfg["aws_region"],
            "message": f"Successfully connected to Amazon S3 bucket '{cfg['s3_bucket_name']}'."
        }
    except (NoCredentialsError, PartialCredentialsError) as e:
        return {
            "status": "auth_error",
            "connected": False,
            "message": f"AWS Authentication failed: Incomplete credentials provided. Details: {e}"
        }
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        return {
            "status": "client_error",
            "connected": False,
            "message": f"Amazon S3 error ({error_code}): {e.response.get('Error', {}).get('Message', str(e))}"
        }
    except Exception as e:
        return {
            "status": "error",
            "connected": False,
            "message": f"Unexpected error connecting to Amazon S3: {str(e)}"
        }


def upload_file_to_s3(
    local_file_path: str,
    s3_key: Optional[str] = None,
    bucket_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Uploads a local file to Amazon S3 and verifies that the object exists in S3.
    Does NOT claim an upload succeeded unless verified via head_object.
    """
    if not os.path.exists(local_file_path):
        return {
            "success": False,
            "message": f"Local file not found: {local_file_path}",
            "verified": False
        }

    cfg = get_s3_config()
    target_bucket = bucket_name or cfg["s3_bucket_name"]

    if not is_s3_configured() and not bucket_name:
        return {
            "success": False,
            "message": "AWS S3 is not configured. Telemetry remains safely stored in local SQLite.",
            "verified": False
        }

    if s3_key is None:
        filename = os.path.basename(local_file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"telemetry_backups/{timestamp}_{filename}"

    try:
        s3 = get_s3_client()
        s3.upload_file(local_file_path, target_bucket, s3_key)

        # Verification step: Ensure object exists in the S3 bucket
        head = s3.head_object(Bucket=target_bucket, Key=s3_key)
        bytes_uploaded = head.get("ContentLength", 0)

        s3_uri = f"s3://{target_bucket}/{s3_key}"
        return {
            "success": True,
            "verified": True,
            "s3_uri": s3_uri,
            "bucket": target_bucket,
            "key": s3_key,
            "size_bytes": bytes_uploaded,
            "message": f"Successfully uploaded and verified {bytes_uploaded:,} bytes to {s3_uri}."
        }
    except ClientError as e:
        err_msg = e.response.get("Error", {}).get("Message", str(e))
        return {
            "success": False,
            "verified": False,
            "message": f"AWS S3 Upload Failed: {err_msg}",
            "error_code": e.response.get("Error", {}).get("Code", "ClientError")
        }
    except Exception as e:
        return {
            "success": False,
            "verified": False,
            "message": f"Unexpected error during S3 upload: {str(e)}"
        }


def backup_telemetry_to_s3(csv_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Utility to export current telemetry data to S3.
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "data", "telemetry.csv")

    if not os.path.exists(csv_path):
        return {
            "success": False,
            "message": f"Telemetry CSV not found at {csv_path}. Please generate data first.",
            "verified": False
        }

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    s3_key = f"driver_telemetry/{timestamp}_telemetry.csv"
    return upload_file_to_s3(csv_path, s3_key=s3_key)


if __name__ == "__main__":
    print("Checking Amazon S3 Cloud Storage configuration...")
    conn_result = test_s3_connection()
    print(f"Status: {conn_result['status']}")
    print(f"Message: {conn_result['message']}")
    print("\nS3 Pricing & Architecture Info:")
    print(AWS_S3_DOCUMENTATION["pricing_summary"])
