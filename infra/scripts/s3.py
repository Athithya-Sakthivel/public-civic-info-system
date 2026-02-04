#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
import logging
from math import ceil
import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("manage_buckets")

AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
S3_BUCKET = os.environ.get("S3_BUCKET", "civic-data-raw-bucket")
PULUMI_STATE_BUCKET = os.environ.get("PULUMI_STATE_BUCKET", "pulumi-backend-670")
FRONTEND_UI_BUCKET = os.environ.get("FRONTEND_UI_BUCKET", "civic-bucket-for-ui")

def die(msg):
    log.error(msg)
    sys.exit(2)

if not S3_BUCKET or not PULUMI_STATE_BUCKET or not FRONTEND_UI_BUCKET:
    die("Env vars S3_BUCKET, PULUMI_STATE_BUCKET, FRONTEND_UI_BUCKET must be set")

s3 = boto3.client("s3", region_name=AWS_REGION)

def bucket_exists_and_owned(bucket):
    try:
        s3.head_bucket(Bucket=bucket)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchBucket"):
            return False
        return True

def create_bucket(bucket):
    try:
        if bucket_exists_and_owned(bucket):
            log.info("bucket exists: %s", bucket)
        else:
            if AWS_REGION == "us-east-1":
                s3.create_bucket(Bucket=bucket)
            else:
                s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": AWS_REGION})
            log.info("created bucket: %s", bucket)
        s3.put_bucket_versioning(Bucket=bucket, VersioningConfiguration={"Status": "Suspended"})
        log.info("ensured versioning suspended for: %s", bucket)
    except ClientError as e:
        die(f"create_bucket failed for {bucket}: {e}")

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def delete_all_objects_and_versions(bucket):
    paginator = s3.get_paginator("list_object_versions")
    temp_count = 0
    for page in paginator.paginate(Bucket=bucket):
        items = []
        for v in page.get("Versions", []) + page.get("DeleteMarkers", []):
            items.append({"Key": v["Key"], "VersionId": v["VersionId"]})
        if not items:
            continue
        for chunk in chunked(items, 1000):
            payload = {"Objects": chunk, "Quiet": True}
            for attempt in range(3):
                try:
                    resp = s3.delete_objects(Bucket=bucket, Delete=payload)
                    deleted = resp.get("Deleted", [])
                    temp_count += len(deleted)
                    break
                except ClientError as e:
                    log.warning("delete_objects attempt %d failed: %s", attempt+1, e)
                    time.sleep(1 + attempt*2)
            else:
                die(f"failed to delete object versions chunk in {bucket}")
    # also remove any current objects (unversioned)
    paginator2 = s3.get_paginator("list_objects_v2")
    for page in paginator2.paginate(Bucket=bucket):
        objs = page.get("Contents", [])
        if not objs:
            continue
        items = [{"Key": o["Key"]} for o in objs]
        for chunk in chunked(items, 1000):
            payload = {"Objects": chunk, "Quiet": True}
            for attempt in range(3):
                try:
                    resp = s3.delete_objects(Bucket=bucket, Delete=payload)
                    break
                except ClientError as e:
                    log.warning("delete_objects (current) attempt %d failed: %s", attempt+1, e)
                    time.sleep(1 + attempt*2)
            else:
                die(f"failed to delete objects chunk in {bucket}")
    log.info("deleted objects (versions + current) for %s", bucket)

def delete_bucket(bucket, force=False):
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchBucket"):
            log.info("bucket not found (already deleted): %s", bucket)
            return
        die(f"head_bucket failed for {bucket}: {e}")
    try:
        if force:
            log.info("force deleting contents of %s", bucket)
            delete_all_objects_and_versions(bucket)
        else:
            objs = s3.list_objects_v2(Bucket=bucket, MaxKeys=1).get("KeyCount", 0)
            versions = s3.list_object_versions(Bucket=bucket, MaxKeys=1)
            has_versions = bool(versions.get("Versions") or versions.get("DeleteMarkers"))
            if objs > 0 or has_versions:
                die(f"bucket {bucket} not empty; rerun with --force to remove contents")
        for attempt in range(3):
            try:
                s3.delete_bucket(Bucket=bucket)
                log.info("deleted bucket: %s", bucket)
                return
            except ClientError as e:
                log.warning("delete_bucket attempt %d failed: %s", attempt+1, e)
                time.sleep(1 + attempt*2)
        die(f"failed to delete bucket {bucket}")
    except ClientError as e:
        die(f"delete_bucket failed for {bucket}: {e}")

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create", action="store_true")
    group.add_argument("--delete", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    buckets = [S3_BUCKET, PULUMI_STATE_BUCKET, FRONTEND_UI_BUCKET]
    if args.create:
        for b in buckets:
            create_bucket(b)
        log.info("create complete")
    elif args.delete:
        for b in buckets:
            delete_bucket(b, force=args.force)
        log.info("delete complete")

if __name__ == "__main__":
    main()
