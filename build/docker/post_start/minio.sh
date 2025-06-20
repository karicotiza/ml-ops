#!/bin/bash

sleep 5

/usr/bin/mc config host add minio-server \
  http://127.0.0.1:9000 \
  "$MINIO_ROOT_USER" \
  "$MINIO_ROOT_PASSWORD"

/usr/bin/mc mb minio-server/mlflow-artifacts --ignore-existing
/usr/bin/mc anonymous set public minio-server/mlflow-artifacts
