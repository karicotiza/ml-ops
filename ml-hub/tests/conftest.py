"""Conftest file."""

from os import environ

from dotenv import load_dotenv

path: str = "../build/env/.ml-infrastructure"

overwrite: dict[str, str] = {
    "MLFLOW_S3_ENDPOINT_URL": "http://localhost:9000",
}


load_dotenv(path)

for key, new_value in overwrite.items():
    environ[key] = new_value
