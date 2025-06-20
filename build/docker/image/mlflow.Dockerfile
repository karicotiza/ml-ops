FROM ghcr.io/mlflow/mlflow:v3.1.0

RUN apt update -y
RUN apt install -y curl=7.74.0-1.3+deb11u15

RUN python -m pip install --no-cache psycopg2-binary==2.9.10
RUN python -m pip install --no-cache boto3==1.38.40

CMD ["bash"]
