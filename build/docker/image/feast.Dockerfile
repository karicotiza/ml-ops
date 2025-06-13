FROM python:3.12.11

RUN python -m pip install --no-cache feast[redis,postgres]==0.49.0
RUN python -m pip install --no-cache protobuf==5.29.0

WORKDIR /feast

EXPOSE 6566

ENTRYPOINT ["feast", "serve", "--host", "0.0.0.0", "--port", "6566"]
