FROM python:3.10-slim

WORKDIR /app

COPY src src
COPY README.md README.md
COPY setup.py setup.py

RUN apt update && apt install -y build-essential portaudio19-dev
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install /app

COPY server server

ENTRYPOINT ["python", "server/server.py"]
