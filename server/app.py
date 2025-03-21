# server.py
import os

import uvicorn
from auralis.common.logging import setup_logger
from server import app

logger = setup_logger(__name__)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=int(os.environ.get("AURALIS_WORKERS", 1)),
        log_level="info"
    )