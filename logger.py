import os
import logging

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/log.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)