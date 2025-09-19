# Project name: Negative Review Classifier
# Goal: Classify movie reviews as negative or positive
# File Name: file_path.py
# Writer: Ali Eren Küçük
# File Goal: Define the file paths for the project

from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FilePath:
    ROOT = Path.cwd()
    DATA_DIR = ROOT / 'data'
    MODELS_DIR = ROOT / 'models'
    ARTIFACTS_DIR = ROOT / 'artifacts'

    def __init__(self):
        for d in (self.MODELS_DIR, self.ARTIFACTS_DIR):
            d.mkdir(parents=True, exist_ok=True)
        logger.info("File paths initialized")
    
    def ensure_file_exists(self, path):
        if not path.exists():
            logger.error("File not found: %s", path)
            raise FileNotFoundError(path)

    def get_file_path(self, dir_name, file_name):
        logger.info("Getting file path for %s/%s", dir_name, file_name)
        try:
            return self.DATA_DIR / dir_name / file_name
        except Exception as e:
            logger.error("Error getting file path for %s/%s: %s", dir_name, file_name, e)
            raise


file_path = FilePath()

logger.info("File paths initialized")

