"""
Configuration module for the Sentiment Analysis Project.
Centralizes file paths and model settings to ensure consistency across modules.
"""
from pathlib import Path

class Config:
    """
    Global configuration class that holds constant paths and hyperparameter settings.
    Acts as a single source of truth for file locations.
    """
    # Project Root
    ROOT_DIR: Path = Path(__file__).parent.absolute()
    
    # Data Directories
    DATA_DIR: Path = ROOT_DIR / "data"
    MODELS_DIR: Path = ROOT_DIR / "models"
    ARTIFACTS_DIR: Path = ROOT_DIR / "artifacts"
    LOGS_DIR: Path = ARTIFACTS_DIR / "logs"
    
    # File Paths
    RAW_DATA_PATH: Path = DATA_DIR / "IMDBDataset.csv"
    TRAIN_DATA_PATH: Path = DATA_DIR / "train.csv"
    TEST_DATA_PATH: Path = DATA_DIR / "test.csv"
    
    # Model Settings
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    MAX_FEATURES: int = 20_000
    
    @classmethod
    def ensure_directories(cls):
        """
        Checks if the necessary directory structure exists.
        If directories are missing, it creates them recursively.
        """
        for path in [cls.DATA_DIR, cls.MODELS_DIR, cls.ARTIFACTS_DIR, cls.LOGS_DIR]:
            path.mkdir(parents=True, exist_ok=True)

# Initialize structure immediately upon import
Config.ensure_directories()