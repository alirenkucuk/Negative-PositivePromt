"""
File Path Configuration Module

This module provides centralized path management for the Negative-Positive Prompt project.
It defines all directory paths and provides utilities for file path operations.

Author: Ali Eren Küçük
"""

import os
from pathlib import Path
from typing import Union, Optional

# Project root directory (where this file is located)
ROOT = Path(__file__).parent.absolute()

# Main directories
DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'
ARTIFACTS_DIR = ROOT / 'artifacts'
NOTEBOOKS_DIR = ROOT / 'notebooks'
TESTS_DIR = ROOT / 'tests'
DOCS_DIR = ROOT / 'docs'

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
EXTERNAL_DATA_DIR = DATA_DIR / 'external'

# Model subdirectories
CLASSICAL_MODELS_DIR = MODELS_DIR / 'classical'
DEEP_LEARNING_MODELS_DIR = MODELS_DIR / 'deep_learning'
TRANSFORMER_MODELS_DIR = MODELS_DIR / 'transformers'

# Artifact subdirectories
LOGS_DIR = ARTIFACTS_DIR / 'logs'
RESULTS_DIR = ARTIFACTS_DIR / 'results'
CONFIGS_DIR = ARTIFACTS_DIR / 'configs'

# Specific data files
IMDB_DATASET_PATH = DATA_DIR / 'IMDBDataset.csv'
X_RAW_PATH = DATA_DIR / 'X_raw.csv'
X_CLEANED_PATH = DATA_DIR / 'X.csv'
X_TRAIN_PATH = DATA_DIR / 'X_train.csv'
X_TEST_PATH = DATA_DIR / 'X_test.csv'
Y_TRAIN_PATH = DATA_DIR / 'y_train.csv'
Y_TEST_PATH = DATA_DIR / 'y_test.csv'
Y_PATH = DATA_DIR / 'y.csv'

# Lemmatized data files
X_LEMMATIZED_PATH = DATA_DIR / 'X_lemmatized.csv'
X_TRAIN_LEMMATIZED_PATH = DATA_DIR / 'X_train_lemmatized.csv'
X_TEST_LEMMATIZED_PATH = DATA_DIR / 'X_test_lemmatized.csv'
X_TRAIN_AND_TEST_LEMMATIZED_PATH = DATA_DIR / 'X_train_and_X_test_lemmatized.csv'

# Model file paths
SKLEARN_MODELS = {
    'logistic': MODELS_DIR / 'sklearn_logistic.joblib',
    'svc': MODELS_DIR / 'sklearn_svc.joblib',
    'nb': MODELS_DIR / 'sklearn_nb.joblib',
    'rf': MODELS_DIR / 'sklearn_rf.joblib',
}

TORCH_MODELS = {
    'tfidf_base': MODELS_DIR / 'torch_tfidf_base.pt',
    'tfidf_fold1': MODELS_DIR / 'torch_tfidf_fold1.pt',
    'tfidf_fold2': MODELS_DIR / 'torch_tfidf_fold2.pt',
    'tfidf_fold3': MODELS_DIR / 'torch_tfidf_fold3.pt',
}

KERAS_MODELS = {
    'lstm_base': MODELS_DIR / 'keras_lstm_base',
    'lstm_fold1': MODELS_DIR / 'keras_lstm_fold1',
    'lstm_fold2': MODELS_DIR / 'keras_lstm_fold2',
    'lstm_fold3': MODELS_DIR / 'keras_lstm_fold3',
}

HF_MODELS = {
    'distilbert': MODELS_DIR / 'hf_distilbert-base-uncased',
    'bert': MODELS_DIR / 'hf_bert-base-uncased',
}

# Artifact file paths
SKLEARN_RESULTS_PATH = ARTIFACTS_DIR / 'sklearn_results.json'
SKLEARN_TEST_EVAL_PATH = ARTIFACTS_DIR / 'sklearn_test_eval.json'
TORCH_RESULTS_PATH = ARTIFACTS_DIR / 'torch_results.json'
KERAS_RESULTS_PATH = ARTIFACTS_DIR / 'keras_results.json'
MODELING_SUMMARY_PATH = ARTIFACTS_DIR / 'modeling_summary.json'

# Vectorizer and tokenizer paths
TFIDF_VECTORIZER_PATH = ARTIFACTS_DIR / 'tfidf_vectorizer.joblib'
TFIDF_FOR_TORCH_PATH = ARTIFACTS_DIR / 'tfidf_for_torch.joblib'
KERAS_TOKENIZER_PATH = ARTIFACTS_DIR / 'keras_tokenizer.joblib'

# Log paths
MAIN_LOG_PATH = LOGS_DIR / 'main.log'
MODEL_TRAINING_LOG_PATH = LOGS_DIR / 'model_training.log'
PREPROCESSING_LOG_PATH = LOGS_DIR / 'preprocessing.log'

# Configuration paths
MODEL_CONFIG_PATH = CONFIGS_DIR / 'model_config.json'
PREPROCESSING_CONFIG_PATH = CONFIGS_DIR / 'preprocessing_config.json'

# Directory creation function
def create_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR, MODELS_DIR, ARTIFACTS_DIR, NOTEBOOKS_DIR, TESTS_DIR, DOCS_DIR,
        RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
        CLASSICAL_MODELS_DIR, DEEP_LEARNING_MODELS_DIR, TRANSFORMER_MODELS_DIR,
        LOGS_DIR, RESULTS_DIR, CONFIGS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Directory created/verified: {directory}")

# Utility functions
def get_data_path(filename: str, subdir: Optional[str] = None) -> Path:
    """
    Get path to a file in the data directory.
    
    Args:
        filename: Name of the file
        subdir: Optional subdirectory (e.g., 'raw', 'processed', 'external')
    
    Returns:
        Path object pointing to the file
    """
    if subdir:
        return DATA_DIR / subdir / filename
    return DATA_DIR / filename

def get_model_path(filename: str, model_type: str = 'classical') -> Path:
    """
    Get path to a model file.
    
    Args:
        filename: Name of the model file
        model_type: Type of model ('classical', 'deep_learning', 'transformers')
    
    Returns:
        Path object pointing to the model file
    """
    if model_type == 'classical':
        return CLASSICAL_MODELS_DIR / filename
    elif model_type == 'deep_learning':
        return DEEP_LEARNING_MODELS_DIR / filename
    elif model_type == 'transformers':
        return TRANSFORMER_MODELS_DIR / filename
    else:
        return MODELS_DIR / filename

def get_artifact_path(filename: str, artifact_type: str = 'results') -> Path:
    """
    Get path to an artifact file.
    
    Args:
        filename: Name of the artifact file
        artifact_type: Type of artifact ('logs', 'results', 'configs')
    
    Returns:
        Path object pointing to the artifact file
    """
    if artifact_type == 'logs':
        return LOGS_DIR / filename
    elif artifact_type == 'results':
        return RESULTS_DIR / filename
    elif artifact_type == 'configs':
        return CONFIGS_DIR / filename
    else:
        return ARTIFACTS_DIR / filename

def file_exists(path: Union[str, Path]) -> bool:
    """Check if a file exists."""
    return Path(path).exists()

def ensure_file_exists(path: Union[str, Path]) -> None:
    """Ensure a file exists, raise FileNotFoundError if not."""
    if not file_exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")

def get_relative_path(path: Union[str, Path]) -> Path:
    """Get path relative to project root."""
    return Path(path).relative_to(ROOT)

def list_data_files() -> list:
    """List all files in the data directory."""
    if not DATA_DIR.exists():
        return []
    return [f for f in DATA_DIR.iterdir() if f.is_file()]

def list_model_files() -> list:
    """List all files in the models directory."""
    if not MODELS_DIR.exists():
        return []
    return [f for f in MODELS_DIR.rglob('*') if f.is_file()]

def list_artifact_files() -> list:
    """List all files in the artifacts directory."""
    if not ARTIFACTS_DIR.exists():
        return []
    return [f for f in ARTIFACTS_DIR.rglob('*') if f.is_file()]

# Path validation
def validate_project_structure() -> dict:
    """
    Validate that the project has the expected structure.
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'project_root': ROOT.exists(),
        'data_dir': DATA_DIR.exists(),
        'models_dir': MODELS_DIR.exists(),
        'artifacts_dir': ARTIFACTS_DIR.exists(),
        'imdb_dataset': IMDB_DATASET_PATH.exists(),
    }
    
    # Check for key data files
    key_files = [
        X_RAW_PATH, X_CLEANED_PATH, X_TRAIN_PATH, X_TEST_PATH,
        Y_TRAIN_PATH, Y_TEST_PATH, Y_PATH
    ]
    
    for file_path in key_files:
        validation_results[f'file_{file_path.stem}'] = file_path.exists()
    
    return validation_results

# Initialize directories on import
create_directories()

# Export all paths and functions
__all__ = [
    # Directory paths
    'ROOT', 'DATA_DIR', 'MODELS_DIR', 'ARTIFACTS_DIR', 'NOTEBOOKS_DIR', 'TESTS_DIR', 'DOCS_DIR',
    'RAW_DATA_DIR', 'PROCESSED_DATA_DIR', 'EXTERNAL_DATA_DIR',
    'CLASSICAL_MODELS_DIR', 'DEEP_LEARNING_MODELS_DIR', 'TRANSFORMER_MODELS_DIR',
    'LOGS_DIR', 'RESULTS_DIR', 'CONFIGS_DIR',
    
    # Data file paths
    'IMDB_DATASET_PATH', 'X_RAW_PATH', 'X_CLEANED_PATH', 'X_TRAIN_PATH', 'X_TEST_PATH',
    'Y_TRAIN_PATH', 'Y_TEST_PATH', 'Y_PATH',
    'X_LEMMATIZED_PATH', 'X_TRAIN_LEMMATIZED_PATH', 'X_TEST_LEMMATIZED_PATH',
    'X_TRAIN_AND_TEST_LEMMATIZED_PATH',
    
    # Model dictionaries
    'SKLEARN_MODELS', 'TORCH_MODELS', 'KERAS_MODELS', 'HF_MODELS',
    
    # Artifact paths
    'SKLEARN_RESULTS_PATH', 'SKLEARN_TEST_EVAL_PATH', 'TORCH_RESULTS_PATH',
    'KERAS_RESULTS_PATH', 'MODELING_SUMMARY_PATH',
    'TFIDF_VECTORIZER_PATH', 'TFIDF_FOR_TORCH_PATH', 'KERAS_TOKENIZER_PATH',
    
    # Log and config paths
    'MAIN_LOG_PATH', 'MODEL_TRAINING_LOG_PATH', 'PREPROCESSING_LOG_PATH',
    'MODEL_CONFIG_PATH', 'PREPROCESSING_CONFIG_PATH',
    
    # Functions
    'create_directories', 'get_data_path', 'get_model_path', 'get_artifact_path',
    'file_exists', 'ensure_file_exists', 'get_relative_path',
    'list_data_files', 'list_model_files', 'list_artifact_files',
    'validate_project_structure'
]
