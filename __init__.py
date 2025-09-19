"""
Negative-Positive Prompt: Movie Review Sentiment Analysis Project

This project implements a comprehensive sentiment analysis pipeline for movie reviews
using multiple machine learning approaches including classical ML, deep learning,
and transformer-based models.

Project Structure:
- Data preprocessing and cleaning
- Text lemmatization and POS tagging
- Multiple model training (sklearn, PyTorch, TensorFlow/Keras, Hugging Face)
- Cross-validation and evaluation
- Model persistence and artifacts

Author: Ali Eren Küçük
"""

__version__ = "1.0.0"
__author__ = "Ali Eren Küçük"

# Import path configurations
from pathlib import Path

# Project root and directory paths
ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'
ARTIFACTS_DIR = ROOT / 'artifacts'

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, ARTIFACTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Import main modules
try:
    from . import file_path
except ImportError:
    # Fallback if file_path module doesn't exist
    pass

try:
    from . import dataset_infos
except ImportError:
    # Fallback if dataset_infos module doesn't exist
    pass

try:
    from . import tagging_words
except ImportError:
    # Fallback if tagging_words module doesn't exist
    pass

try:
    from . import x_and_y_data
except ImportError:
    # Fallback if x_and_y_data module doesn't exist
    pass

try:
    from . import model
except ImportError:
    # Fallback if model module doesn't exist
    pass

# Public API - Core functions and classes
__all__ = [
    # Project metadata
    '__version__',
    '__author__',
    
    # Paths
    'ROOT',
    'DATA_DIR', 
    'MODELS_DIR',
    'ARTIFACTS_DIR',
    
    # Core modules (conditional imports)
    'file_path',
    'dataset_infos', 
    'tagging_words',
    'x_and_y_data',
    'model',
]

# Convenience imports for common operations
def get_data_path(filename: str) -> Path:
    """Get path to a file in the data directory."""
    return DATA_DIR / filename

def get_model_path(filename: str) -> Path:
    """Get path to a file in the models directory."""
    return MODELS_DIR / filename

def get_artifacts_path(filename: str) -> Path:
    """Get path to a file in the artifacts directory."""
    return ARTIFACTS_DIR / filename

# Add convenience functions to __all__
__all__.extend(['get_data_path', 'get_model_path', 'get_artifacts_path'])

# Project description and usage info
PROJECT_INFO = {
    'name': 'Negative-Positive Prompt',
    'description': 'Movie review sentiment analysis using multiple ML approaches',
    'version': __version__,
    'author': __author__,
    'modules': [
        'file_path - Path configuration utilities',
        'dataset_infos - Dataset information and utilities', 
        'tagging_words - Text preprocessing and lemmatization',
        'x_and_y_data - Data preprocessing and train/test splitting',
        'model - Model training, validation, and evaluation'
    ],
    'data_files': [
        'IMDBDataset.csv - Original dataset',
        'X_raw.csv - Raw text data',
        'X.csv - Cleaned text data', 
        'X_train.csv, X_test.csv - Training and test splits',
        'y_train.csv, y_test.csv - Training and test labels'
    ],
    'output_directories': [
        'models/ - Trained model files',
        'artifacts/ - Training logs and metadata'
    ]
}

def print_project_info():
    """Print project information and structure."""
    print(f"Project: {PROJECT_INFO['name']}")
    print(f"Description: {PROJECT_INFO['description']}")
    print(f"Version: {PROJECT_INFO['version']}")
    print(f"Author: {PROJECT_INFO['author']}")
    print("\nModules:")
    for module in PROJECT_INFO['modules']:
        print(f"  - {module}")
    print("\nData Files:")
    for file in PROJECT_INFO['data_files']:
        print(f"  - {file}")
    print("\nOutput Directories:")
    for directory in PROJECT_INFO['output_directories']:
        print(f"  - {directory}")

# Add project info functions to __all__
__all__.extend(['PROJECT_INFO', 'print_project_info'])
