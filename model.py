# Project name: Negative Review Classifier
# Goal: Classify movie reviews as negative or positive
# File Name: model.py
# Writer: Ali Eren Küçük (Ali Eren Küçük)
# File Goal: Define, train, cross-validate and save multiple models (scikit-learn, PyTorch, TensorFlow/Keras, Hugging Face).
#             The script loads preprocessed CSVs produced by `x_and_y_data.py` (X_train, X_test, y_train, y_test)
#             and performs model selection across different libraries. Best models and artifacts are saved
#             in `models/` with clear naming. The script is defensive, logs details, and attempts to use GPU
#             when available for deep learning models.

import os
import logging
import json
import math
import time
from pathlib import Path
from typing import Tuple, Dict, Any, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Classical models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# Deep learning
import tensorflow as tf
import keras
from keras import layers
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer # type: ignore
from keras.utils import pad_sequences


# Optional: Hugging Face transformers
try:
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              Trainer, TrainingArguments, DataCollatorWithPadding)
    from datasets import Dataset as HFDataset
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# -------------------------- Logging setup --------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("All libraries imported successfully.\n")
logger.info("Numpy version: %s", np.__version__)
logger.info("Pandas version: %s", pd.__version__)
logger.info("Scikit-learn version: %s", sklearn.__version__)
logger.info("TensorFlow version: %s", tf.__version__)
logger.info("Keras version: %s", keras.__version__)
logger.info("Hugging Face version: %s", transformers.__version__)
logger.info("Datasets version: %s", datasets.__version__)

logger.info("Model training script start")

# -------------------------- Paths and utilities --------------------------
ROOT = Path.cwd()
DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'
ARTIFACTS_DIR = ROOT / 'artifacts'


# --------------------------(TF-IDF + classic classifiers) --------------------------




# -------------------------- TensorFlow / Keras model (Tokenization + simple NN or LSTM)
