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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

logger.info("Model training script start")

# -------------------------- Paths and utilities --------------------------
ROOT = Path.cwd()
DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'
ARTIFACTS_DIR = ROOT / 'artifacts'

for d in (MODELS_DIR, ARTIFACTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def ensure_file_exists(path):
    if not path.exists():
        logger.error("Required file not found: %s", path)
        raise FileNotFoundError(path)


def load_csvs():
    """Load pre-saved CSVs created by x_and_y_data.py

    Expected files:
      data/X_train.csv, data/X_test.csv, data/y_train.csv, data/y_test.csv

    Returns
    -------
    X_train, X_test, y_train, y_test (pd.Series)
    """
    paths = {
        'X_train': DATA_DIR / 'X_train.csv',
        'X_test': DATA_DIR / 'X_test.csv',
        'y_train': DATA_DIR / 'y_train.csv',
        'y_test': DATA_DIR / 'y_test.csv',
    }

    for p in paths.values():
        ensure_file_exists(p)

    X_train = pd.read_csv(paths['X_train'])['review']
    X_test = pd.read_csv(paths['X_test'])['review']
    y_train = pd.read_csv(paths['y_train'])['sentiment']
    y_test = pd.read_csv(paths['y_test'])['sentiment']

    logger.info("Loaded train/test CSVs. Sizes: X_train=%d, X_test=%d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


# -------------------------- sklearn models (TF-IDF + classic classifiers) --------------------------

def sklearn_pipeline_cv(X, y, n_splits = 5, random_state = 42):
    """Train and cross-validate a set of scikit-learn models using TF-IDF features.

    Returns a dictionary with model names -> best estimator and CV metrics.
    """
    results = {}

    tfidf = TfidfVectorizer(max_features=50_000, ngram_range=(1,2))

    estimators = {
        'logistic': LogisticRegression(max_iter=200, solver='saga'),
        'svc': LinearSVC(max_iter=10_000),
        'nb': MultinomialNB(),
        'rf': RandomForestClassifier(n_estimators=200, n_jobs=-1),
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for name, estimator in estimators.items():
        pipe = Pipeline([('tfidf', tfidf), ('clf', estimator)])
        logger.info("Running CV for sklearn model: %s", name)

        # Use cross_val_score for accuracy and f1 (macro)
        acc_scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
        f1_scores = cross_val_score(pipe, X, y, cv=skf, scoring='f1_macro', n_jobs=-1)

        logger.info("%s - accuracy: mean=%.4f std=%.4f | f1_macro: mean=%.4f std=%.4f",
                    name, acc_scores.mean(), acc_scores.std(), f1_scores.mean(), f1_scores.std())

        # Fit on all data for later use
        pipe.fit(X, y)

        # save pipeline
        model_path = MODELS_DIR / f"sklearn_{name}.joblib"
        joblib.dump(pipe, model_path)
        logger.info("Saved sklearn pipeline to %s", model_path)

        results[name] = {
            'pipeline': pipe,
            'accuracy_cv_mean': float(acc_scores.mean()),
            'accuracy_cv_std': float(acc_scores.std()),
            'f1_cv_mean': float(f1_scores.mean()),
            'f1_cv_std': float(f1_scores.std()),
            'model_path': str(model_path)
        }

    # Save metadata
    meta_path = ARTIFACTS_DIR / 'sklearn_results.json'
    with open(meta_path, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2)
    logger.info("Saved sklearn results metadata to %s", meta_path)

    return results


# -------------------------- PyTorch model (simple Text classifier using TF-IDF or token embedding)

class TfidfDataset(Dataset):
    def __init__(self, X_tfidf, y):
        # X_tfidf: scipy sparse matrix or numpy array
        if hasattr(X_tfidf, 'toarray'):
            self.X = X_tfidf.toarray().astype(np.float32)
        else:
            self.X = np.asarray(X_tfidf).astype(np.float32)
        self.y = np.asarray(y).astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleTorchNN(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(hidden//2, 2)
        )

    def forward(self, x):
        return self.net(x)


def train_torch_tfidf(X, y, n_splits = 3, epochs = 6, batch_size = 64):
    """Cross-validate a simple PyTorch model on TF-IDF features. Returns best model path and metrics."""
    logger.info("Preparing TF-IDF for PyTorch training")
    tfidf = TfidfVectorizer(max_features=50_000, ngram_range=(1,2))
    X_tfidf = tfidf.fit_transform(X)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("PyTorch device: %s", device)

    fold_metrics = []
    best_fold_path = None
    best_acc = -1.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_tfidf, y), start=1):
        logger.info("PyTorch fold %d/%d", fold, n_splits)
        X_tr = X_tfidf[train_idx, :] # type: ignore
        X_val = X_tfidf[val_idx, :] # type: ignore
        y_tr = y.iloc[train_idx].values
        y_val = y.iloc[val_idx].values

        train_ds = TfidfDataset(X_tr, y_tr)
        val_ds = TfidfDataset(X_val, y_val)

        tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        model = SimpleTorchNN(input_dim=X_tfidf.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(1, epochs+1):
            model.train()
            total_loss = 0.0
            for xb, yb in tr_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                total_loss += loss.item() * xb.size(0)

            avg_loss = total_loss / len(train_ds)
            logger.info("Fold %d Epoch %d training loss: %.4f", fold, epoch, avg_loss)

        # Validation
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1).cpu().numpy()
                preds.extend(pred.tolist())
                trues.extend(yb.numpy().tolist())

        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average='macro')
        logger.info("Fold %d - val accuracy: %.4f f1_macro: %.4f", fold, acc, f1)

        # save fold model if best
        fold_path = MODELS_DIR / f"torch_tfidf_fold{fold}.pt"
        torch.save({'model_state_dict': model.state_dict(), 'input_dim': X_tfidf.shape[1]}, fold_path)

        fold_metrics.append({'fold': fold, 'accuracy': float(acc), 'f1_macro': float(f1), 'model_path': str(fold_path)})

        if acc > best_acc:
            best_acc = acc
            best_fold_path = str(fold_path)

    # save tfidf vectorizer
    vec_path = ARTIFACTS_DIR / 'tfidf_for_torch.joblib'
    joblib.dump(tfidf, vec_path)
    logger.info("Saved TF-IDF vectorizer for PyTorch to %s", vec_path)

    out = {'folds': fold_metrics, 'best_model_path': best_fold_path, 'tfidf_path': str(vec_path)}
    with open(ARTIFACTS_DIR / 'torch_results.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)

    return out


# -------------------------- TensorFlow / Keras model (Tokenization + simple NN or LSTM)
def build_keras_model(vocab_size, maxlen):
    model = keras.Sequential([
        layers.Input(shape=(maxlen,), dtype='int32'),
        layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=maxlen),
        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
    return model


# -------------------------- Hugging Face fine-tuning (optional, if transformers available)

def train_huggingface(X_train, X_val, y_train, y_val,
                      model_name = 'distilbert-base-uncased', epochs = 2, batch_size = 8):
    if not HF_AVAILABLE:
        logger.warning("Hugging Face not available (transformers/datasets). Skipping HF training.")
        return {}

    logger.info("Preparing Hugging Face dataset and tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)

    train_ds = HFDataset.from_dict({'text': X_train.tolist(), 'label': y_train.astype(int).tolist()})
    val_ds = HFDataset.from_dict({'text': X_val.tolist(), 'label': y_val.astype(int).tolist()})

    train_ds = train_ds.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=256), batched=True)
    val_ds = val_ds.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=256), batched=True)

    train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    args = TrainingArguments(
        output_dir=str(MODELS_DIR / 'hf_output'),
        save_strategy='epoch',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        logging_dir=str(ARTIFACTS_DIR / 'hf_logs')
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            'accuracy': (preds == labels).mean(),
            'f1_macro': float(f1_score(labels, preds, average='macro'))
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
# -------------------------- Keras cross-validation

def train_keras_cv(X, y, n_splits=3, epochs=4, maxlen=256):
    tokenizer = Tokenizer(num_words=50_000, oov_token='<OOV>')
    tokenizer.fit_on_texts(X)

    sequences = tokenizer.texts_to_sequences(X)
    sequences = pad_sequences(sequences, maxlen=maxlen)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []
    best_path = None
    best_acc = -1.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(sequences, y), start=1):
        logger.info("Keras fold %d/%d", fold, n_splits)
        X_tr = sequences[train_idx]
        X_val = sequences[val_idx]
        y_tr = y.iloc[train_idx].values
        y_val = y.iloc[val_idx].values

        callbacks = [EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)]
        model = build_keras_model(vocab_size=min(len(tokenizer.word_index)+1, 50_000), maxlen=maxlen)

        history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs, batch_size=64, callbacks=callbacks, verbose="auto")
        val_acc = float(history.history['val_accuracy'][-1])
        val_loss = float(history.history['val_loss'][-1])
        logger.info("Keras fold %d - val_acc: %.4f val_loss: %.4f", fold, val_acc, val_loss)

        model_path = MODELS_DIR / f"keras_fold{fold}"
        model.save(model_path)
        logger.info("Saved Keras model to %s", model_path)

        fold_metrics.append({'fold': fold, 'val_acc': val_acc, 'val_loss': val_loss, 'model_path': str(model_path)})

        if val_acc > best_acc:
            best_acc = val_acc
            best_path = str(model_path)

    # save tokenizer
    tok_path = ARTIFACTS_DIR / 'keras_tokenizer.joblib'
    joblib.dump(tokenizer, tok_path)
    logger.info("Saved Keras tokenizer to %s", tok_path)

    out = {'folds': fold_metrics, 'best_model_path': best_path, 'tokenizer_path': str(tok_path)}
    with open(ARTIFACTS_DIR / 'keras_results.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)

    return out

# -------------------------- Hugging Face fine-tuning (optional, if transformers available)
    trainer.train()
    best_model_dir = MODELS_DIR / 'hf_output' / 'checkpoint-best'
    # Save tokenizer and model
    hf_save_dir = MODELS_DIR / f"hf_{model_name.replace('/', '_')}"
    hf_save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(hf_save_dir)
    tokenizer.save_pretrained(hf_save_dir)

    logger.info("Saved Hugging Face model + tokenizer to %s", hf_save_dir)
    return {'hf_save_dir': str(hf_save_dir)}


# -------------------------- Utilities: evaluation on test set and selection


def evaluate_model_on_test(pipeline, X_test, y_test):
    """
    Fill Here
    """
    pred = pipeline.predict(X_test)
    probs = None
    try:
        probs = pipeline.predict_proba(X_test)[:,1]
    except Exception:
        pass

    metrics = {
        'accuracy': float(accuracy_score(y_test, pred)),
        'f1_macro': float(f1_score(y_test, pred, average='macro')),
        'precision': float(precision_score(y_test, pred, average='macro')),
        'recall': float(recall_score(y_test, pred, average='macro')),
    }
    if probs is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_test, probs))
        except Exception:
            pass

    return metrics


# -------------------------- Orchestrator: bring everything together


def main():
    logger.info("Starting full modeling pipeline")
    X_train, X_test, y_train, y_test = load_csvs()

    # Combine train for CV where needed
    X_all = pd.concat([X_train, X_test], ignore_index=True)
    y_all = pd.concat([y_train, y_test], ignore_index=True)

    # 1) sklearn CV and save
    sklearn_results = sklearn_pipeline_cv(X_all, y_all, n_splits=5)

    # Evaluate sklearn models on hold-out test set (the original X_test / y_test)
    sklearn_eval = {}
    for name, info in sklearn_results.items():
        pipe = info['pipeline']
        metrics = evaluate_model_on_test(pipe, X_test, y_test)
        sklearn_eval[name] = metrics
        logger.info("Sklearn model %s test metrics: %s", name, metrics)

    with open(ARTIFACTS_DIR / 'sklearn_test_eval.json', 'w', encoding='utf-8') as fh:
        json.dump(sklearn_eval, fh, indent=2)

    # 2) PyTorch using TF-IDF cross-validation
    torch_results = train_torch_tfidf(X_all, y_all, n_splits=3, epochs=4)

    # 3) Keras cross-validation
    keras_results = train_keras_cv(X_all, y_all, n_splits=3, epochs=4)

    # 4) Hugging Face (optional) - we'll do a train/val split from X_train
    hf_results = {}
    if HF_AVAILABLE:
        # small validation split
        split_idx = int(0.8 * len(X_train))
        hf_results = train_huggingface(X_train.iloc[:split_idx], X_train.iloc[split_idx:], y_train.iloc[:split_idx], y_train.iloc[split_idx:], epochs=1)

    # 5) Summarize and pick the best classical model by CV f1
    best_sklearn = max(sklearn_results.items(), key=lambda kv: kv[1]['f1_cv_mean'])
    logger.info("Best sklearn model by CV f1: %s", best_sklearn[0])

    summary = {
        'sklearn': {k: v for k, v in sklearn_results.items()},
        'sklearn_test_eval': sklearn_eval,
        'torch': torch_results,
        'keras': keras_results,
        'huggingface': hf_results,
    }

    with open(ARTIFACTS_DIR / 'modeling_summary.json', 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)

    logger.info("Modeling pipeline finished. Artifacts saved to %s and %s", MODELS_DIR, ARTIFACTS_DIR)


if __name__ == '__main__':
    main()
