"""
Optimized Training Module (Hybrid AI).
Trains Classical (Sklearn), PyTorch, and TensorFlow models efficiently.
Saves Keras models natively (.keras) and others via Joblib to prevent serialization errors.
Implements specific optimizations for CPU and RAM usage.
"""
import os
import json

# Limits usage to 4 threads to prevent CPU overheating
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pandas as pd
import joblib
import logging
from typing import Any, Tuple

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# UI
from rich.console import Console
from rich.table import Table

from config import Config

console = Console()
logger = logging.getLogger("Trainer")

# --- DEFINITIONS ---

class SimpleTorchNN(nn.Module):
    """
    A simple Feed-Forward Neural Network in PyTorch.
    Structure: Input -> Linear -> ReLU -> Dropout -> Linear -> Sigmoid.
    """
    def __init__(self, input_dim):
        super(SimpleTorchNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.layer2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return self.sigmoid(x)

class SparseTorchDataset(Dataset):
    """
    Custom PyTorch Dataset that handles Sparse Matrices.
    Converts sparse data to dense tensors on-the-fly to save RAM.
    """
    def __init__(self, X_sparse, y):
        self.X_sparse = X_sparse
        self.y = np.array(y) 
    def __len__(self):
        return self.X_sparse.shape[0]
    def __getitem__(self, idx):
        row_dense = self.X_sparse[idx].toarray().squeeze()
        label = self.y[idx]
        return torch.tensor(row_dense, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class UnifiedModelWrapper(BaseEstimator):
    """
    A wrapper class that standardizes the `predict` API across Sklearn, Keras, and PyTorch.
    Allows the inference script to work without knowing the underlying framework.
    """
    def __init__(self, model, vectorizer, framework: str):
        self.model = model
        self.vectorizer = vectorizer
        self.framework = framework

# --- TRAINER ---

class ModelTrainer:
    """
    Main controller class for the training process.
    Manages data loading, model training for different frameworks, comparison, and saving.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=Config.MAX_FEATURES, ngram_range=(1, 2), stop_words='english')
        self.best_accuracy = 0.0
        self.best_model_wrapper = None
        self.best_model_name = ""

    def load_data(self):
        """Loads processed training and testing CSV files."""
        if not Config.TRAIN_DATA_PATH.exists():
            raise FileNotFoundError("Data not found. Run data_processor.py first.")
        train = pd.read_csv(Config.TRAIN_DATA_PATH).dropna()
        test = pd.read_csv(Config.TEST_DATA_PATH).dropna()
        return train['text'], train['label'], test['text'], test['label']

    def train_sklearn(self, X_train_vec, y_train, X_test_vec, y_test):
        """Trains classical Scikit-Learn models (Logistic Regression, SVM, RF)."""
        models = {
            'LogisticRegression': LogisticRegression(max_iter=500, solver='saga', n_jobs=2),
            'LinearSVC': LinearSVC(max_iter=1000, dual="auto"),
            'RandomForest': RandomForestClassifier(n_estimators=50, n_jobs=4) 
        }
        results = {}
        for name, clf in models.items():
            clf.fit(X_train_vec, y_train)
            acc = accuracy_score(y_test, clf.predict(X_test_vec))
            results[name] = {'model': clf, 'acc': acc}
        return results

    def train_keras(self, X_train_vec, y_train, X_test_vec, y_test):
        """
        Trains a TensorFlow/Keras model.
        Uses a custom batch generator to handle sparse inputs efficiently.
        """
        input_dim = X_train_vec.shape[1]
        def batch_generator(X_sparse, y, batch_size):
            num_samples = X_sparse.shape[0]
            while True:
                indices = np.arange(num_samples)
                np.random.shuffle(indices)
                for start in range(0, num_samples, batch_size):
                    end = min(start + batch_size, num_samples)
                    batch_idx = indices[start:end]
                    yield X_sparse[batch_idx].toarray(), y.iloc[batch_idx]

        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        es = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
        
        model.fit(
            batch_generator(X_train_vec, y_train, 32),
            steps_per_epoch=X_train_vec.shape[0] // 32,
            epochs=3, verbose=0, callbacks=[es]
        )
        loss, acc = model.evaluate(X_test_vec.toarray(), y_test, batch_size=64, verbose=0)
        return {'TensorFlow_Keras': {'model': model, 'acc': acc}}

    def train_torch(self, X_train_vec, y_train, X_test_vec, y_test):
        """
        Trains a PyTorch model.
        Uses a SparseTorchDataset to prevent RAM spikes.
        """
        input_dim = X_train_vec.shape[1]
        train_dataset = SparseTorchDataset(X_train_vec, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        model = SimpleTorchNN(input_dim)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(3):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.unsqueeze(1))
                loss.backward()
                optimizer.step()
                
        model.eval()
        test_dataset = SparseTorchDataset(X_test_vec, y_test)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        preds = []
        targets = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                out = model(X_batch)
                preds.extend((out > 0.5).float().numpy().flatten())
                targets.extend(y_batch.numpy())
        acc = accuracy_score(targets, preds)
        return {'PyTorch_NN': {'model': model, 'acc': acc}}

    def run(self):
        """
        Executes the training pipeline:
        1. Vectorize Data
        2. Train Models (Sklearn, TF, Torch)
        3. Compare Performance
        4. Save Best Model and Metadata
        """
        console.rule("[bold blue]Training Frameworks (Safe Mode)[/bold blue]")
        X_train_txt, y_train, X_test_txt, y_test = self.load_data()
        
        console.print("[cyan]Vectorizing...[/cyan]")
        X_train_vec = self.vectorizer.fit_transform(X_train_txt)
        X_test_vec = self.vectorizer.transform(X_test_txt)
        
        all_results = {}
        all_results.update(self.train_sklearn(X_train_vec, y_train, X_test_vec, y_test))
        console.print("[yellow]Sklearn Done.[/yellow]")
        all_results.update(self.train_keras(X_train_vec, y_train, X_test_vec, y_test))
        console.print("[yellow]Keras Done.[/yellow]")
        all_results.update(self.train_torch(X_train_vec, y_train, X_test_vec, y_test))
        console.print("[yellow]Torch Done.[/yellow]")

        table = Table(title="Leaderboard")
        table.add_column("Model", style="cyan")
        table.add_column("Accuracy", justify="right", style="green")

        for name, data in all_results.items():
            acc = data['acc']
            table.add_row(name, f"{acc:.4f}")
            if acc > self.best_accuracy:
                self.best_accuracy = acc
                self.best_model_name = name
                raw_model = data['model']
                fw_type = 'sklearn'
                if 'TensorFlow' in name: fw_type = 'keras'
                elif 'PyTorch' in name: fw_type = 'torch'
                self.best_model_wrapper = UnifiedModelWrapper(raw_model, self.vectorizer, fw_type)

        console.print(table)
        
        # --- ROBUST SAVING LOGIC ---
        if self.best_model_wrapper:
            save_dir = Config.MODELS_DIR
            # Clean old files
            for f in save_dir.glob("*"):
                try: f.unlink()
                except: pass

            metadata = {"framework": self.best_model_wrapper.framework}
            
            if self.best_model_wrapper.framework == 'keras':
                # Save Keras natively
                self.best_model_wrapper.model.save(save_dir / "best_model.keras")
                joblib.dump(self.best_model_wrapper.vectorizer, save_dir / "vectorizer.joblib")
                console.print(f"\n[bold green]✔ Saved Keras Model (.keras) to:[/bold green] {save_dir}")
            else:
                # Save others via joblib
                joblib.dump(self.best_model_wrapper, save_dir / "best_model.joblib")
                console.print(f"\n[bold green]✔ Saved Model (.joblib) to:[/bold green] {save_dir}")

            # Save metadata so app knows how to load
            with open(save_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()