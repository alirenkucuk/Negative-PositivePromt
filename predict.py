"""
CLI Inference module.
Allows for quick sentiment prediction from the command line without launching the web app.
Supports interactive mode and single-shot prediction.
"""
import os
import sys
import argparse
import joblib
import numpy as np

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# UI
from rich.console import Console
from rich.panel import Panel
from config import Config

# --- CRITICAL IMPORTS FOR UNPICKLING ---
# These libraries must be imported because the saved model object depends on them.
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.base import BaseEstimator
import json

console = Console()

# --- RE-DEFINING CLASSES ---
class SimpleTorchNN(nn.Module):
    """PyTorch Neural Network definition."""
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

class UnifiedModelWrapper(BaseEstimator):
    """Wrapper to handle different model frameworks."""
    def __init__(self, model, vectorizer, framework: str):
        self.model = model
        self.vectorizer = vectorizer
        self.framework = framework

    def predict(self, texts):
        if isinstance(texts, str): texts = [texts]
        vectors = self.vectorizer.transform(texts)
        
        if self.framework == 'sklearn':
            return self.model.predict(vectors)
        elif self.framework == 'keras':
            probs = self.model.predict(vectors.toarray(), verbose=0)
            return (probs > 0.5).astype("int32").flatten()
        elif self.framework == 'torch':
            dense_vectors = vectors.toarray()
            inputs = torch.tensor(dense_vectors, dtype=torch.float32)
            self.model.eval()
            with torch.no_grad():
                probs = self.model(inputs)
            return (probs > 0.5).float().numpy().flatten()

    def get_confidence(self, texts):
        if isinstance(texts, str): texts = [texts]
        vectors = self.vectorizer.transform(texts)

        if self.framework == 'sklearn':
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(vectors)[:, 1]
            return self.model.decision_function(vectors)
        elif self.framework == 'keras':
            return self.model.predict(vectors.toarray(), verbose=0).flatten()
        elif self.framework == 'torch':
            inputs = torch.tensor(vectors.toarray(), dtype=torch.float32)
            self.model.eval()
            with torch.no_grad():
                return self.model(inputs).numpy().flatten()

# --- MAIN INFERENCE LOGIC ---

def load_model():
    """
    Loads the best model using metadata to decide between Keras/Joblib loading.
    """
    save_dir = Config.MODELS_DIR
    meta_path = save_dir / "metadata.json"
    
    if not meta_path.exists():
        console.print("[bold red]Error:[/bold red] Metadata not found. Train the model first.")
        sys.exit(1)
        
    with open(meta_path, "r") as f:
        meta = json.load(f)

    console.print(f"[cyan]Loading {meta['framework'].upper()} model...[/cyan]")
    
    if meta['framework'] == 'keras':
        model = keras.models.load_model(save_dir / "best_model.keras")
        vec = joblib.load(save_dir / "vectorizer.joblib")
        return UnifiedModelWrapper(model, vec, 'keras')
    else:
        return joblib.load(save_dir / "best_model.joblib")

def predict_sentiment(text: str, model):
    """Predicts and displays sentiment in a rich panel."""
    try:
        prediction = model.predict(text)[0]
        confidence_raw = model.get_confidence(text)[0]
        
        if model.framework in ['keras', 'torch', 'logistic']:
            conf_percent = confidence_raw if prediction == 1 else (1 - confidence_raw)
        else:
            conf_percent = abs(confidence_raw)
            
        sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
        color = "green" if sentiment == "POSITIVE" else "red"
        emoji = "😊" if sentiment == "POSITIVE" else "😡"
        
        output_text = f"[bold]Text:[/bold] {text}\n"
        output_text += f"[bold]Sentiment:[/bold] [{color}]{sentiment} {emoji}[/{color}]\n"
        output_text += f"[dim]Model Framework: {model.framework.upper()} ({conf_percent:.2%})[/dim]"
        
        console.print(Panel(output_text, title="Prediction Result", expand=False, border_style=color))
        
    except Exception as e:
        console.print(f"[bold red]Prediction Error:[/bold red] {e}")

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description="Sentiment Analysis Predictor")
    parser.add_argument("--text", "-t", type=str, help="Text to analyze")
    args = parser.parse_args()

    model_wrapper = load_model()

    if args.text:
        predict_sentiment(args.text, model_wrapper)
    else:
        console.print("[yellow]Entering Interactive Mode (Type 'exit' to quit)[/yellow]")
        while True:
            user_input = console.input("\n[bold cyan]Enter review:[/bold cyan] ")
            if user_input.lower() in ['exit', 'quit']:
                break
            predict_sentiment(user_input, model_wrapper)

if __name__ == "__main__":
    main()