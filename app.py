"""
Streamlit Web Application.
Provides a visual interface to test the trained model.
Includes LIME explanation and Neural Network Weight Visualization.
"""
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from lime.lime_text import LimeTextExplainer
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from config import Config
import json

# --- CONFIG & PAGE SETUP ---
st.set_page_config(page_title="Sentiment AI Visualizer", page_icon="🧠", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stTextInput > div > div > input { background-color: #262730; color: white; }
    h1 { color: #00FFAA; text-align: center; }
    .metric-card { background-color: #262730; padding: 20px; border-radius: 10px; border-left: 5px solid #00FFAA; }
    /* Customize plots background */
    div[data-testid="stMarkdownContainer"] p { color: #cccccc; }
    </style>
""", unsafe_allow_html=True)

# --- CLASS DEFINITIONS ---
class SimpleTorchNN(nn.Module):
    """PyTorch Neural Network definition (Must match trainer.py)."""
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
    """Wrapper to handle different model frameworks during inference."""
    def __init__(self, model, vectorizer, framework: str):
        self.model = model
        self.vectorizer = vectorizer
        self.framework = framework

    def predict(self, texts):
        if isinstance(texts, str): texts = [texts]
        vectors = self.vectorizer.transform(texts)
        if self.framework == 'sklearn': return self.model.predict(vectors)
        elif self.framework == 'keras':
            probs = self.model.predict(vectors.toarray(), verbose=0)
            return (probs > 0.5).astype("int32").flatten()
        elif self.framework == 'torch':
            inputs = torch.tensor(vectors.toarray(), dtype=torch.float32)
            self.model.eval()
            with torch.no_grad(): probs = self.model(inputs)
            return (probs > 0.5).float().numpy().flatten()

    def predict_proba(self, texts):
        if isinstance(texts, str): texts = [texts]
        vectors = self.vectorizer.transform(texts)
        if self.framework == 'keras':
            prob_pos = self.model.predict(vectors.toarray(), verbose=0).flatten()
            return np.vstack([1-prob_pos, prob_pos]).T
        elif self.framework == 'torch':
            inputs = torch.tensor(vectors.toarray(), dtype=torch.float32)
            self.model.eval()
            with torch.no_grad(): prob_pos = self.model(inputs).numpy().flatten()
            return np.vstack([1-prob_pos, prob_pos]).T
        elif self.framework == 'sklearn':
            if hasattr(self.model, "predict_proba"): return self.model.predict_proba(vectors)
            else:
                d = self.model.decision_function(vectors)
                prob_pos = 1 / (1 + np.exp(-d))
                return np.vstack([1-prob_pos, prob_pos]).T

# --- ROBUST LOADER ---
@st.cache_resource
def load_model():
    """
    Loads the trained model based on the metadata.json file.
    Handles distinct loading logic for Keras (.keras) vs Joblib (.joblib).
    """
    save_dir = Config.MODELS_DIR
    meta_path = save_dir / "metadata.json"
    
    if not meta_path.exists():
        st.error("Model metadata not found! Please run 'python trainer.py' first.")
        st.stop()
        
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    if meta['framework'] == 'keras':
        model = keras.models.load_model(save_dir / "best_model.keras")
        vec = joblib.load(save_dir / "vectorizer.joblib")
        return UnifiedModelWrapper(model, vec, 'keras')
    else:
        return joblib.load(save_dir / "best_model.joblib")

try:
    model_wrapper = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# --- MAIN UI ---
st.title("🧠 AI Sentiment & Neural Visualizer")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Analyze Review")
    user_input = st.text_area("Enter review:", height=150, placeholder="Type here...")
    
    if st.button("Analyze Sentiment", type="primary"):
        if user_input:
            prediction = model_wrapper.predict([user_input])[0]
            probs = model_wrapper.predict_proba([user_input])[0]
            confidence = probs[1] if prediction == 1 else probs[0]
            sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
            color = "#00FFAA" if sentiment == "POSITIVE" else "#FF0055"
            
            st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid {color};">
                    <h2 style="color: {color}; margin:0;">{sentiment}</h2>
                    <p style="color: white; margin:0;">Confidence: <b>{confidence:.2%}</b></p>
                    <p style="color: gray; margin:0;">Model: {model_wrapper.framework.upper()}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # --- LIME EXPLAINER ---
            st.markdown("### 🔍 Decision Explanation (LIME)")
            with st.spinner("Analyzing context..."):
                explainer = LimeTextExplainer(class_names=['NEGATIVE', 'POSITIVE'])
                exp = explainer.explain_instance(user_input, model_wrapper.predict_proba, num_features=10)
                st.pyplot(exp.as_pyplot_figure())

            # --- NEURAL ACTIVATION MAP ---
            if model_wrapper.framework in ['keras', 'torch']:
                st.markdown("### 🕸️ Neural Activation Heatmap")
                st.info("Visualizing which neurons light up for each word in your text.")
                
                # Get vector and weights
                vector = model_wrapper.vectorizer.transform([user_input]).toarray()
                
                if model_wrapper.framework == 'keras':
                    # Layer 0 is Dense, Layer 1 is Dropout (in Keras 3 sequential)
                    weights = model_wrapper.model.layers[0].get_weights()[0]
                else:
                    weights = model_wrapper.model.layer1.weight.data.numpy().T
                
                # Find active words
                relevant_indices = np.where(vector[0] > 0)[0]
                
                if len(relevant_indices) > 0:
                    # Get the actual words corresponding to indices
                    feature_names = model_wrapper.vectorizer.get_feature_names_out()
                    active_words = [feature_names[i] for i in relevant_indices]
                    
                    # Filter weights for these words
                    relevant_weights = weights[relevant_indices, :50] # Show top 50 neurons
                    
                    # Plot Heatmap
                    fig, ax = plt.subplots(figsize=(12, max(4, len(active_words) * 0.5)))
                    sns.heatmap(relevant_weights, cmap="coolwarm", center=0, cbar=True, ax=ax,
                                yticklabels=active_words) 
                    
                    ax.set_xlabel("Neurons (Hidden Layer)")
                    ax.set_ylabel("Input Words")
                    plt.xticks(rotation=90)
                    plt.yticks(rotation=0)
                    st.pyplot(fig)
                else:
                    st.warning("No known words found in the model's vocabulary.")

with col2:
    st.subheader("📊 Model Stats")
    st.metric("Framework", model_wrapper.framework.capitalize())
    vocab = list(model_wrapper.vectorizer.vocabulary_.keys())
    st.metric("Vocabulary", f"{len(vocab):,}")
    
    st.markdown("### ☁️ Context Cloud")
    if len(vocab) > 0:
        wc = WordCloud(width=400, height=400, background_color='#0E1117', colormap='cool').generate(" ".join(np.random.choice(vocab, min(500, len(vocab)))))
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)