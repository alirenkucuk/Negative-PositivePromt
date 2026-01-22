# 🧠 Hybrid AI Sentiment Analysis Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

A professional, production-ready Machine Learning pipeline that trains and compares **Classical ML**, **TensorFlow (Keras)**, and **PyTorch** models simultaneously to classify movie reviews. It features a state-of-the-art visualizer to "open the black box" of the AI.

> 📖 **Read the Story Behind the Code:** > [**Opening the Black Box: Hybrid Sentiment Analysis and Model Explainability**](https://medium.com/@alirenkucuk/opening-the-black-box-hybrid-sentiment-analysis-and-model-explainability-with-tensorflow-pytorch-23bbf785a3ec?postPublishedType=initial) on Medium.

---

## 🚀 Features

* **Hybrid Training Architecture:** Automatically trains and races `LogisticRegression`, `LinearSVC`, `RandomForest`, `TensorFlow (NN)`, and `PyTorch (NN)` against each other.
* **Neural Visualization:** Interactive **Heatmaps** showing how individual neurons activate for specific words.
* **Explainable AI (LIME):** Breaks down prediction logic to show which words influenced the decision (Positive vs Negative).
* **Production Grade:** Includes robust error handling, memory optimization for large datasets (Sparse Matrix handling), and safe serialization for Keras/Torch models.
* **Modern UI:** A Cyberpunk-styled web interface built with Streamlit.

## 📊 Benchmark Results

The pipeline evaluated 5 different architectures on the IMDB Dataset (50K reviews).

| Model Architecture | Accuracy | Status |
| :--- | :--- | :--- |
| **TensorFlow (Keras)** | **89.27%** | 🏆 **Winner** |
| PyTorch (Neural Net) | 89.23% | 🥈 Runner Up |
| Logistic Regression | 88.98% | Very Close |
| Linear SVC | 88.90% | Competitive |
| Random Forest | 84.32% | Baseline |

> *Note: The system automatically selects and saves the best performing model (TensorFlow in this case) for inference.*

## 📂 Project Structure

```text
├── artifacts/       # Logs and training metadata
├── data/            # Dataset storage
├── models/          # Saved serialized models (keras/joblib)
├── config.py        # Centralized configuration
├── data_processor.py# ETL Pipeline: Cleaning & Lemmatization
├── trainer.py       # Multi-Framework Training Logic
├── predict.py       # CLI Inference Script
├── app.py           # Web Application (Streamlit)
└── requirements.txt # Dependencies
🛠️ Installation
1. Clone the repository:

Bash
git clone [https://github.com/alirenkucuk/Negative-PositivePromt.git](https://github.com/alirenkucuk/Negative-PositivePromt.git)
cd Negative-PositivePromt
2. Create a clean environment (Recommended):

Bash
conda create -n ai_env python=3.10 -y
conda activate ai_env
3. Install dependencies:

Bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
4. Setup Data:

The repository structure includes the data/ folder.

Ensure IMDBDataset.csv is present in data/ (Download from Kaggle if not included due to size limits).

⚡ Usage
1. Data Processing
Clean raw text and prepare training files (Train/Test split).

Bash
python data_processor.py
2. Train All Frameworks
Run the race between Sklearn, TF, and Torch. This will save the best model to models/.

Bash
python trainer.py
3. Visual Interface (Web App)
Launch the visualizer to see Heatmaps and LIME graphs.

Bash
python -m streamlit run app.py
4. CLI Prediction
Quick check from terminal without launching the UI.

Bash
python predict.py --text "The cinematography was great but the plot was boring."
🧠 Visualization Preview
Neural Activation Heatmap: See which words trigger the hidden neurons in the Deep Learning model.

LIME Explanation: Understand why the model labeled a review as Positive or Negative.

Dataset provided by Kaggle.