"""
Data processing module (ETL Pipeline).
Handles loading raw data, cleaning text, performing lemmatization, 
and splitting data into training and testing sets.
"""
import pandas as pd
import spacy
import re
import logging
from typing import Tuple, List
from rich.console import Console
from rich.progress import track
from sklearn.model_selection import train_test_split
from config import Config

# Setup Rich Console and Logger
console = Console()
logging.basicConfig(
    level=logging.INFO, 
    format="%(message)s", 
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("DataProcessor")

class TextPreprocessor:
    """
    A class to handle the end-to-end text preprocessing pipeline.
    Uses spaCy for linguistic operations and Regex for noise removal.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initializes the preprocessor by loading the specified spaCy model.
        
        Args:
            spacy_model (str): The name of the spaCy language model to load.
        """
        try:
            self.nlp = spacy.load(spacy_model, disable=["ner", "parser"])
        except OSError:
            console.print(f"[bold red]Error:[/bold red] Model '{spacy_model}' not found. Run: [green]python -m spacy download {spacy_model}[/green]")
            raise

    def clean_text(self, text: str) -> str:
        """
        Performs basic text cleaning using Regular Expressions.
        Removes HTML tags, URLs, numbers, and special characters.
        
        Args:
            text (str): The raw input string.
            
        Returns:
            str: The cleaned, lowercased string.
        """
        text = str(text).lower()
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'http\S+', '', text) # Remove URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove special chars/numbers
        return text.strip()

    def lemmatize_batch(self, texts: List[str], batch_size: int = 100) -> List[str]:
        """
        Efficiently lemmatizes a list of texts using spaCy's pipe method.
        Removes stopwords and short words during the process.
        
        Args:
            texts (List[str]): List of cleaned strings.
            batch_size (int): Number of documents to process in one go (for speed).
            
        Returns:
            List[str]: List of lemmatized strings (tokens joined by space).
        """
        lemmatized_texts = []
        
        # Using Rich progress bar
        total_docs = len(texts)
        description = "[cyan]Lemmatizing reviews...[/cyan]"
        
        for doc in track(self.nlp.pipe(texts, batch_size=batch_size), description=description, total=total_docs):
            tokens = [token.lemma_ for token in doc if not token.is_stop and len(token.text) > 2]
            lemmatized_texts.append(" ".join(tokens))
            
        return lemmatized_texts

    def process_pipeline(self):
        """
        Orchestrates the full ETL pipeline:
        1. Load Raw CSV
        2. Clean Text (Regex)
        3. Lemmatize (SpaCy)
        4. Encode Labels
        5. Split Train/Test
        6. Save Processed CSVs
        """
        console.rule("[bold blue]Starting Data Processing[/bold blue]")
        
        # 1. Load Data
        if not Config.RAW_DATA_PATH.exists():
            console.print(f"[bold red]Error:[/bold red] Dataset not found at {Config.RAW_DATA_PATH}")
            return

        df = pd.read_csv(Config.RAW_DATA_PATH)
        console.print(f"[green]✔[/green] Loaded dataset with shape {df.shape}")

        # 2. Basic Cleaning
        df['cleaned'] = df['review'].apply(self.clean_text)
        
        # 3. Lemmatization (Heavy lifting)
        # Note: Process full dataset for production
        df['processed_text'] = self.lemmatize_batch(df['cleaned'].tolist())

        # 4. Label Encoding
        df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

        # 5. Split
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], 
            df['label'], 
            test_size=Config.TEST_SIZE, 
            random_state=Config.RANDOM_STATE,
            stratify=df['label']
        )

        # 6. Save
        train_df = pd.DataFrame({'text': X_train, 'label': y_train})
        test_df = pd.DataFrame({'text': X_test, 'label': y_test})
        
        train_df.to_csv(Config.TRAIN_DATA_PATH, index=False)
        test_df.to_csv(Config.TEST_DATA_PATH, index=False)
        
        console.print(f"[green]✔[/green] Data saved to [bold]{Config.DATA_DIR}[/bold]")

if __name__ == "__main__":
    processor = TextPreprocessor()
    processor.process_pipeline()