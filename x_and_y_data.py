# Project name: Negative Review Classifier
# Goal: Classify movie reviews as negative or positive
# File Name: x_and_y_data.py
# Writer: Ali Eren Küçük
# File Goal: Load the IMDB dataset, preprocess it, and save X and y to CSV files

import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
import re
import html


# -------------------------- Logging setup --------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("All libraries imported successfully.")
logger.info("Numpy version: %s", np.__version__)
logger.info("Pandas version: %s", pd.__version__)
logger.info("Matplotlib version: %s", mpl.__version__)
logger.info("OS name: %s", os.name)


def ensure_dir_exists(path: str):
    """Ensure the parent directory for `path` exists.

    If the parent directory doesn't exist, it is created (recursive).
    """
    directory = os.path.dirname(path)
    if not directory:
        logger.debug("No parent directory for path '%s' (file will be created in current directory).", path)
        return

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.info("Created directory: %s", directory)
    else:
        logger.debug("Directory already exists: %s", directory)


def load_csv_or_fail(path, expected_columns=None, nrows_preview=5):
    if not os.path.isfile(path):
        logger.error("File not found: %s", path)
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded CSV from %s with shape %s", path, df.shape)

    if expected_columns is not None:
        missing = [c for c in expected_columns if c not in df.columns]
        if missing:
            logger.error("Missing expected columns: %s", missing)
            raise ValueError(f"Missing expected columns: {missing}")
        logger.debug("All expected columns are present: %s", expected_columns)

    logger.info("Data preview (first %d rows): %s", nrows_preview, df.head(nrows_preview).to_string(index=False))
    return df


def clean_text(series: pd.Series,
               lower: bool = True,
               remove_html: bool = True,
               remove_urls: bool = True,
               remove_emails: bool = True,
               remove_punct: bool = True,
               remove_numbers: bool = False,
               min_word_length: int = 3) -> pd.Series:
    """Clean text contained in a pandas Series.

    This variant does NOT use NLTK. Stopword removal and lemmatization
    have been removed so that NLTK-dependent processing can be handled in
    a separate module/file.

    Parameters
    ----------
    series : pd.Series
        Input text series (one document per row). Non-string values are converted to empty string.
    lower : bool
        Convert text to lowercase.
    remove_html : bool
        Remove HTML tags and unescape HTML entities.
    remove_urls : bool
        Remove URLs (http/https/www).
    remove_emails : bool
        Remove email-like tokens.
    remove_punct : bool
        Remove punctuation and keep only letters/numbers/whitespace.
    remove_numbers : bool
        Remove whole numeric tokens (optional).
    min_word_length : int
        Drop tokens shorter than this length (e.g., 3 removes 1-2 letter words).

    Returns
    -------
    pd.Series
        Cleaned text series.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("clean_text expects a pandas Series as input")

    def _clean_doc(doc: str) -> str:
        if not isinstance(doc, str):
            return ''
        text = doc
        if remove_html:
            text = html.unescape(text)
            text = re.sub(r'<[^>]+>', ' ', text)

        if remove_urls:
            text = re.sub(r'http\S+|www\.\S+', ' ', text)

        if remove_emails:
            text = re.sub(r'\S+@\S+', ' ', text)

        if lower:
            text = text.lower()

        if remove_punct:
            # keep latin letters and numbers and whitespace
            text = re.sub(r'[^a-z0-9\s]', ' ', text)

        if remove_numbers:
            text = re.sub(r'\b\d+\b', ' ', text)

        # normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # tokenize
        tokens = text.split()

        # drop short tokens
        if min_word_length and min_word_length > 0:
            tokens = [t for t in tokens if len(t) >= min_word_length]

        return ' '.join(tokens)

    logger.info("Starting text cleaning on series of length %d (options: lower=%s, remove_html=%s, remove_urls=%s, remove_punct=%s, remove_numbers=%s, min_word_length=%d)",
                len(series), lower, remove_html, remove_urls, remove_punct, remove_numbers, min_word_length)

    cleaned = series.fillna('').astype(str).map(_clean_doc)

    logger.info("Completed text cleaning. Example before/after (first 3 rows):")
    for i in range(min(3, len(series))):
        logger.info("BEFORE: %s", series.iloc[i])
        logger.info("AFTER : %s", cleaned.iloc[i])

    return cleaned


def save_series(series, path, column_name=None):
    ensure_dir_exists(path)

    col = column_name or series.name or 'value'
    df = pd.DataFrame({col: series.reset_index(drop=True)})
    df.to_csv(path, index=False)

    logger.info("Saved Series to %s (column name: '%s', rows: %d)", path, col, len(df))


def create_train_test_split(X, y, test_size=0.2, random_state=1234, stratify=True):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    stratify_arg = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )

    logger.info(
        "Performed train/test split: train=%d rows, test=%d rows (test_size=%s, stratify=%s)",
        len(X_train), len(X_test), test_size, stratify,
    )

    return X_train, X_test, y_train, y_test


def main():
    try:
        current_directory = os.getcwd()
        logger.info("Current working directory: %s", current_directory)

        data_path = os.path.join(current_directory, 'data', 'IMDBDataset.csv')

        data = load_csv_or_fail(data_path, expected_columns=['review', 'sentiment'])

        logger.info("Dataset columns: %s", list(data.columns))
        logger.info("Missing values per column: %s", data.isnull().sum().to_dict())

        X_raw = data['review']
        y = data['sentiment'].map({'positive': 1, 'negative': 0})
        logger.info("Mapped sentiment to binary labels. Unique labels after mapping: %s", sorted(y.unique()))

        # Save raw X and y for reproducibility
        save_series(X_raw, os.path.join(current_directory, 'data', 'X_raw.csv'), column_name='review')
        save_series(y, os.path.join(current_directory, 'data', 'y.csv'), column_name='sentiment')

        # CLEANING: apply improved clean_text function (NLTK-free)
        X = clean_text(X_raw,
                       lower=True,
                       remove_html=True,
                       remove_urls=True,
                       remove_emails=True,
                       remove_punct=True,
                       remove_numbers=False,
                       min_word_length=3)

        # Save cleaned features
        save_series(X, os.path.join(current_directory, 'data', 'X.csv'), column_name='review')

        # Plot class distribution
        plot_class_distribution(y, show=True, save_path=os.path.join(current_directory, 'data', 'class_distribution.png'))

        # Train/test split
        X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.2, random_state=42, stratify=True)

        # Save splits
        save_series(X_train, os.path.join(current_directory, 'data', 'X_train.csv'), column_name='review')
        save_series(X_test, os.path.join(current_directory, 'data', 'X_test.csv'), column_name='review')
        save_series(y_train, os.path.join(current_directory, 'data', 'y_train.csv'), column_name='sentiment')
        save_series(y_test, os.path.join(current_directory, 'data', 'y_test.csv'), column_name='sentiment')

        logger.info("Data preprocessing completed and files saved successfully.")

    except Exception as exc:
        logger.exception("An error occurred in main(): %s", exc)
        raise


if __name__ == '__main__':
    main()
