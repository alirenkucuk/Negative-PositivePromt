# File: x_and_y_data.py
"""
Simplified and cleaned preprocessing script for the IMDB dataset.
- Removes unnecessary imports and unused code (no plotting, no matplotlib).
- Keeps text cleaning focused: lowercasing, HTML unescape & tag removal, URL/email/punctuation removal,
  optional number removal and minimum token length.
- Saves raw and cleaned X/y and train/test splits to data/.
"""
import os
import logging
import re
import html
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def ensure_dir_exists(path: str) -> None:
    """Ensure parent directory for `path` exists."""
    parent = os.path.dirname(path)
    if not parent:
        return
    os.makedirs(parent, exist_ok=True)


def load_csv_or_fail(path: str, expected_columns: Optional[list] = None) -> pd.DataFrame:
    if not os.path.isfile(path):
        logger.error("File not found: %s", path)
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if expected_columns is not None:
        missing = [c for c in expected_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")
    logger.info("Loaded %s rows from %s", len(df), path)
    return df


def clean_text(series: pd.Series,
               lower: bool = True,
               remove_html: bool = True,
               remove_urls: bool = True,
               remove_emails: bool = True,
               remove_punct: bool = True,
               remove_numbers: bool = False,
               min_word_length: int = 3) -> pd.Series:
    """Clean a pandas Series of text and return cleaned Series.

    Behavior:
    - non-string entries become empty string
    - optional HTML unescape + tag removal
    - optional URL and email removal
    - lowercase (default True)
    - remove punctuation (keep a-z and 0-9)
    - optional removal of pure numeric tokens
    - remove short tokens shorter than min_word_length
    """

    if not isinstance(series, pd.Series):
        raise TypeError("clean_text expects a pandas Series")

    url_re = re.compile(r"http\S+|www\.\S+")
    email_re = re.compile(r"\S+@\S+")
    html_tag_re = re.compile(r"<[^>]+>")
    non_allowed_re = re.compile(r"[^a-z0-9\s]")
    number_token_re = re.compile(r"\b\d+\b")

    def _clean(doc: object) -> str:
        if not isinstance(doc, str):
            return ""
        text = doc
        if remove_html:
            text = html.unescape(text)
            text = html_tag_re.sub(' ', text)
        if remove_urls:
            text = url_re.sub(' ', text)
        if remove_emails:
            text = email_re.sub(' ', text)
        if lower:
            text = text.lower()
        if remove_punct:
            text = non_allowed_re.sub(' ', text)
        if remove_numbers:
            text = number_token_re.sub(' ', text)
        # normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""
        tokens = [t for t in text.split() if len(t) >= (min_word_length or 0)]
        return " ".join(tokens)

    logger.info("Cleaning text series (%d rows)", len(series))
    cleaned = series.fillna("").astype(str).map(_clean)
    # show small sample
    for i in range(min(3, len(series))):
        logger.debug("BEFORE: %s", series.iloc[i])
        logger.debug("AFTER : %s", cleaned.iloc[i])
    return cleaned


def save_series(series: pd.Series, path: str, column_name: Optional[str] = None) -> None:
    ensure_dir_exists(path)
    col = column_name or (series.name if series.name else 'value')
    df = pd.DataFrame({col: series.reset_index(drop=True)})
    df.to_csv(path, index=False)
    logger.info("Saved %s (rows=%d)", path, len(df))


def create_train_test_split(X: pd.Series, y: pd.Series, test_size: float = 0.2, random_state: int = 1234, stratify: bool = True):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    stratify_arg = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg)
    logger.info("Train/test split: %d / %d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def main():
    try:
        cwd = os.getcwd()
        data_dir = os.path.join(cwd, 'data')
        data_path = os.path.join(data_dir, 'IMDBDataset.csv')

        df = load_csv_or_fail(data_path, expected_columns=['review', 'sentiment'])

        X_raw = df['review']
        y = df['sentiment'].map({'positive': 1, 'negative': 0})

        # save raw
        save_series(X_raw, os.path.join(data_dir, 'X_raw.csv'), column_name='review')
        save_series(y, os.path.join(data_dir, 'y.csv'), column_name='sentiment')

        # clean
        X = clean_text(X_raw, lower=True, remove_html=True, remove_urls=True, remove_emails=True, remove_punct=True, remove_numbers=False, min_word_length=3)
        save_series(X, os.path.join(data_dir, 'X.csv'), column_name='review')

        # train/test
        X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.2, random_state=42, stratify=True)

        save_series(X_train, os.path.join(data_dir, 'X_train.csv'), column_name='review')
        save_series(X_test, os.path.join(data_dir, 'X_test.csv'), column_name='review')
        save_series(y_train, os.path.join(data_dir, 'y_train.csv'), column_name='sentiment')
        save_series(y_test, os.path.join(data_dir, 'y_test.csv'), column_name='sentiment')

        logger.info("Preprocessing finished successfully.")

    except Exception as e:
        logger.exception("Error in main: %s", e)
        raise


if __name__ == '__main__':
    main()

