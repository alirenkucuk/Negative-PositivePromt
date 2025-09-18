# Project name: Negative Review Classifier
# Goal: Classify movie reviews as negative or positive
# File Name: x_and_y_data.py
# Writer: Ali Eren Küçük
# File Goal: POS tag and lemmatize the words in the reviews


from pathlib import Path
import logging
from typing import List, Optional, Tuple, Union

import pandas as pd

import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import data


# -------------------------- Logger setup --------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
_handler.setFormatter(_formatter)
if not logger.handlers:
    logger.addHandler(_handler)


# -------------------------- NLTK helpers --------------------------
_REQUIRED_NLTK_PACKAGES = [
    ('tokenizers/punkt', 'punkt'),
    ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
    ('corpora/wordnet', 'wordnet'),
    ('corpora/omw-1.4', 'omw-1.4'),
]


def ensure_nltk_resources(download_if_missing: bool = True) -> None:
    """
    Checks for required NLTK resources (and downloads them if requested).

    Parameters
    ----------
    download_if_missing : bool
        If True, attempts to download missing packages. If False and a required
        package is missing, a RuntimeError is raised.

    Raises
    ------
    RuntimeError
        If required resources are missing and download_if_missing is False.
    """
    missing = []
    for resource_path, package_name in _REQUIRED_NLTK_PACKAGES:
        try:
            nltk.data.find(resource_path)
            logger.debug("NLTK resource found: %s", package_name)
        except LookupError:
            logger.warning("NLTK resource missing: %s", package_name)
            missing.append(package_name)

    if missing:
        if not download_if_missing:
            raise RuntimeError(f"Missing NLTK packages: {missing}. `download_if_missing` is False, so no download was attempted.")

        for pkg in missing:
            try:
                logger.info("Downloading NLTK package: %s", pkg)
                nltk.download(pkg)
            except Exception as exc:
                logger.exception("Error while downloading NLTK package: %s", exc)
                raise


# Single lemmatizer instance (reused)
_LEMMATIZER = WordNetLemmatizer()


def nltk_tag_to_wordnet_tag(nltk_tag: str) -> Optional[str]:
    """
    Converts NLTK POS tags to WordNet-compatible tags.

    Parameters
    ----------
    nltk_tag : str
        POS tag produced by NLTK (e.g. 'NN', 'VB', 'JJ', 'RB').

    Returns
    -------
    Optional[str]
        WordNet-compatible tag (wordnet.NOUN/VERB/ADJ/ADV) or None.
    """
    if not nltk_tag:
        return None
    tag = nltk_tag[0].upper()
    if tag == 'J':
        return wordnet.ADJ
    if tag == 'V':
        return wordnet.VERB
    if tag == 'N':
        return wordnet.NOUN
    if tag == 'R':
        return wordnet.ADV
    return None


def tag_words(text: str) -> List[Tuple[str, str]]:
    """
    Returns POS tags for the words in the given text.

    Parameters
    ----------
    text : str
        Text to be tagged.

    Returns
    -------
    List[Tuple[str, str]]
        List of (word, POS) pairs.

    Notes
    -----
    A try/except is included; on error an empty list is returned and the error is logged.
    """
    try:
        if not isinstance(text, str):
            text = str(text)
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        logger.debug("Tagged %d tokens", len(tagged))
        return tagged
    except Exception:
        logger.exception("An error occurred during tag_words. Input: %r", text)
        return []


def lemmatize_sentence(sentence: str) -> str:
    """
    Takes a sentence, lemmatizes it using POS tags, and rejoins the tokens.

    Parameters
    ----------
    sentence : str
        Sentence to lemmatize.

    Returns
    -------
    str
        Lemmatized sentence (tokens separated by spaces).
    """
    try:
        if not isinstance(sentence, str):
            sentence = str(sentence)

        tagged = tag_words(sentence)
        lemmatized_tokens: List[str] = []
        for word, tag in tagged:
            wn_tag = nltk_tag_to_wordnet_tag(tag)
            if wn_tag is None:
                # If there is no suitable WordNet tag, do default lemmatize (acts like assuming noun)
                lemma = _LEMMATIZER.lemmatize(word)
            else:
                lemma = _LEMMATIZER.lemmatize(word, wn_tag)
            lemmatized_tokens.append(lemma)

        result = ' '.join(lemmatized_tokens)
        logger.debug("Lemmatized: %s -> %s", sentence[:60], result[:60])
        return result

    except Exception:
        logger.exception("Error during lemmatize_sentence. Input: %r", sentence)
        # On error, return the original input converted to a safe string
        return str(sentence)


def lemmatize_series(series: pd.Series) -> pd.Series:
    """
    Lemmatizes each text item in a pandas Series.

    Parameters
    ----------
    series : pd.Series
        pandas Series containing text.

    Returns
    -------
    pd.Series
        Series with lemmatized texts, same index and length.
    """
    try:
        logger.info("Starting lemmatization on series (row count: %d)", len(series))
        # apply the operation with fallback to original on error
        return series.fillna('').astype(str).apply(lambda s: lemmatize_sentence(s))
    except Exception:
        logger.exception("Unexpected error during lemmatize_series.")
        raise


def process_csv(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    text_column: Optional[str] = None,
    overwrite: bool = True,
) -> None:
    """
    Lemmatizes a text column in a CSV and saves the result to a new CSV.

    Behavior:
    - If text_column is not provided and the CSV has a single column, that column is used.
    - If text_column is not provided and there are multiple columns, the function looks for a column named 'review'.
      If not found, a ValueError is raised.
    - Other columns from the original DataFrame are preserved.

    Parameters
    ----------
    input_path : Union[str, Path]
        Path to the input CSV.
    output_path : Union[str, Path]
        Path to the output CSV.
    text_column : Optional[str]
        Name of the text column to process. If None, automatic selection is attempted.
    overwrite : bool
        If True, overwrite existing output file. If False and output exists, an error is raised.

    Raises
    ------
    FileNotFoundError
        If input_path does not exist.
    ValueError
        If a suitable text column cannot be found, or if overwrite is False and output_path exists.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path.exists() and not overwrite:
        logger.error("Output file already exists and overwrite=False: %s", output_path)
        raise ValueError(f"Output file already exists: {output_path}")

    try:
        df = pd.read_csv(input_path)
        logger.info("CSV loaded: %s (shape: %s)", input_path, df.shape)

        # Automatic column selection
        if text_column is None:
            if df.shape[1] == 1:
                text_column = df.columns[0]
                logger.info("Single-column CSV detected; using that column: %s", text_column)
            elif 'review' in df.columns:
                text_column = 'review'
                logger.info("'review' column found and will be used.")
            else:
                logger.error("Text column not specified and automatic selection failed. Columns: %s", list(df.columns))
                raise ValueError("Text column not specified and 'review' column not found.")

        if text_column not in df.columns:
            logger.error("Specified text column not found in CSV: %s", text_column)
            raise ValueError(f"Specified text column not found in CSV: {text_column}")

        # Prepare NLTK resources
        ensure_nltk_resources(download_if_missing=True)

        # Get the lemmatized series
        df[text_column] = lemmatize_series(df[text_column])

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Lemmatized CSV saved: %s", output_path)

    except Exception:
        logger.exception("An error occurred during process_csv.")
        raise


# -------------------------- CLI entrypoint --------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='POS tag and lemmatize text columns in a CSV.')
    parser.add_argument('--input', '-i', type=str, default=str(Path.cwd() / 'data' / 'X.csv'), help='Input CSV path')
    parser.add_argument('--output', '-o', type=str, default=str(Path.cwd() / 'data' / 'X_lemmatized.csv'), help='Output CSV path')
    parser.add_argument('--column', '-c', type=str, default=None, help='Name of the text column to process (default: automatic)')
    parser.add_argument('--no-download', dest='download', action='store_false', help='Do not download missing NLTK packages')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output if it exists')

    args = parser.parse_args()

    try:
        # Initial info
        logger.info("Started. Input: %s, Output: %s, Column: %s", args.input, args.output, args.column)

        # If the user disabled NLTK downloading, reflect that in the ensure_nltk_resources call
        if not args.download:
            # only check, no download
            try:
                ensure_nltk_resources(download_if_missing=False)
            except RuntimeError as e:
                logger.error("Required NLTK resources are missing and automatic download is disabled: %s", e)
                raise

        process_csv(args.input, args.output, text_column=args.column, overwrite=args.overwrite)
        logger.info("Processing completed successfully.")

    except Exception as exc:
        logger.exception("Error in main program: %s", exc)
        raise
