
# -----------------------------------------------------------------
# File: tagging_words.py
# Logic preserved from original file, cleaned formatting and comments.
# -----------------------------------------------------------------

from pathlib import Path
import logging
from typing import List, Optional, Set, Iterable

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords
import spacy
import pandas as pd

# Import file_path module for path management
try:
    import file_path
    X_PATH = file_path.DATA_DIR / "X.csv"
except ImportError:
    X_PATH = Path("data") / "X.csv"

# --- Ensure required NLTK corpora are available ---
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# --- Load spaCy model (must be installed separately) ---
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
except Exception as e:
    raise RuntimeError("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm") from e

lemmatizer = WordNetLemmatizer()

# Build combined STOPWORDS set from NLTK + spaCy + optional extras
_NLTK_STOPWORDS: Set[str] = {w.lower() for w in nltk_stopwords.words("english")}
_SPACY_STOPWORDS: Set[str] = {w.lower() for w in getattr(nlp.Defaults, "stop_words", set())}
_EXPLICIT_EXTRA_STOPWORDS: Set[str] = set()
STOPWORDS: Set[str] = set().union(_NLTK_STOPWORDS, _SPACY_STOPWORDS, _EXPLICIT_EXTRA_STOPWORDS)

# --- Logger ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(h)


def spacy_pos_to_wordnet_pos(spacy_pos: str) -> Optional[str]:
    if spacy_pos in ("NOUN", "PROPN"):
        return "n"
    if spacy_pos == "VERB":
        return "v"
    if spacy_pos == "ADJ":
        return "a"
    if spacy_pos == "ADV":
        return "r"
    return None


def lemmatize_doc_tokens(doc) -> List[str]:
    lemmas: List[str] = []
    for token in doc:
        if not token.is_alpha:
            continue
        if token.is_stop:
            continue
        txt = token.text.lower().strip()
        if not txt or txt in STOPWORDS:
            continue
        wn_pos = spacy_pos_to_wordnet_pos(token.pos_)
        if wn_pos:
            lem = lemmatizer.lemmatize(txt, pos=wn_pos)
        else:
            lem = lemmatizer.lemmatize(txt)
        if lem and lem not in STOPWORDS:
            lemmas.append(lem)
    return lemmas


def get_spacy_lemmatized_words_with_wordnet(text: str) -> List[str]:
    doc = nlp(text)
    return lemmatize_doc_tokens(doc)


def get_spacy_lemmatized_words(text: str) -> List[str]:
    return get_spacy_lemmatized_words_with_wordnet(text)


def get_spacy_pos_tags(text: str) -> List[str]:
    doc = nlp(text)
    return [token.pos_ for token in doc]


def get_spacy_lemmas(text: str) -> List[str]:
    doc = nlp(text)
    return [token.lemma_ for token in doc]


def get_nouns_and_verbs_spacy(text: str) -> dict:
    doc = nlp(text)
    nouns = [(token.text, token.lemma_) for token in doc if token.pos_ in ("NOUN", "PROPN")]
    verbs = [(token.text, token.lemma_) for token in doc if token.pos_ == "VERB"]
    return {"nouns": nouns, "verbs": verbs}


def remove_stopwords(text: str, extra_stopwords: Optional[Iterable[str]] = None) -> str:
    extra = {w.lower() for w in extra_stopwords} if extra_stopwords else set()
    parts = []
    for token in nlp(text):
        t = token.text.strip()
        if not t or not token.is_alpha:
            continue
        lc = t.lower()
        if token.is_stop or lc in STOPWORDS or lc in extra:
            continue
        parts.append(t)
    return " ".join(parts)


def add_stopwords(words: Iterable[str]) -> None:
    global STOPWORDS
    STOPWORDS.update({w.lower() for w in words})


def remove_stopwords_from_set(words: Iterable[str]) -> None:
    global STOPWORDS
    for w in words:
        STOPWORDS.discard(w.lower())


def load_stopwords_from_file(path: Path, sep: Optional[str] = None) -> Set[str]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Stopwords file not found: {path}")
    content = Path(path).read_text(encoding="utf8")
    if sep:
        words = {w.strip().lower() for w in content.split(sep) if w.strip()}
    else:
        words = {w.strip().lower() for w in content.splitlines() if w.strip()}
    add_stopwords(words)
    return words


def reset_stopwords(to_include_spacy_and_nltk: bool = True) -> None:
    global STOPWORDS
    if to_include_spacy_and_nltk:
        STOPWORDS = set().union(_NLTK_STOPWORDS, _SPACY_STOPWORDS, _EXPLICIT_EXTRA_STOPWORDS)
    else:
        STOPWORDS = set()


def combine(text: str) -> List[str]:
    doc = nlp(text)
    return lemmatize_doc_tokens(doc)


def process_texts_batch(texts: List[str], batch_size: int = 64) -> List[List[str]]:
    results = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        results.append(lemmatize_doc_tokens(doc))
    return results


def main(x_path: Optional[Path] = None):
    print("Starting main function...")
    # If no path provided, combine train and test data
    if x_path is None:
        print("No x_path provided, combining train and test data...")
        try:
            import file_path
            train_path = file_path.X_TRAIN_PATH
            test_path = file_path.X_TEST_PATH
            print(f"Train path: {train_path}")
            print(f"Test path: {test_path}")
            if not train_path.exists() or not test_path.exists():
                logger.error("Required files not found: %s or %s", train_path, test_path)
                print(f"Train exists: {train_path.exists()}")
                print(f"Test exists: {test_path.exists()}")
                return
            df = pd.concat([
                pd.read_csv(train_path),
                pd.read_csv(test_path)
            ], ignore_index=True)
            print(f"Combined dataframe shape: {df.shape}")
        except ImportError as e:
            logger.error("file_path module not found: %s", e)
            print(f"Import error: {e}")
            return
    else:
        print(f"Using provided x_path: {x_path}")
        if not x_path.exists():
            logger.error("Input file not found: %s", x_path)
            return
        df = pd.read_csv(x_path)
    text_column = df.columns[0]
    texts = df[text_column].fillna("").astype(str).tolist()
    total = len(texts)
    logger.info("Processing %d texts", total)

    lemmatized_lists = []
    batch_size = 64
    processed = 0
    print(f"Processing {total} texts...")
    print("Progress: [", end="", flush=True)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        for doc in nlp.pipe(batch, batch_size=batch_size):
            lemmatized_lists.append(lemmatize_doc_tokens(doc))
        processed += len(batch)
        progress = processed / total
        bar_len = 50
        filled = int(bar_len * progress)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\rProgress: [{bar}] {progress:.1%} ({processed}/{total})", end="", flush=True)
    print()
    df['lemmatized'] = [' '.join(l) for l in lemmatized_lists]
    
    # Determine output path
    if x_path is None:
        # For combined train+test data, use a default name
        try:
            import file_path
            out_path = file_path.X_TRAIN_AND_TEST_LEMMATIZED_PATH
        except ImportError:
            out_path = Path("data") / "X_train_and_X_test_lemmatized.csv"
    else:
        out_path = x_path.parent / (x_path.stem + "_lemmatized" + x_path.suffix)
    
    df.to_csv(out_path, index=False)
    logger.info("Saved lemmatized file: %s", out_path)


if __name__ == '__main__':
    # Use the file_path module to get the correct paths
    try:
        import file_path
        train_path = file_path.X_TRAIN_PATH
        test_path = file_path.X_TEST_PATH
        if train_path.exists() and test_path.exists():
            main()  # Use default behavior to combine train and test
        else:
            print(f"Required files not found: {train_path} or {test_path}")
    except ImportError:
        print("file_path module not found. Please ensure file_path.py exists.")
