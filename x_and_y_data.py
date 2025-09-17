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


# -------------------------- Logging setup --------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("All libraries imported successfully.")
logger.info("Numpy version: %s", np.__version__)
logger.info("Pandas version: %s", pd.__version__)
logger.info("Matplotlib version: %s", mpl.__version__)
logger.info("OS name: %s", os.name)


def ensure_dir_exists(path):
    """Ensure the parent directory for `path` exists.

    Parameters
    ----------
    path : str
        File path whose parent directory should exist (e.g. '/some/dir/file.csv').

    Behavior
    --------
    - If the parent directory doesn't exist, it is created (recursive).
    - Logs whether creation was needed or the directory already existed.

    Notes
    -----
    This function is idempotent and safe to call before any file-writing operation.
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


def load_csv_or_fail(path, expected_columns = None, nrows_preview = 5):
    """Load a CSV file into a pandas DataFrame with helpful validation and logging.

    Parameters
    ----------
    path : str
        Path to the CSV file to load.
    expected_columns : Optional[List[str]], optional
        If provided, ensure these columns are present in the loaded DataFrame. Raises
        ValueError if any are missing.
    nrows_preview : int, optional
        Number of rows to print as a preview in the logs.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file doesn't exist.
    ValueError
        If expected_columns is provided but some columns are missing.
    """
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

    # Log a small preview for quick debugging
    logger.info("Data preview (first %d rows): %s", nrows_preview, df.head(nrows_preview).to_string(index=False))

    return df


def save_series(series, path, column_name = None):
    """Save a pandas Series to CSV as a single-column file.

    Parameters
    ----------
    series : pd.Series
        The series to save. If unnamed, provide `column_name`.
    path : str
        Destination CSV path.
    column_name : Optional[str]
        Column name to use in the output CSV. If None, will use `series.name` or 'value'.

    Behavior
    --------
    - Ensures parent directory exists.
    - Resets the index and writes CSV without the index column.
    - Logs the number of saved rows and the filepath.
    """
    ensure_dir_exists(path)

    col = column_name or series.name or 'value'
    df = pd.DataFrame({col: series.reset_index(drop=True)})
    df.to_csv(path, index=False)

    logger.info("Saved Series to %s (column name: '%s', rows: %d)", path, col, len(df))


def plot_class_distribution(y, show = True, save_path = None):
    """Plot and optionally save the class distribution of a binary label series.

    Parameters
    ----------
    y : pd.Series
        Binary labels (e.g. 0/1). Non-binary values are aggregated by value.
    show : bool
        Whether to call plt.show() after plotting.
    save_path : Optional[str]
        If provided, the plot will be saved to this filepath (parent folder created if needed).
    """
    counts = y.value_counts().sort_index()
    logger.info("Class counts: %s", counts.to_dict())

    ax = counts.plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.tight_layout()

    if save_path:
        ensure_dir_exists(save_path)
        plt.savefig(save_path)
        logger.info("Saved class distribution plot to %s", save_path)

    if show:
        plt.show()
    else:
        plt.close()


def create_train_test_split(X, y, test_size = 0.2, random_state = 1234, stratify = True):
    """Create a train/test split for feature and label series with checks and logging.

    Parameters
    ----------
    X : pd.Series
        Feature series (text reviews).
    y : pd.Series
        Label series (binary labels compatible with sklearn).
    test_size : float
        Fraction of the dataset to reserve for testing (0 < test_size < 1).
    random_state : int
        RNG seed passed to sklearn's train_test_split.
    stratify : bool
        Whether to stratify the split using `y` to preserve class balance.

    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
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
    """Main entry point: load IMDB CSV, validate, save X/y and train/test splits, and plot distribution.

    The function uses helper functions above and logs important steps/errors. It is
    intentionally defensive: any exception is logged with a stack trace and re-raised
    to aid debugging during development.
    """
    try:
        current_directory = os.getcwd()
        logger.info("Current working directory: %s", current_directory)

        data_path = os.path.join(current_directory, 'data', 'IMDBDataset.csv')

        # Load dataset with expected columns validation
        data = load_csv_or_fail(data_path, expected_columns=['review', 'sentiment'])

        # Basic checks and information
        logger.info("Dataset columns: %s", list(data.columns))
        logger.info("Missing values per column: %s", data.isnull().sum().to_dict())

        # Features and labels
        X = data['review']
        y = data['sentiment'].map({'positive': 1, 'negative': 0})
        logger.info("Mapped sentiment to binary labels. Unique labels after mapping: %s", sorted(y.unique()))

        # Save raw X and y
        save_series(X, os.path.join(current_directory, 'data', 'X.csv'), column_name='review')
        save_series(y, os.path.join(current_directory, 'data', 'y.csv'), column_name='sentiment')

        # Plot class distribution and save a copy
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
