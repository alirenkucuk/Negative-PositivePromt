# Project name: Negative Review Classifier
# Goal: Classify movie reviews as negative or positive
# File Name: dataset_infos.py
# Writer: Ali Eren Küçük
# File Goal: Get information about the dataset


import logging
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Optional, Union
import os

# -------------------------- Logging setup --------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------- File path setup --------------------------
from __init__ import file_path
if not file_path.ensure_file_exists(file_path.DATA_DIR / 'IMDBDataset.csv'):
    raise FileNotFoundError("IMDBDataset.csv not found")

df = pd.read_csv(file_path.DATA_DIR / 'IMDBDataset.csv')
logger.info("Loaded IMDBDataset.csv with shape %s", df.shape)



def get_dataset_infos():
    logger.info("Getting dataset infos")
    logger.info("Dataset columns: %s", list(df.columns))
    logger.info("Missing values per column: %s", df.isnull().sum().to_dict())
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Dataset head: %s", df.head())
    logger.info("Dataset tail: %s", df.tail())
    logger.info("Dataset info: %s", df.info())
    logger.info("Dataset describe: %s", df.describe())
    logger.info("Dataset value counts: %s", df.value_counts())
    logger.info("Dataset unique values: %s", df.nunique())
    logger.info("Dataset correlation: %s", df.corr())
    logger.info("Dataset covariance: %s", df.cov())
    logger.info("Dataset skewness: %s", df.skew())
    logger.info("Dataset kurtosis: %s", df.kurt())
    logger.info("Dataset min: %s", df.min())
    logger.info("Dataset max: %s", df.max())
    logger.info("Dataset mean: %s", df.mean())
    logger.info("Dataset median: %s", df.median())
    logger.info("Dataset mode: %s", df.mode())
    logger.info("Dataset std: %s", df.std())
    logger.info("Dataset var: %s", df.var())
    logger.info("Dataset sum: %s", df.sum())
    logger.info("Dataset product: %s", df.prod())
    logger.info("Dataset quantile: %s", df.quantile())
    logger.info("Dataset cumsum: %s", df.cumsum())
    logger.info("Dataset cumprod: %s", df.cumprod())
    logger.info("Dataset cummax: %s", df.cummax())
    logger.info("Dataset cummin: %s", df.cummin())
    logger.info("Dataset cumcount: %s", df.cumcount())
    logger.info("Dataset cumsum: %s", df.cumsum())
    logger.info("Dataset cumprod: %s", df.cumprod())
    logger.info("Dataset cummax: %s", df.cummax())
    logger.info("Dataset cummin: %s", df.cummin())
    logger.info("Dataset cumcount: %s", df.cumcount())
    logger.info("Dataset cumsum: %s", df.cumsum())
    logger.info("Dataset cumprod: %s", df.cumprod())
    logger.info("Dataset cummax: %s", df.cummax())
    logger.info("Dataset cummin: %s", df.cummin())
    logger.info("Dataset cumcount: %s", df.cumcount())
    logger.info("Dataset cumsum: %s", df.cumsum())
    logger.info("Dataset cumprod: %s", df.cumprod())
    logger.info("Dataset cummax: %s", df.cummax())
    logger.info("Dataset cummin: %s", df.cummin())
    logger.info("Dataset cumcount: %s", df.cumcount())
    logger.info("Dataset cumsum: %s", df.cumsum())
    logger.info("Dataset cumprod: %s", df.cumprod())
    logger.info("Dataset cummax: %s", df.cummax())
    logger.info("Dataset cummin: %s", df.cummin())




def plot_class_distribution(y, show=True, save_path=None):
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

    # print distribution is balanced or not
    if counts.min() / counts.max() < 0.9:
        logger.info("Distribution is balanced")
    else:
        logger.info("Distribution is not balanced")

