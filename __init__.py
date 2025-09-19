# Project name: Negative Review Classifier
# Goal: Classify movie reviews as negative or positive
# File Name: __init__.py
# Writer: Ali Eren Küçük
# File Goal: Initialize the project

from file_path import file_path
from x_and_y_data import load_csv_or_fail, save_series, plot_class_distribution, create_train_test_split
from tagging_words import process_csv
from model import train_model, cross_validate_model, save_model

file_path.ensure_file_exists(file_path.DATA_DIR / 'IMDBDataset.csv')
file_path.ensure_file_exists(file_path.DATA_DIR / 'X.csv')
file_path.ensure_file_exists(file_path.DATA_DIR / 'y.csv')  