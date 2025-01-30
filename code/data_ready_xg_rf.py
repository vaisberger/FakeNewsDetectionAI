import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import zipfile
import chardet
import io


def prepare_data(file_path, test_size=0.25, random_state=42, max_features=100000, ngram_range=(1, 3), min_df=0.01):
    """
    Process the dataset and return TF-IDF vectors for training and testing.

    Parameters:
        file_path (str): Path to the ZIP file containing the cleaned dataset.
        test_size (float): Proportion of the data to be used for testing.
        random_state (int): Random state for reproducibility.
        max_features (int): Maximum number of features for TF-IDF.
        ngram_range (tuple): N-gram range for TF-IDF.
        min_df (float): Minimum document frequency for TF-IDF.

    Returns:
        xv_train (sparse matrix): TF-IDF vectorized training data.
        xv_test (sparse matrix): TF-IDF vectorized testing data.
        y_train (Series): Training labels.
        y_test (Series): Testing labels.
    """

    cleaned_df = pd.read_csv(file_path)


    # Ensure rows with missing or empty 'cleaned_text' or 'cleaned_title' are removed
    cleaned_df = cleaned_df.dropna(subset=['text', 'title', 'label'])  # Drop rows with NaNs
    cleaned_df = cleaned_df[
        (cleaned_df['text'].str.strip() != '') &
        (cleaned_df['title'].str.strip() != '')
    ]

    # Extract features and labels
    x = cleaned_df['text']
    y = cleaned_df['label']


    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df)

    # Fit-transform on training data, transform on testing data
    xv_train = vectorizer.fit_transform(x_train)
    xv_test = vectorizer.transform(x_test)

    print(f"Training data shape: {xv_train.shape}")
    print(f"Testing data shape: {xv_test.shape}")

    return xv_train, xv_test, y_train, y_test


# Example usage
if __name__ == "__main__":
    # Define the path to the ZIP file
    zip_path = r'C:\Users\wisbr\FakeNewsDetectionAI\data\Processed_Dataset_news.zip'

    # Call the function to prepare data
    X_train, X_test, y_train, y_test = prepare_data(zip_path)

    # Print shapes to confirm everything works
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)









