# this file is training the second set of data the same way as the first
import logging
from model_training import train_random_forest, train_xgboost, train_svm
from sklearn.metrics import accuracy_score, classification_report
import joblib
from xgboost import DMatrix
import re
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
import os

nltk.download('stopwords')
nltk.download('punkt')


stop_words = list(stopwords.words('english'))
stemmer = SnowballStemmer('english')


# Cleans the input text
def clean_text(text):

    if pd.isnull(text):  # Handle null values
        return ""

    text = text.lower()   # Convert to lowercase

    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs

    text = re.sub(r'\[.*?\]', '', text)   # Remove content inside square brackets

    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags

    text = re.sub(r'\W', ' ', text)  # Replace all non-word characters with a space

    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)  # Remove punctuation

    text = re.sub(r'\w*\d\w*', '', text) # Remove words with digits

    # Collapse multiple spaces into one and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = nltk.word_tokenize(text)

    # Stem each token using the SnowballStemmer
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(stemmed_tokens)

#    Prepares the data for model training by cleaning, handling missing values
#    , and splitting into train/test sets.
def prepare_data(file_path):

    # Load the cleaned dataset
    df = pd.read_csv(file_path, encoding='ISO-8859-1')

    # Invert label values (0 <-> 1) because the dataset was labeled the opposite
    df['label'] = df['label'].map({0: 1, 1: 0})

    # Split the data manual check and remaining
    manual_check_df = df.sample(frac=0.005, random_state=42)
    remaining_df = df.drop(manual_check_df.index)

    x = remaining_df['text']
    y = remaining_df['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, df


# Train models on new data and save them.
def train_and_evaluate_new_data(new_data_path, save_dir):

    data = pd.read_csv(new_data_path, encoding="ISO-8859-1")
    data['text'] = data['text'].astype(str).apply(clean_text)
    logging.info("Data cleaning completed.")

    cleaned_data_path = os.path.join(save_dir, "Cleaned_New_Data.csv")
    data.to_csv(cleaned_data_path, index=False, encoding="ISO-8859-1")
    logging.info(f"Cleaned data saved to {cleaned_data_path}")


    X_train, X_test, y_train, y_test, manual_check_df = prepare_data("C:/Users/wisbr/FakeNewsDetectionAI/new_data_training/Cleaned_New_Data.csv")
    logging.info("New data preparation completed.")

    # Save manual check data
    manual_check_path = os.path.join(save_dir, "Manual_Check_Data2.csv")
    manual_check_df[['text', 'label']].to_csv(manual_check_path, index=False)
    logging.info(f"Manual check data saved to {manual_check_path}")

    # Vectorize the data
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    xv_train = vectorizer.fit_transform(X_train)  # Vectorize training data
    xv_test = vectorizer.transform(X_test)  # Vectorize testing data

    # Save the vectorizer
    joblib.dump(vectorizer, os.path.join(save_dir, "vectorizer2.pkl"))
    logging.info("Vectorizer saved.")

    print(f"Training data shape: {xv_train.shape}")
    print(f"Testing data shape: {xv_test.shape}")
    print(f"Manual Check Data: {len(manual_check_df)} samples")

    # Train models
    logging.info("Training Random Forest...")
    rf_model = train_random_forest(xv_train, y_train)
    logging.info("Random Forest training completed.")

    logging.info("Training XGBoost...")
    xgb_model = train_xgboost(xv_train, y_train, xv_test, y_test)
    logging.info("XGBoost training completed.")

    logging.info("Training SVM...")
    svm_model, X_test_scaled = train_svm(xv_train, y_train, xv_test, y_test)
    logging.info("SVM training completed.")

    # Evaluate models
    def evaluate_model(model, X_test, y_test, model_name):
        """
        Evaluate a given model on test data.
        """
        if model_name == "XGBoost":
            y_pred = model.predict(DMatrix(X_test))
            y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary
        else:
            y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"{model_name} Accuracy: {accuracy:.2f}")
        logging.info(f"{model_name} Classification Report:\n{classification_report(y_test, y_pred)}")

    logging.info("Evaluating models...")
    evaluate_model(rf_model, xv_test, y_test, "Random Forest")
    evaluate_model(xgb_model, xv_test, y_test, "XGBoost")
    evaluate_model(svm_model, X_test_scaled, y_test, "SVM")

    # Save models
    logging.info("Saving models...")
    joblib.dump(rf_model, os.path.join(save_dir, "random_forest_model2.pkl"))
    joblib.dump(xgb_model, os.path.join(save_dir, "xgboost_model2.pkl"))
    joblib.dump(svm_model, os.path.join(save_dir,"svm_model2.pkl"))
    logging.info("Models saved successfully.")


if __name__ == "__main__":
    print("Starting the training process...")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Starting the training process...")

    new_data_csv_path = r'/data/new_dataset.csv'
    model_save_directory = r'C:\Users\wisbr\FakeNewsDetectionAI\new_data_training'

    try:
        train_and_evaluate_new_data(new_data_csv_path, model_save_directory)
    except Exception as e:
        logging.error(f"Error occurred: {e}")