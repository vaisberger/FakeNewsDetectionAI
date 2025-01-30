import pandas as pd
import zipfile
import io
import chardet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

def prepare_data(zip_path, test_size=0.25, random_state=42, max_features=500, ngram_range=(1, 2), min_df=0.01):
    """
    Process the dataset and return TF-IDF vectors for training and testing.

    Parameters:
        zip_path (str): Path to the ZIP file containing the cleaned dataset.
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
    # Extract and read the cleaned dataset
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_name = z.namelist()[0]
        print(f"Reading file: {file_name}")

        # Detect encoding using a sample
        with z.open(file_name) as f:
            sample = f.read(10000)
            encoding = chardet.detect(sample)['encoding']
            print(f"Detected file encoding: {encoding}")

        # Read the entire file into memory and decode it
        with z.open(file_name) as f:
            decoded_file = f.read().decode(encoding, errors='replace')

        # Convert the decoded file string to a DataFrame
        cleaned_df = pd.read_csv(io.StringIO(decoded_file))

    # Ensure rows with missing or empty 'cleaned_text' or 'cleaned_title' are removed
    cleaned_df = cleaned_df.dropna(subset=['cleaned_text', 'cleaned_title', 'lable'])
    cleaned_df = cleaned_df[
        (cleaned_df['cleaned_text'].str.strip() != '') &
        (cleaned_df['cleaned_title'].str.strip() != '')
    ]

    # Extract features and labels
    x = cleaned_df['cleaned_text']
    y = cleaned_df['lable']

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

if __name__ == "__main__":
    # Define the path to the ZIP file
    zip_path = r'C:\Users\User\Desktop\app\Cleaned_Dataset_news.zip'

    # Call the function to prepare data
    X_train, X_test, y_train, y_test = prepare_data(zip_path)

    # Print shapes to confirm everything works
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    LR = LogisticRegression()
    LR.fit(X_train, y_train)
    pred_lr = LR.predict(X_test)
    print(classification_report(y_test, pred_lr))
    # Use StandardScaler with with_mean=False for sparse matrices
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Convert scaled data back to DataFrame
    X_train_scaled_df = pd.DataFrame(X_train_scaled)
    X_test_scaled_df = pd.DataFrame(X_test_scaled)

    # Save processed data
    X_train_scaled_df.to_csv('Processed_Train_Dataset_SVM.csv', index=False)
    X_test_scaled_df.to_csv('Processed_Test_Dataset_SVM.csv', index=False)

    print("Data has been scaled and saved as 'Processed_Train_Dataset_SVM.csv' and 'Processed_Test_Dataset_SVM.csv'.")

    # Create and train SVM model
    model = LinearSVC(C=0.5, max_iter=8000)
    model.fit(X_train_scaled, y_train)

    # Predict on test data
    y_pred = model.predict(X_test_scaled)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')

    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)

    # Save trained model
    joblib.dump(model, 'svm_model.pkl')
    print("Model has been trained and saved as 'svm_model.pkl'.")
