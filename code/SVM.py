import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

def prepare_data(file_path, test_size=0.3, random_state=42, max_features=10000, ngram_range=(1, 3), min_df=0.01):
    """
    Process the dataset and return TF-IDF vectors for training and testing.

    Parameters:
        file_path (str): Path to the CSV file containing the cleaned dataset.
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
        vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
    """

    # Load dataset
    df = pd.read_csv(file_path)

    # Ensure the correct column names exist
    expected_columns = ['text', 'label']
    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")

    # Drop missing values and empty texts
    df = df.dropna(subset=['text', 'label'])
    df = df[df['text'].str.strip() != '']

    x = df['text']
    y = df['label']

    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words='english'
    )

    xv_train = vectorizer.fit_transform(x_train)
    xv_test = vectorizer.transform(x_test)

    print(f"Training data shape: {xv_train.shape}")
    print(f"Testing data shape: {xv_test.shape}")

    return xv_train, xv_test, y_train, y_test, vectorizer

if __name__ == "__main__":
    file_path = r'C:\Users\User\Desktop\app\Cleaned_Dataset.csv'

    # Step 1: Prepare the data
    X_train, X_test, y_train, y_test, vectorizer = prepare_data(file_path)

    # Save the vectorizer for later use
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

    # Step 2: Balance the data using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Resampled training data shape: {X_train_resampled.shape}")

    # Step 3: Hyperparameter tuning with GridSearchCV for LinearSVC
    param_grid = {
        'C': [0.1, 1, 10],
        'max_iter': [5000, 10000, 20000]
    }

    grid = GridSearchCV(LinearSVC(), param_grid, cv=5)
    grid.fit(X_train_resampled, y_train_resampled)
    print(f"Best parameters: {grid.best_params_}")
    best_svc_model = grid.best_estimator_

    # Evaluate the optimized LinearSVC model
    y_pred_svc = best_svc_model.predict(X_test)
    accuracy_svc = accuracy_score(y_test, y_pred_svc)
    print(f'LinearSVC Optimized Accuracy: {accuracy_svc:.4f}')
    print("LinearSVC Classification Report:\n", classification_report(y_test, y_pred_svc))

    # Save the optimized LinearSVC model
    joblib.dump(best_svc_model, 'linear_svc_optimized.pkl')

    # Step 4: Train XGBoost model
    xgb_model = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train_resampled.toarray(), y_train_resampled)  # Convert sparse matrix to dense array

    y_pred_xgb = xgb_model.predict(X_test.toarray())
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    print(f'XGBoost Accuracy: {accuracy_xgb:.4f}')
    print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))

    # Save the XGBoost model
    joblib.dump(xgb_model, 'xgboost_model.pkl')
    print("Optimized models have been trained and saved.")
