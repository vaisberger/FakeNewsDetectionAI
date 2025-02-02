import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.exceptions import NotFittedError
import xgboost as xgb

from app.app_script import svm_model2_path, load_vectorizer_from_zip


def load_model(model_path):
    """Load a trained model from the specified path."""
    return joblib.load(model_path)


def load_vectorizer(vectorizer_path):
    """Load the saved TfidfVectorizer from the specified path."""
    return joblib.load(vectorizer_path)


def check_vectorizer_fitted(vectorizer):
    """Check if the vectorizer has been fitted."""
    try:
        vectorizer.transform(["test text"])  # Test if the vectorizer is fitted
    except NotFittedError:
        print("Vectorizer is not fitted!")
        return False
    return True


def predict_with_model(model, vectorizer, manual_check_data, is_xgb=False, is_svm=False):
    """
    Transform the manual check data and make predictions using the given model.

    Parameters:
        model: Trained model (Random Forest, XGBoost, or SVM).
        vectorizer: The fitted TF-IDF vectorizer.
        manual_check_data: Data for manual review.
        is_xgb: Boolean flag indicating if the model is XGBoost (default False).
        is_svm: Boolean flag indicating if the model is SVM (default False).

    Returns:
        predictions: Model predictions.
    """
    # Check if the vectorizer has been fitted
    if not check_vectorizer_fitted(vectorizer):
        return None

    # Transform the manual check data using the vectorizer (use the same vectorizer fitted on the training data)
    X_manual = vectorizer.transform(manual_check_data['text'])

    if is_xgb:
        # If using XGBoost model, convert the data to DMatrix format
        X_manual = xgb.DMatrix(X_manual)
        # Get the probabilities (not the discrete labels)
        probs = model.predict(X_manual)
        # Convert probabilities to binary predictions (threshold of 0.5)
        predictions = [1 if prob >= 0.5 else 0 for prob in probs]

    elif is_svm:
        # For SVM, just call predict (no need for DMatrix)
        predictions = model.predict(X_manual)

    else:
        # For Random Forest, the prediction is already binary
        predictions = model.predict(X_manual)

    return predictions


def evaluate_predictions(predictions, manual_check_data):
    """Evaluate the model predictions with respect to the manual check data's labels."""
    accuracy = accuracy_score(manual_check_data['label'], predictions)

    # הוספת zero_division=1 תמנע את האזהרה
    class_report = classification_report(manual_check_data['label'], predictions, zero_division=1)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Classification Report:\n{class_report}")


def main():
    """Main function to load models, vectorizer, and perform manual testing."""
    # File paths
    vectorizer2_zip = "vectorizer1.zip"
    xgb_model2_path = "xgboost_model1.pkl"
    rf_model2_path = "random_forest_model1.pkl"
    svm_model2_path = "svm_model1.pkl"

    # Load models
    rf_model = load_model(rf_model2_path)
    xgb_model = load_model(xgb_model2_path)
    svm_model = load_model(svm_model2_path)

    # Load vectorizer
    try:
        vectorizer1 = load_vectorizer_from_zip(vectorizer2_zip, "vectorizer1.pkl")
    except Exception as e:
        print(f"Error loading vectorizer: {e}")
        return

    # Load manual check data
    manual_check_data = pd.read_csv(r'../data/Manual_Check_Data.csv')

    print("Original labels distribution:")
    print(manual_check_data['label'].value_counts())

    # Make predictions with Random Forest
    print("\nMaking predictions with Random Forest model...")
    rf_predictions = predict_with_model(rf_model, vectorizer1, manual_check_data)
    if rf_predictions is not None:
        print("\nEvaluating Random Forest model...")
        evaluate_predictions(rf_predictions, manual_check_data)

    # Make predictions with XGBoost
    print("\nMaking predictions with XGBoost model...")
    xgb_predictions = predict_with_model(xgb_model, vectorizer1, manual_check_data, is_xgb=True)
    if xgb_predictions is not None:
        print("\nEvaluating XGBoost model...")
        evaluate_predictions(xgb_predictions, manual_check_data)

    # Make predictions with SVM
    print("\nMaking predictions with SVM model...")
    svm_predictions = predict_with_model(svm_model, vectorizer1, manual_check_data, is_svm=True)
    print("\nPredicted labels distribution for SVM:")
    if svm_predictions is not None:
        print("\nEvaluating SVM model...")
        evaluate_predictions(svm_predictions, manual_check_data)


if __name__ == "__main__":
    main()












