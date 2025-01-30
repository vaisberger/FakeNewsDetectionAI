import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.exceptions import NotFittedError
import xgboost as xgb


def load_model(model_path):
    """
    Load a trained model from the specified path.
    """
    return joblib.load(model_path)


def load_vectorizer(vectorizer_path):
    """
    Load the saved TfidfVectorizer from the specified path.
    """
    return joblib.load(vectorizer_path)


def check_vectorizer_fitted(vectorizer):
    """
    Check if the vectorizer has been fitted.
    """
    try:
        vectorizer.transform(["test text"])  # Test if the vectorizer is fitted
    except NotFittedError:
        print("Vectorizer is not fitted!")
        return False
    return True


def predict_with_model(model, vectorizer, manual_check_data, is_xgb=False):
    """
    Transform the manual check data and make predictions using the given model.

    Parameters:
        model: Trained model (Random Forest or XGBoost).
        vectorizer: The fitted TF-IDF vectorizer.
        manual_check_data: Data for manual review.
        is_xgb: Boolean flag indicating if the model is XGBoost (default False for Random Forest).

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
    else:
        # For Random Forest, the prediction is already binary
        predictions = model.predict(X_manual)

    return predictions


def evaluate_predictions(predictions, manual_check_data):
    """
    Evaluate the model predictions with respect to the manual check data's labels.
    """
    accuracy = accuracy_score(manual_check_data['label'], predictions)
    class_report = classification_report(manual_check_data['label'], predictions)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Classification Report:\n{class_report}")


def main():
    """
    Main function to load models, vectorizer, and perform manual testing.
    """
    # File paths
    vectorizer_path = "vectorizer.pkl"  # Path to the saved vectorizer
    rf_model_path = "random_forest_model2.pkl"  # Path to the saved model
    xgb_model_path = "xgboost_model2.pkl"  # Path to the saved XGBoost model

    # Load models
    rf_model = load_model(rf_model_path)
    xgb_model = load_model(xgb_model_path)

    # Load vectorizer
    try:
        vectorizer = load_vectorizer(vectorizer_path)
    except Exception as e:
        print(f"Error loading vectorizer: {e}")
        return

    # Load manual check data
    manual_check_data = pd.read_csv(r'C:\Users\wisbr\FakeNewsDetectionAI\data\Manual_Check_Data.csv')

    # Make predictions with Random Forest
    print("\nMaking predictions with Random Forest model...")
    rf_predictions = predict_with_model(rf_model, vectorizer, manual_check_data)
    if rf_predictions is not None:
        print("\nEvaluating Random Forest model...")
        evaluate_predictions(rf_predictions, manual_check_data)

    # Make predictions with XGBoost
    print("\nMaking predictions with XGBoost model...")
    xgb_predictions = predict_with_model(xgb_model, vectorizer, manual_check_data, is_xgb=True)
    if xgb_predictions is not None:
        print("\nEvaluating XGBoost model...")
        evaluate_predictions(xgb_predictions, manual_check_data)


if __name__ == "__main__":
    main()











