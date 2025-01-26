from data_ready_xg_rf import prepare_data
from model_training import train_random_forest, train_xgboost
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main():
    """
    Main function to orchestrate the Fake News Detection AI pipeline.
    It includes data preparation, model training, evaluation, and saving models.
    """
    # Define the path to the ZIP file
    zip_path = r'C:\Users\wisbr\FakeNewsDetectionAI\data\Cleaned_Dataset_news.zip'

    # Step 1: Prepare the data
    print("Preparing data...")
    x_train, x_test, y_train, y_test = prepare_data(zip_path)
    print("Data preparation completed.\n")

    # Step 2: Train the Random Forest model
    print("Training Random Forest model...")
    rf_model = train_random_forest(x_train, y_train)
    print("Random Forest model training completed.\n")

    # Step 3: Train the XGBoost model
    print("Training XGBoost model...")
    xgb_model = train_xgboost(x_train, y_train)
    print("XGBoost model training completed.\n")

    # Step 4: Evaluate models
    print("Evaluating models...")

    # Random Forest Evaluation
    rf_pred = rf_model.predict(x_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_class_report = classification_report(y_test, rf_pred)
    print(f"Random Forest Accuracy: {rf_accuracy}")
    print(f"Random Forest Classification Report:\n{rf_class_report}")

    # XGBoost Evaluation
    xgb_pred = xgb_model.predict(x_test)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    xgb_class_report = classification_report(y_test, xgb_pred)
    print(f"XGBoost Accuracy: {xgb_accuracy}")
    print(f"XGBoost Classification Report:\n{xgb_class_report}")

    print("Evaluation completed.\n")

    # Step 5: Save the models
    print("Saving models...")
    joblib.dump(rf_model, "random_forest_model.pkl")
    joblib.dump(xgb_model, "xgboost_model.pkl")
    print("Models saved successfully.\n")

if __name__ == "__main__":
    main()


