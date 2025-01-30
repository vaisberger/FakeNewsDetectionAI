import os
import logging
from data_ready_xg_rf import prepare_data
from model_training import train_random_forest, train_xgboost, train_gradient_boosting
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import numpy as np
import tensorflow as tf
from xgboost import DMatrix


def parallel_predict(model, X_test, model_name):
    """
    Handle prediction for XGBoost and other models.
    """
    if model_name == "XGBoost":
        # Use the model's built-in predict for XGBoost
        return model.predict(DMatrix(X_test))
    else:
        # Parallel prediction for other models
        predictions = Parallel(n_jobs=-1)(delayed(model.predict)(X_test[i:i + 500])
                                          for i in range(0, X_test.shape[0], 500))

        # Flatten the result to ensure it is a 1D array
        return np.concatenate(predictions)


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate the given model and print metrics.
    Handles XGBoost's specific requirements.
    """
    # Parallel prediction
    y_pred = parallel_predict(model, X_test, model_name)

    # Flatten the predictions to make sure it's a 1D array
    y_pred = np.ravel(y_pred)

    if len(y_pred) == 0:
        logging.error(f"{model_name} prediction returned an invalid result.")
        return

    if model_name == "XGBoost":
        # XGBoost returns probabilities; round to 0 or 1 for classification
        y_pred = np.where(y_pred > 0.5, 1, 0)

    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    logging.info(f"{model_name} Accuracy: {accuracy:.2f}")
    logging.info(f"{model_name} Classification Report:\n{class_report}")


def main():
    """
    Main pipeline for data preparation, model training, and evaluation.
    """
    # Logging setup
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # File paths
    zip_path = r"C:\Users\wisbr\FakeNewsDetectionAI\data\Cleaned_Dataset.csv"

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file not found at {zip_path}")

    # Data preparation
    logging.info("Preparing data...")
    X_train, X_test, y_train, y_test, manual_check_df = prepare_data(zip_path)  # ✅ Capture manual_check_df
    logging.info("Data preparation completed.")

    # Save manual check data
    manual_check_path = r"C:\Users\wisbr\FakeNewsDetectionAI\data\Manual_Check_Data.csv"
    manual_check_df[['text', 'label']].to_csv(manual_check_path, index=False)
    logging.info(f"Manual check data saved to {manual_check_path}")

    # Train Random Forest
    logging.info("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    logging.info("Random Forest training completed.")

    # Train XGBoost
    logging.info("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)
    logging.info("XGBoost training completed.")

    # Train Gradient Boosting
    logging.info("Training Gradient Boosting...")
    gb_model, pred_gb = train_gradient_boosting(X_train, y_train, X_test, y_test)
    logging.info("Gradient Boosting training completed.")

    # Evaluate models
    logging.info("Evaluating Random Forest...")
    evaluate_model(rf_model, X_test, y_test, "Random Forest")

    logging.info("Evaluating XGBoost...")
    evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    logging.info("Evaluating Gradient Boosting...")
    evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")

    # Save models
    logging.info("Saving models...")
    joblib.dump(rf_model, "random_forest_model2.pkl")
    joblib.dump(xgb_model, "xgboost_model2.pkl")
    joblib.dump(gb_model, "gradient_boosting_model.pkl")
    logging.info("Models saved successfully.")

if __name__ == "__main__":
    main()







