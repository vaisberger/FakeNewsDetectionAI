#Model Training and Evaluation Pipeline for the first dataset we trained
import os
import logging
import joblib
import numpy as np
from joblib import Parallel, delayed
from xgboost import DMatrix
from sklearn.metrics import accuracy_score, classification_report
from Tfidf_prepare import prepare_data
from model_training import train_random_forest, train_xgboost, train_svm


#Handle prediction for XGBoost and other models.
def parallel_predict(model, X_test, model_name):
    if model_name == "XGBoost":
        return model.predict(DMatrix(X_test))  # XGBoost uses DMatrix for prediction
    else:
        predictions = Parallel(n_jobs=-1)(delayed(model.predict)(X_test[i:i + 500])
                                          for i in range(0, X_test.shape[0], 500))
        return np.concatenate(predictions)  # Flatten results


#Evaluate the given model and log metrics.
def evaluate_model(model, X_test, y_test, model_name):

    y_pred = parallel_predict(model, X_test, model_name)
    y_pred = np.ravel(y_pred)

    if len(y_pred) == 0:
        logging.error(f"{model_name} prediction returned an invalid result.")
        return

    if model_name == "XGBoost":
        y_pred = np.where(y_pred > 0.5, 1, 0)  # Convert probabilities to binary labels

    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"{model_name} Accuracy: {accuracy:.2f}")
    logging.info(f"{model_name} Classification Report:\n{classification_report(y_test, y_pred)}")


#Main pipeline for data preparation, model training, and evaluation.
def main():

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    cvs_path = r"C:\Users\wisbr\FakeNewsDetectionAI\data\Cleaned_Dataset.csv"

    if not os.path.exists(cvs_path):
        raise FileNotFoundError(f"File not found at {cvs_path}")

    # Data preparation
    logging.info("Preparing data...")
    X_train, X_test, y_train, y_test, manual_check_df = prepare_data(cvs_path)
    logging.info("Data preparation completed.")

    # Save manual check data for later
    manual_check_path = r"C:\Users\wisbr\FakeNewsDetectionAI\data\Manual_Check_Data.csv"
    manual_check_df[['text', 'label']].to_csv(manual_check_path, index=False)
    logging.info(f"Manual check data saved to {manual_check_path}")

    # Train models
    logging.info("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)

    logging.info("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)

    logging.info("Training SVM...")
    svm_model, X_test_scaled = train_svm(X_train, y_train, X_test, y_test)
    logging.info("SVM training completed.")

    # Evaluate models
    logging.info("Evaluating Random Forest...")
    evaluate_model(rf_model, X_test, y_test, "Random Forest")

    logging.info("Evaluating XGBoost...")
    evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    logging.info("Evaluating SVM...")
    evaluate_model(svm_model, X_test_scaled, y_test, "SVM")

    # Save models
    logging.info("Saving models...")
    joblib.dump(rf_model, "random_forest_model1.pkl")
    joblib.dump(xgb_model, "xgboost_model1.pkl")
    joblib.dump(svm_model, "svm_model1.pkl")
    logging.info("All models saved successfully.")

if __name__ == "__main__":
    main()








