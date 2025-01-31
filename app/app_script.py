import joblib
import pandas as pd
import xgboost as xgb
import re
import string
import nltk
from nltk.stem import SnowballStemmer
from scipy.special import expit

# Download necessary NLTK resources
nltk.download('punkt')

# Initialize stemmer
stemmer = SnowballStemmer("english")


# Function to clean the text
def clean_text(text):
    if pd.isnull(text):
        return ""

    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = nltk.word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(stemmed_tokens)


# Load model and vectorizer
def load_model(model_path):
    return joblib.load(model_path)


def load_vectorizer(vectorizer_path):
    return joblib.load(vectorizer_path)


# Paths to saved models and vectorizers
vectorizer1_path = "../code/vectorizer1.pkl"
xgb_model1_path = "../code/xgboost_model1.pkl"
rf_model1_path = "../code/random_forest_model1.pkl"
svm_model1_path = "../code/svm_model1.pkl"

vectorizer2_path = "../trained_models/vectorizer2.pkl"
xgb_model2_path = "../trained_models/xgboost_model2.pkl"
rf_model2_path = "../trained_models/random_forest_model2.pkl"
svm_model2_path = "../trained_models/svm_model2.pkl"

# Load vectorizers and models
vectorizer1 = load_vectorizer(vectorizer1_path)
xgb_model1 = load_model(xgb_model1_path)
rf_model1 = load_model(rf_model1_path)
svm_model1 = load_model(svm_model1_path)

vectorizer2 = load_vectorizer(vectorizer2_path)
xgb_model2 = load_model(xgb_model2_path)
rf_model2 = load_model(rf_model2_path)
svm_model2 = load_model(svm_model2_path)


# Prediction function
def predict_with_model(model, vectorizer, text):
    cleaned_text = clean_text(text)
    X_text = vectorizer.transform([cleaned_text])

    if isinstance(model, xgb.Booster):
        X_text_dmatrix = xgb.DMatrix(X_text)
        raw_prediction = model.predict(X_text_dmatrix)
        probability = expit(raw_prediction)[0]
    elif hasattr(model, "predict_proba"):
        probability = model.predict_proba(X_text)[:, 1][0]
    else:
        raw_prediction = model.decision_function(X_text)
        probability = expit(raw_prediction[0])

    return probability


# Function to return averaged model prediction
def check_news(text):
    model_predictions = {
        "XGBoost": (predict_with_model(xgb_model1, vectorizer1, text),
                    predict_with_model(xgb_model2, vectorizer2, text)),
        "Random Forest": (predict_with_model(rf_model1, vectorizer1, text),
                          predict_with_model(rf_model2, vectorizer2, text)),
        "SVM": (predict_with_model(svm_model1, vectorizer1, text),
                predict_with_model(svm_model2, vectorizer2, text))
    }

    final_results = {}

    for algo, (prob1, prob2) in model_predictions.items():
        final_probability = (prob1 + prob2) / 2
        final_prediction = "TRUE" if final_probability >= 0.5 else "FAKE"
        final_results[algo] = (final_prediction, round(final_probability, 2))

    # Majority voting: If 2 or more models predict TRUE, return TRUE; otherwise, FAKE
    true_count = sum(1 for result in final_results.values() if result[0] == "TRUE")
    majority_result = "TRUE" if true_count >= 2 else "FAKE"

    return majority_result





