import joblib
import pandas as pd
import xgboost as xgb
import re
import string
import nltk
from nltk.stem import SnowballStemmer
from scipy.special import expit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

app = Flask(__name__)
# Download necessary NLTK resources
nltk.download('punkt')

# Initialize stemmer
stemmer = SnowballStemmer("english")


# Function to clean the text
def clean_text(text):
    if pd.isnull(text):  # Handle NaN values
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


def predict_with_model(model, vectorizer, text):
    """
    Cleans and vectorizes the input text, then predicts using the model.
    """
    cleaned_text = clean_text(text)
    X_text = vectorizer.transform([cleaned_text])

    if isinstance(model, xgb.Booster):
        X_text_dmatrix = xgb.DMatrix(X_text)  # Convert for XGBoost
        raw_prediction = model.predict(X_text_dmatrix)  # Get raw score
        probability = expit(raw_prediction)[0]  # Apply sigmoid
    else:
        raw_prediction = model.predict_proba(X_text)[:, 1]  # Get probability for class 1
        probability = raw_prediction[0]

    prediction = 1 if probability >= 0.5 else 0
    return prediction, probability


def evaluate_on_new_dataset(model, vectorizer, dataset_path):
    """
    Evaluates the model on an unseen dataset.
    """
    try:
        # Attempt to read the dataset with different encoding options
        df_test = pd.read_csv(dataset_path, encoding="latin1")  # Change encoding to "latin1"

        # Ensure the dataset has the required columns
        if 'text' not in df_test.columns or 'label' not in df_test.columns:
            print("Error: Dataset must contain 'text' and 'label' columns.")
            return

        df_test['cleaned_text'] = df_test['text'].apply(clean_text)
        X_test = vectorizer.transform(df_test['cleaned_text'])
        y_test = df_test['label']

        # Predict using the correct method
        if isinstance(model, xgb.Booster):
            X_test_dmatrix = xgb.DMatrix(X_test)
            raw_predictions = model.predict(X_test_dmatrix)
            probabilities = expit(raw_predictions)
            predictions = (probabilities >= 0.5).astype(int)
        else:
            predictions = model.predict(X_test)

        # Compute accuracy
        accuracy = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)

        # Display confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()
        print(f"Model accuracy on unseen dataset: {accuracy:.2f}")

    except FileNotFoundError:
        print("Error: Dataset file not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    """
    Main function to take user input and predict whether the text is true or fake.
    """

    # Paths to your saved model and vectorizer
    vectorizer_path = "vectorizer.pkl"
    xgb_model_path = "xgboost_model2.pkl"
    rf_model_path = "random_forest_model2.pkl"
    dataset_path = "C:/Users/wisbr/FakeNewsDetectionAI/data/new_dataset.csv"  # Your unseen dataset

    # Load model and vectorizer
    vectorizer = load_vectorizer(vectorizer_path)
    xgb_model = load_model(xgb_model_path)
    rf_model = load_model(rf_model_path)

    # Choose which model to use (change to `xgb_model` if needed)
    selected_model = rf_model

    # Accept user input
   # user_input = input("Enter the text to check if it's true or fake: ")

    # Predict using the selected model
   # prediction, probability = predict_with_model(selected_model, vectorizer, user_input)

    # Print the prediction
   # label = "TRUE" if prediction == 1 else "FAKE"
   # print(f"The text is classified as {label} with a probability of {probability:.2f}")

    # Evaluate model on unseen dataset
    evaluate_on_new_dataset(selected_model, vectorizer, dataset_path)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        user_text = request.form.get("text")
        result = predict_with_model(load_model(user_text),load_vectorizer(user_text),user_text)
    return render_template("index.html", result=result)


if __name__ == "__main__":
    main()
app.run(debug=True)
