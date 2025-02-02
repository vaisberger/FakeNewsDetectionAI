# prepares the first set of data
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

#Process the dataset, shuffle it, and return TF-IDF vectors for training and testing.
def prepare_data(file_path, test_size=0.25, random_state=42, max_features=100000
                 , ngram_range=(1, 3), min_df=0.01):

    # Load dataset
    cleaned_df = pd.read_csv(file_path)

    # Remove rows with missing or empty text, title, or label
    cleaned_df = cleaned_df.dropna(subset=['text', 'title', 'label'])
    cleaned_df = cleaned_df[
        (cleaned_df['text'].str.strip() != '') &
        (cleaned_df['title'].str.strip() != '')
    ]

    # Shuffle the dataset before splitting
    cleaned_df = cleaned_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Split data into manual check and remaining data
    manual_check_df = cleaned_df.sample(frac=0.005, random_state=random_state)
    remaining_df = cleaned_df.drop(manual_check_df.index)

    x = remaining_df['text']
    y = remaining_df['label']

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    # Create and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df)
    xv_train = vectorizer.fit_transform(x_train)
    xv_test = vectorizer.transform(x_test)

    # Save the trained vectorizer
    vectorizer_path = r"/code/vectorizer1.pkl"
    joblib.dump(vectorizer, vectorizer_path)
    print(f"TF-IDF vectorizer saved to {vectorizer_path}")

    print(f"Training data shape: {xv_train.shape}")
    print(f"Testing data shape: {xv_test.shape}")

    return xv_train, xv_test, y_train, y_test, manual_check_df


# Example usage
if __name__ == "__main__":
    # Define the path to the dataset
    file_path = r'C:\Users\wisbr\FakeNewsDetectionAI\data\Processed_Dataset_news.zip'

    # Call the function to prepare data
    X_train, X_test, y_train, y_test, manual_check_df = prepare_data(file_path)

    # Print shapes to confirm everything works
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)










