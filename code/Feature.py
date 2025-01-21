import chardet
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import zipfile
import io  # Required to handle the string as file

# Define your custom keywords
custom_keywords = ['breaking', 'exclusive', 'alert', 'shocking', 'percent', 'miracle', 'exposed']

# Path to the ZIP file containing the cleaned dataset
zip_path = r'C:\Users\wisbr\FakeNewsDetectionAI\data\Cleaned_Dataset_news.zip'

# Extract and read the cleaned dataset
with zipfile.ZipFile(zip_path, 'r') as z:
    # Extract the first file name in the ZIP
    file_name = z.namelist()[0]
    print(f"Reading file: {file_name}")

    # Detect encoding using a sample
    with z.open(file_name) as f:
        sample = f.read(10000)  # Read a sample to detect encoding
        encoding = chardet.detect(sample)['encoding']
        print(f"Detected file encoding: {encoding}")

    # Read the entire file into memory and decode it
    with z.open(file_name) as f:
        decoded_file = f.read().decode(encoding, errors='replace')  # Decode the file

    # Convert the decoded file string to a file-like object for pandas
    cleaned_df = pd.read_csv(io.StringIO(decoded_file))  # Read CSV from decoded string

# Drop rows with NaNs in 'cleaned_text' and 'cleaned_title'
cleaned_df = cleaned_df.dropna(subset=['cleaned_text', 'cleaned_title'])

# FEATURE ENGINEERING

# 1. Word Count and Sentence Length Features for both 'cleaned_text' and 'cleaned_title'
for column in ['cleaned_text', 'cleaned_title']:
    cleaned_df[f'{column}_word_count'] = cleaned_df[column].apply(lambda x: len(str(x).split()))
    cleaned_df[f'{column}_sentence_count'] = cleaned_df[column].apply(
        lambda x: len(re.split(r'[.!?]+', str(x))) - 1
    )
    cleaned_df[f'{column}_avg_sentence_length'] = np.where(
        cleaned_df[f'{column}_sentence_count'] == 0, 0,
        cleaned_df[f'{column}_word_count'] / cleaned_df[f'{column}_sentence_count']
    )

# 2. TF-IDF Features for both 'cleaned_text' and 'cleaned_title'
# Set min_df to exclude rare words
tfidf_vectorizer_text = TfidfVectorizer(max_features=500, min_df=0.01)  # Adjust min_df as needed
tfidf_matrix_text = tfidf_vectorizer_text.fit_transform(cleaned_df['cleaned_text'])
tfidf_df_text = pd.DataFrame(tfidf_matrix_text.toarray(), columns=tfidf_vectorizer_text.get_feature_names_out())

tfidf_vectorizer_title = TfidfVectorizer(max_features=500, min_df=0.01)  # Adjust min_df as needed
tfidf_matrix_title = tfidf_vectorizer_title.fit_transform(cleaned_df['cleaned_title'])
tfidf_df_title = pd.DataFrame(tfidf_matrix_title.toarray(), columns=tfidf_vectorizer_title.get_feature_names_out())

# Concatenate TF-IDF features
cleaned_df = pd.concat([cleaned_df, tfidf_df_text, tfidf_df_title], axis=1)

# 3. Keyword Features for both 'cleaned_text' and 'cleaned_title'
for keyword in custom_keywords:
    cleaned_df[f'contains_{keyword}_text'] = cleaned_df['cleaned_text'].apply(
        lambda x: int(keyword.lower() in str(x).lower())
    )
    cleaned_df[f'contains_{keyword}_title'] = cleaned_df['cleaned_title'].apply(
        lambda x: int(keyword.lower() in str(x).lower())
    )

# Save the processed data to a new CSV file
output_path = r'C:\Users\wisbr\FakeNewsDetectionAI\data\Processed_Dataset_news.csv'
cleaned_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"Processed data saved to {output_path}")




