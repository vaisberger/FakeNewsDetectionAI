import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re
import zipfile

# load data from the zip file
with zipfile.ZipFile("Cleaned_Dataset_news.zip", 'r') as zip_ref:
    zip_ref.extractall(".")

cleaned_df = pd.read_cvs("Cleaned_Dataset_news.cvs")

# 1. Word Count and Sentence Length Features
cleaned_df['word_count'] = cleaned_df['cleaned_text'].apply(lambda x: len(str(x).split()))
cleaned_df['sentence_count'] = cleaned_df['cleaned_text'].apply(lambda x: len(re.split(r'[.!?]+', str(x))) - 1)
cleaned_df['avg_sentence_length'] = np.where(
    cleaned_df['sentence_count'] == 0, 0,
    cleaned_df['word_count'] / cleaned_df['sentence_count']
)

# 2. TF-IDF Features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_df['cleaned_text'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
cleaned_df = pd.concat([cleaned_df, tfidf_df], axis=1)

# 3. Keyword Features
keywords = ['breaking', 'exclusive', 'alert', 'update']
for keyword in keywords:
    cleaned_df[f'contains_{keyword}'] = cleaned_df['cleaned_text'].apply(lambda x: int(keyword in str(x)))

# 4. Label Encoding
label_encoder = LabelEncoder()
cleaned_df['label'] = label_encoder.fit_transform(cleaned_df['label'])

# Save processed data for model training
output_path = 'data/Processed_Dataset_news.csv'
cleaned_df.to_csv(output_path, index=False)
print(f"Processed data saved to {output_path}")
