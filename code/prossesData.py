import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('path_to_your_file.csv')

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return text

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def process_text(text):
    text = clean_text(text)
    words = text.split()
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
    return ' '.join(words)

df['processed_text'] = df['TEXT'].apply(process_text)

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['processed_text'])

print(df[['TITLE', 'processed_text', 'LABEL']].head())
