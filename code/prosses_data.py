import chardet
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import SnowballStemmer
import zipfile
import io
import nltk

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')  # Download punkt tokenizer
nltk.data.clear_cache()

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

def clean_text(text):
    """
    Cleans the input text by:
    - Lowercasing
    - Removing URLs, HTML tags, punctuation, numbers
    - Tokenizing
    - Removing stopwords
    """
    if pd.isnull(text):  # Handle NaN values
        return ""
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove content inside square brackets
    text = re.sub(r'\[.*?\]', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Replace all non-word characters with a space
    text = re.sub(r'\W', ' ', text)

    # Remove punctuation (we will keep spaces, as they are word boundaries)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)

    # Remove words with digits
    text = re.sub(r'\w*\d\w*', '', text)

    # Collapse multiple spaces into one and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = nltk.word_tokenize(text)

    # Stem each token using the SnowballStemmer
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(stemmed_tokens)

# Path to the files
true = pd.read_csv('C:/Users/wisbr/FakeNewsDetectionAI/data/true.csv')
fake = pd.read_csv('C:/Users/wisbr/FakeNewsDetectionAI/data/fake.csv')

true['label'] = 1
fake['label'] = 0

true = true.drop_duplicates()
fake = fake.drop_duplicates()

data=pd.concat([fake,true],axis=0)

data['title']=data['title'].astype(str)
data['text']=data['text'].astype(str)

# Save the cleaned DataFrame to a CSV file
data['title']=data['title'].apply(clean_text)
data['text']=data['text'].apply(clean_text)

data.insert(0, 'id', range(1, len(data) + 1))
output_path = r'C:\Users\wisbr\FakeNewsDetectionAI\data\Cleaned_Dataset.csv'
# Use encoding="utf-8-sig" for better compatibility
data.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"Cleaned data saved to {output_path} with UTF-8 encoding")# UTF-8 with BOM








