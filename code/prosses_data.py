import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.data.clear_cache()

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

#Cleans the input text
def clean_text(text):

    if pd.isnull(text):  # Handle NaN values
        return ""

    text = text.lower()   # Convert to lowercase

    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs

    text = re.sub(r'\[.*?\]', '', text)   # Remove content inside square brackets

    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags

    text = re.sub(r'\W', ' ', text)  # Replace all non-word characters with a space

    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)  # Remove punctuation

    text = re.sub(r'\w*\d\w*', '', text) # Remove words with digits

    # Collapse multiple spaces into one and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = nltk.word_tokenize(text)

    # Stem each token using the SnowballStemmer
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(stemmed_tokens)


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








