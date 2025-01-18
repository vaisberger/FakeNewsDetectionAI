import chardet
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import zipfile
import io

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')  # Download punkt tokenizer
nltk.data.clear_cache()

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Cleans the input text by:
    - Lowercasing
    - Removing URLs, HTML tags, punctuation, numbers
    - Tokenizing
    - Removing stopwords
    - Lemmatizing
    """
    if pd.isnull(text):  # Handle NaN values
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)  # Remove punctuation
    text = re.sub(r'\w*\d\w*', '', text)  # Remove digits and words with digits
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Path to the ZIP file
zip_path = r'C:\Users\wisbr\FakeNewsDetectionAI\data\Dataset_news.zip'

# Open the ZIP file
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

    # Use StringIO to create a file-like buffer for Pandas
    cleaned_chunks = []
    buffer = io.StringIO(decoded_file)
    for chunk in pd.read_csv(buffer, chunksize=10000):
        # Rename column if the ID column is labeled as 0
        if '0' in chunk.columns:
            chunk.rename(columns={'0': 'id'}, inplace=True)

        # Clean the text and title columns
        chunk['cleaned_text'] = chunk['text'].apply(clean_text)
        chunk['cleaned_title'] = chunk['title'].apply(clean_text)

        # Append cleaned chunk
        cleaned_chunks.append(chunk)

    # Combine all chunks into a single DataFrame
    cleaned_df = pd.concat(cleaned_chunks, ignore_index=True)

# Save the cleaned DataFrame to a CSV file

output_path = r'C:\Users\wisbr\FakeNewsDetectionAI\data\Cleaned_Dataset_news.csv'
# Use encoding="utf-8-sig" for better compatibility
cleaned_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"Cleaned data saved to {output_path} with UTF-8 encoding")# UTF-8 with BOM








