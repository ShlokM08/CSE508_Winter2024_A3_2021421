import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import unidecode
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from q1 import *
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import warnings
import re
import unidecode
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Ensure necessary NLTK resources are available
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text):
    """Preprocess review text according to specified steps with warning suppression."""
    if not isinstance(text, str) or not text.strip():
        return ""  # Return empty string for non-str input or empty strings
    
    # Suppress BeautifulSoup warning for non-HTML content
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MarkupResemblesLocatorWarning)
        text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
        
    text = unidecode.unidecode(text)  # Remove accented characters
    # Placeholders for further preprocessing steps (e.g., acronym expansion)
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters, keep letters and spaces
    words = text.lower().split()  # Normalize text to lowercase
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    meaningful_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatization
    return " ".join(meaningful_words)


def main():
    # Load the processed data
    file_path = 'processed_data.csv'  # Update to your file path
    df = pd.read_csv(file_path)

    # Apply preprocessing to the 'reviewText' or 'processed_reviewText' column, as appropriate
    df['processed_reviewText'] = df['processed_reviewText'].apply(preprocess_text)

    # Descriptive output to verify preprocessing
    print(df.head())

    # Optionally, save the fully preprocessed DataFrame for future use
    df.to_csv('fully_preprocessed_data.csv', index=False)

if __name__ == "__main__":
    main()
