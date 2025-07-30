import pandas as pd
import numpy as np

import string 
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Downloading required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", '', text)
    
    # Remove non-alphabetic characters (keep only a-z and space)
    text = re.sub(r"[^a-z\s]", '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and short words
    tokens = [word for word in tokens if word not in stop_words and len(word) >= 2]
    
    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join back to string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text