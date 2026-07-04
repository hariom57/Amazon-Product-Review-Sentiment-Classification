import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset

# This ensures your data stays out of C: drive
os.environ['HF_HOME'] = './huggingface_cache'

def load_and_clean_data():
    # Load a small sample (10k reviews) for quick iteration
    # The 'amazon_polarity' dataset is standard for sentiment tasks
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset('SetFit/amazon_polarity', split='test[:10000]') 
    df = dataset.to_pandas()
    
    # The dataset has 'content' (review) and 'label' (0 or 1)
    df = df.rename(columns={'text': 'review_text', 'label': 'sentiment'})
    
    # Basic cleaning
    df = df.dropna(subset=['review_text'])
    df['review_text'] = df['review_text'].str.lower()
    
    return df

def get_splits(df):
    return train_test_split(df['review_text'], df['sentiment'], test_size=0.2, random_state=42)

def get_vectorizer():
    return TfidfVectorizer(stop_words='english', max_features=5000)