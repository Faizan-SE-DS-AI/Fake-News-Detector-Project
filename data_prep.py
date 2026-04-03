import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading stopwords...")
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans the input text by:
    1. Lowercasing
    2. Removing special characters and numbers
    3. Removing stopwords
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove special characters, numbers, and multiple spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    
    return " ".join(cleaned_words)

def generate_sample_data():
    """Generates a small synthetic dataset for demonstration purposes."""
    print("Generating sample data...")
    fake_news = [
        "Aliens land in New York City and demand pizza.",
        "Scientists discover that eating chocolate makes you invisible.",
        "Breaking: The moon is actually made of Swiss cheese.",
        "Government secretly replaces all birds with surveillance drones.",
        "New law requires citizens to walk backwards on Tuesdays."
    ] * 200
    
    real_news = [
        "NASA launches its most powerful telescope to study distant galaxies.",
        "The global economy shows signs of recovery after the recent downturn.",
        "New clinical trials show promising results for a malaria vaccine.",
        "Local community garden wins award for sustainable urban farming.",
        "Tech giant announces new environmental initiative to reduce carbon footprint."
    ] * 200
    
    data = []
    for news in fake_news:
        data.append({"text": news, "label": 1}) # 1 for Fake
    for news in real_news:
        data.append({"text": news, "label": 0}) # 0 for Real
        
    df = pd.DataFrame(data)
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def prepare_dataset(csv_path=None):
    """
    Loads dataset from CSV or generates a sample one.
    Cleans and saves the processed data.
    """
    if csv_path:
        try:
            print(f"Loading data from {csv_path}...")
            df = pd.read_csv(csv_path)
            # Ensure we have 'text' and 'label' columns
            # Mapping Fake/Real to 1/0 if needed
            if 'label' in df.columns:
                if df['label'].dtype == object:
                    df['label'] = df['label'].map({'FAKE': 1, 'REAL': 0, 'Fake': 1, 'Real': 0})
            elif 'category' in df.columns: # Sometimes it's called category
                 df['label'] = df['category'].map({'FAKE': 1, 'REAL': 0, 'Fake': 1, 'Real': 0})
            
            # Use 'text' or 'title' + 'text'
            if 'title' in df.columns and 'text' in df.columns:
                 df['text'] = df['title'] + " " + df['text']
            
            df = df[['text', 'label']]
        except Exception as e:
            print(f"Error loading CSV: {e}. Falling back to sample data.")
            df = generate_sample_data()
    else:
        df = generate_sample_data()
    
    print("Cleaning text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Save processed data
    df.to_csv('processed_news.csv', index=False)
    print("Processed data saved to 'processed_news.csv'.")
    return df

if __name__ == "__main__":
    # You can pass a real CSV path here if you have one
    prepare_dataset()
