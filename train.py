import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    """ Trains the fake news detection model. """
    print("Loading processed data...")
    df = pd.read_csv('processed_news.csv')
    
    # Fill any missing values in 'cleaned_text'
    df['cleaned_text'] = df['cleaned_text'].fillna('')
    
    # Split features and labels
    X = df['cleaned_text']
    y = df['label']
    
    # Train-test split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature Extraction using TF-IDF
    print("Extracting features using TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Model Training: Logistic Regression
    print("Training Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    
    # Model Evaluation
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and vectorizer
    print("Saving model and vectorizer to disk...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('tfidf.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    
    print("Model and vectorizer saved successfully.")
    return accuracy

if __name__ == "__main__":
    train_model()
