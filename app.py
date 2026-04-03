from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load dataset (using news.csv which was renamed from processed_news.csv)
try:
    df = pd.read_csv("news.csv")
    
    # Train model on startup for simplicity
    X = df['text']
    y = df['label']

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X, y)
except Exception as e:
    print(f"Error loading data or training model: {e}")

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    confidence = ""

    if request.method == "POST":
        news = request.form["news"]
        
        if news:
            vec = vectorizer.transform([news])
            prediction = model.predict(vec)
            prob = model.predict_proba(vec)
            
            confidence = round(max(prob[0]) * 100, 2)

            if prediction[0] == 1:
                result = "Real ✅"
            else:
                result = "Fake ❌"

    return render_template("index.html", result=result, confidence=confidence)

if __name__ == "__main__":
    # Using port 5001 to avoid conflicts with existing services on port 5000
    app.run(debug=True, port=5001)