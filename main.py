import re
import contractions
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# Load Saved Files
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing
def preprocess(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    text = contractions.fix(text)

    words = word_tokenize(text)

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# Prediction
def predict(text):

    text = preprocess(text)

    vector = vectorizer.transform([text])

    prediction = model.predict(vector)

    return prediction[0]
