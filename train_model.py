import pandas as pd
import re
import contractions
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Download NLTK files
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# Load Dataset
df = pd.read_csv("email_spam.csv", encoding="latin-1")

# Data Cleaning
df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
df = df.rename(columns={"v1": "Category", "v2": "Text"})
df["Category"] = df["Category"].map({"ham": 0, "spam": 1})

# Stopwords
stop_words = set(stopwords.words("english"))

# Preprocessing Function
def preprocess(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    text = contractions.fix(text)

    words = word_tokenize(text)

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# Apply preprocessing
df["Text"] = df["Text"].apply(preprocess)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df["Text"],
    df["Category"],
    test_size=0.25,
    random_state=42
)

# TF-IDF
tfidf = TfidfVectorizer()

X_train = tfidf.fit_transform(X_train)

# Train Model
svm = SVC(probability=True, random_state=42)

svm.fit(X_train, y_train)

# Save Files
joblib.dump(svm, "svm_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("Model Saved Successfully!")
