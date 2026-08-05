# 📧 Spam Email Classifier

A Machine Learning project that classifies emails as **Spam** or **Ham (Not Spam)** using **Natural Language Processing (NLP)**, **TF-IDF Vectorization**, and a **Support Vector Machine (SVM)** model.

---

## 🌐 Live Demo

🚀 **Try the application here:**

https://email-spam-classifier-9ezwd7up4y97ed4jqdbwyx.streamlit.app/

---

## 📌 Project Overview

This project demonstrates an end-to-end NLP workflow for spam email classification.

The email dataset is preprocessed using common NLP techniques, transformed into numerical features using **TF-IDF Vectorization**, and classified using a **Support Vector Machine (SVM)**.

To make the project interactive, a **Streamlit web application** allows users to enter any email message and instantly predict whether it is **Spam** or **Ham**.

---

## 🚀 Features

* Email text preprocessing
* Text cleaning and normalization
* Stopword removal
* TF-IDF Vectorization
* Spam email prediction using SVM
* Interactive Streamlit web application
* Saved trained model for fast predictions

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Contractions
* Joblib
* Streamlit

---

## 📂 Project Structure

```text
Spam-Email-Classifier/
│
├── app.py                  # Streamlit application
├── main.py                 # Prediction functions
├── train_model.py          # Train and save the model
├── email_spam.csv          # Dataset
├── svm_model.pkl           # Trained SVM model
├── tfidf_vectorizer.pkl    # Saved TF-IDF vectorizer
├── requirements.txt
├── README.md
└── Spam_Email_Classifier.ipynb
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Himanshu-97-cloud/Spam-Email-Classifier.git
```

Go to the project directory:

```bash
cd Spam-Email-Classifier
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

Run the following command once to train and save the model.

```bash
python train_model.py
```

This will generate:

* `svm_model.pkl`
* `tfidf_vectorizer.pkl`

---

## ▶️ Run the Streamlit App

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 🧠 NLP Pipeline

1. Load the dataset
2. Clean and prepare the data
3. Convert labels into numerical values
4. Convert text to lowercase
5. Remove special characters
6. Expand contractions
7. Tokenize text
8. Remove stopwords
9. Convert text into TF-IDF vectors
10. Train the Support Vector Machine (SVM)
11. Save the trained model
12. Predict whether new emails are Spam or Ham

---

## 📊 Model Used

### Support Vector Machine (SVM)

The deployed application uses an **SVM classifier**, which achieved the best performance during experimentation.

Other models explored during development include:

* Bernoulli Naive Bayes
* Logistic Regression
* Random Forest

---

## 📸 Example

### Input

```text
Congratulations!

You have won a FREE iPhone.

Click the link below to claim your prize now.
```

### Prediction

```text
🚨 Spam Email
```

---

## 📚 Learning Outcomes

Through this project, I learned:

* NLP text preprocessing
* TF-IDF Vectorization
* Spam email classification
* Machine Learning model training
* Saving and loading models using Joblib
* Building and deploying Streamlit applications

---

## 👨‍💻 Author

**Himanshu Pal**

If you found this project useful, consider giving it a ⭐ on GitHub.
