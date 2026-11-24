# 🔐 AI-Based Phishing & Scam Detection

An AI-powered machine learning system designed to identify potentially malicious, phishing, and scam messages using Natural Language Processing (NLP) and machine learning techniques.

## 📌 Overview

Phishing and online scams are increasingly used to trick users into revealing sensitive information or interacting with malicious content.

This project uses **Natural Language Processing and Machine Learning** to analyze text and classify it as either **legitimate** or **potentially malicious**.

The project also provides an interactive **Streamlit web interface** where users can enter a message and receive a prediction.

---

## 🎯 Objectives

- Detect potentially malicious phishing and scam messages.
- Apply NLP techniques to process textual data.
- Compare multiple machine learning algorithms.
- Evaluate models using standard classification metrics.
- Provide an easy-to-use web interface for predictions.

---

## 🚀 Features

- 🧠 Machine Learning-based text classification
- 🔤 TF-IDF text vectorization
- 🔍 Phishing and scam detection
- 📊 Model performance evaluation
- 📈 Confusion matrix and classification metrics
- 🌐 Interactive Streamlit interface
- ⚡ Fast predictions
- 🔒 Designed as a detection aid for suspicious messages

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| TF-IDF | Text feature extraction |
| Streamlit | Web interface |
| Matplotlib | Data visualization |
| Seaborn | Visualization |

---

## 🤖 Machine Learning Models

Multiple classification algorithms were explored and compared:

- **Multinomial Naive Bayes**
- **Support Vector Machine (SVM)**
- **Random Forest**

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

---

## 🔄 Workflow

```text
Input Dataset
     ↓
Data Cleaning
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Train/Test Split
     ↓
Machine Learning Models
     ↓
Model Evaluation
     ↓
Best Performing Model
     ↓
Streamlit Web Application
     ↓
Phishing / Scam Prediction
