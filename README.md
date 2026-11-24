# AI-Based Phishing & Scam Detection System

An NLP and machine learning-based application that classifies text messages as legitimate (ham) or potentially unwanted/spam. It includes a Streamlit interface for testing messages.

## Project Overview

The project applies text preprocessing, TF-IDF vectorization, feature selection and machine learning algorithms to identify spam messages.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF
- Matplotlib and Seaborn
- Streamlit

## Dataset

The project uses the UCI SMS Spam Collection dataset, along with 20 additional custom examples.

The dataset contains two classes:
- Ham: legitimate messages
- Spam: unwanted or suspicious messages

## Machine Learning Workflow

1. Text preprocessing
2. Feature extraction using TF-IDF
3. Feature selection using SelectKBest with Chi-square
4. Stratified train-test split (80:20)
5. Model training
6. Model evaluation and comparison
7. Best-model selection
8. Streamlit application integration

## Algorithms Implemented

- Multinomial Naive Bayes
- Support Vector Machine (SVM)
- Random Forest Classifier

## Model Evaluation

The models were evaluated using accuracy, precision, recall, F1-score, ROC-AUC and confusion matrices.

### Results

| Metric | Naive Bayes | SVM | Random Forest |
|---|---:|---:|---:|
| Accuracy | 98.17% | 98.17% | 97.69% |
| Precision | 100% | 92.59% | 100% |
| Recall | 85.82% | 93.28% | 82.09% |
| F1-score | 92.37% | 92.94% | 90.16% |
| ROC-AUC | 98.73% | 98.84% | 99.15% |

**Selected model: SVM**, based on its highest F1-score.

## Project Structure

```text
phishing_detection_system/
├── app.py
├── train_model.py
├── requirements.txt
├── dataset/
│   └── SMSSpamCollection
├── utils/
│   └── preprocessing.py
├── best_model.pkl
├── tfidf_vectorizer_new.pkl
├── feature_selector.pkl
├── model_results.csv
├── model_comparison.png
└── confusion_matrices.png
```

## Installation and Execution

Clone the repository:

```bash
git clone https://github.com/suprita3112/AI-Based-Phishing-Scam-Detection-.git
cd AI-Based-Phishing-Scam-Detection-
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the models:

```bash
python train_model.py
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

## Limitations

- The current dataset focuses on SMS messages.
- Performance on other forms of phishing, such as malicious websites and email links, has not been independently validated.
- Predictions are classification outputs and should be treated as decision-support information, not guarantees.

## Conclusion

The project demonstrates how NLP and machine learning can be combined to classify SMS messages. Among the three evaluated algorithms, SVM achieved the highest F1-score of 92.94% and was selected for integration into the Streamlit application.