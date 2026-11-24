
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from utils.preprocessing import preprocess_text


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

print("Original dataset loaded:", df.shape)


# ==========================================
# 2. ADD CUSTOM PHISHING DATA
# ==========================================

custom_data = pd.DataFrame({

    "label": [
        "spam", "spam", "spam", "spam", "spam",
        "spam", "spam", "spam", "spam", "spam",
        "spam", "spam", "spam", "spam", "spam",
        "ham", "ham", "ham", "ham", "ham"
    ],

    "message": [
        "Your ATM card is blocked. Update KYC immediately.",
        "Your bank account has been suspended. Verify now.",
        "Click this link to claim your reward immediately.",
        "Urgent! Your account will be locked today.",
        "Verify your PAN card details now.",
        "Your Netflix payment failed. Re-enter card details now.",
        "Amazon alert: payment failed. Update billing information.",
        "Your WhatsApp account will be deleted unless verified.",
        "Your PayPal account is temporarily restricted.",
        "Apple ID locked due to suspicious login attempt.",
        "Google account compromised. Verify password immediately.",
        "Dear customer, your UPI access has been suspended.",
        "Your debit card has unusual activity. Confirm now.",
        "Final warning! Your account will be terminated today.",
        "You won a free iPhone. Claim reward immediately.",
        "Hey bro are we meeting tomorrow?",
        "Can you send me the notes?",
        "Lets go for lunch after class.",
        "Happy birthday have a great day.",
        "Assignment submission deadline is tomorrow."
    ]
})

df = pd.concat([df, custom_data], ignore_index=True)

df = df.dropna(subset=["label", "message"])
df["label"] = df["label"].str.strip().str.lower()
df = df[df["label"].isin(["ham", "spam"])]
df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)


# ==========================================
# 3. FEATURE SELECTION
# ==========================================

df["label_num"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

df["clean_message"] = df["message"].apply(
    preprocess_text
)

X = df["clean_message"]
y = df["label_num"]

print("\nDataset shape:", df.shape)
print("\nClass distribution:")
print(df["label"].value_counts())


# ==========================================
# 4. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 5. TF-IDF VECTORIZATION
# ==========================================

tfidf = TfidfVectorizer(
    max_features=7000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# ==========================================
# 6. CHI-SQUARE FEATURE SELECTION
# ==========================================

selector = SelectKBest(
    score_func=chi2,
    k=min(5000, X_train_tfidf.shape[1])
)

X_train_selected = selector.fit_transform(
    X_train_tfidf,
    y_train
)

X_test_selected = selector.transform(
    X_test_tfidf
)

print("\nFeatures before selection:", X_train_tfidf.shape[1])
print("Features after selection:", X_train_selected.shape[1])


# ==========================================
# 7. APPLY THREE ML ALGORITHMS
# ==========================================

models = {

    "Naive Bayes": MultinomialNB(alpha=0.5),

    "SVM": SVC(
        kernel="linear",
        probability=True,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


# ==========================================
# 8. MODEL TRAINING AND PREDICTION
# ==========================================

results = []
trained_models = {}
predictions = {}

for name, model in models.items():

    print("\nTraining", name, "...")

    model.fit(
        X_train_selected,
        y_train
    )

    y_pred = model.predict(X_test_selected)

    y_probability = model.predict_proba(
        X_test_selected
    )[:, 1]

    trained_models[name] = model
    predictions[name] = y_pred

    # ======================================
    # 9. MODEL EVALUATION
    # ======================================

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    auc = roc_auc_score(
        y_test, y_probability
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": auc
    })

    print("\n", name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("ROC-AUC:", auc)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Ham", "Spam"],
        zero_division=0
    ))


# ==========================================
# 10. CONFUSION MATRICES
# ==========================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, y_pred) in zip(
    axes, predictions.items()
):

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        ax=ax
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=300)
plt.show()


# ==========================================
# 11. MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1-Score",
    ascending=False
)

print("\nMODEL COMPARISON:")
print(results_df.to_string(index=False))

results_df.to_csv(
    "model_results.csv",
    index=False
)

results_df.set_index("Model").plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("ML Model Performance Comparison")
plt.ylabel("Performance Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("model_comparison.png", dpi=300)
plt.show()


# ==========================================
# 12. SELECT BEST MODEL
# ==========================================

best_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_name]

print("\nBEST MODEL:", best_name)

# Save the selected model and preprocessing objects.
# These use new filenames so the existing Streamlit
# application is not broken during this step.

joblib.dump(
    best_model,
    "best_model.pkl"
)

joblib.dump(
    tfidf,
    "tfidf_vectorizer_new.pkl"
)

joblib.dump(
    selector,
    "feature_selector.pkl"
)

print("\nAll models trained successfully!")

print("\nGenerated files:")
print("1. best_model.pkl")
print("2. tfidf_vectorizer_new.pkl")
print("3. feature_selector.pkl")
print("4. model_results.csv")
print("5. model_comparison.png")
print("6. confusion_matrices.png")
