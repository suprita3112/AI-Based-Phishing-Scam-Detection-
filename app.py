
import streamlit as st
import joblib
import pandas as pd

from utils.preprocessing import preprocess_text


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Phishing Detector",
    page_icon="🛡️",
    layout="centered"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #050816;
    color: white;
}

/* Main Title */
h1 {
    color: #4FCBFF !important;
    text-align: center;
    font-size: 38px !important;
    font-weight: bold;
    text-shadow: 0px 0px 4px rgba(79, 203, 255, 0.4);
}

/* Subheaders */
h2, h3 {
    color: #00D9FF !important;
}

/* Paragraph Text */
p {
    color: #D6E4FF;
}

/* Text Area */
textarea {
    background-color: #0B1023 !important;
    color: white !important;
    border: 2px solid #00D9FF !important;
    border-radius: 12px !important;
}

/* Buttons */
.stButton>button {
    background-color: transparent;
    color: #00D9FF;
    border: 2px solid #00D9FF;
    border-radius: 12px;
    padding: 12px 25px;
    font-size: 18px;
    font-weight: bold;
    transition: 0.3s;
    box-shadow: 0px 0px 10px #00D9FF;
}

.stButton>button:hover {
    background-color: #00D9FF;
    color: black;
    transform: scale(1.05);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0B1023;
    border-right: 2px solid #00D9FF;
}

/* Success Box */
.stSuccess {
    border: 2px solid #00FF88;
    border-radius: 10px;
}

/* Warning Box */
.stWarning {
    border: 2px solid #FFD700;
    border-radius: 10px;
}

/* Error Box */
.stError {
    border: 2px solid #FF4B4B;
    border-radius: 10px;
}

/* Card Effect */
.block-container {
    padding-top: 2rem;
}

.cyber-box {
    border: 2px solid #00D9FF;
    padding: 20px;
    border-radius: 15px;
    background: rgba(0, 217, 255, 0.05);
    box-shadow: 0px 0px 15px rgba(0, 217, 255, 0.3);
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD TRAINED MODEL AND PREPROCESSORS
# ==========================================

@st.cache_resource
def load_models():

    model = joblib.load("best_model.pkl")

    vectorizer = joblib.load(
        "tfidf_vectorizer_new.pkl"
    )

    selector = joblib.load(
        "feature_selector.pkl"
    )

    return model, vectorizer, selector


try:
    model, vectorizer, selector = load_models()

except FileNotFoundError as e:

    st.error(
        f"Required model file not found: {e.filename}. "
        "Please run train_model.py first."
    )

    st.stop()


# ==========================================
# TITLE
# ==========================================

st.title("🛡️ AI-Based Real-Time Phishing Detection")

st.markdown("""
<div class="cyber-box">

<h3>🚨 Smart Cybersecurity Protection System</h3>

<p>
This AI-powered system uses Natural Language Processing (NLP)
and Machine Learning to analyze messages and identify
potentially suspicious, spam, and phishing-related content.
</p>

</div>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("⚡ Detection Pipeline")

st.sidebar.markdown("""
### 🧠 AI Workflow

1. Message Input
2. Text Preprocessing
3. TF-IDF Vectorization
4. Chi-Square Feature Selection
5. SVM Classification
6. Threat Prediction

---

### 🔐 Technologies Used

- Python
- Pandas
- Scikit-learn
- Natural Language Processing
- TF-IDF
- Chi-Square Feature Selection
- Support Vector Machine
- Streamlit

---

### 🤖 Models Evaluated

- Multinomial Naive Bayes
- Support Vector Machine (SVM)
- Random Forest

**Selected Model:** SVM

---

### 🎯 Threat Categories

- 🟢 Safe
- 🟡 Suspicious
- 🔴 Phishing
""")


# ==========================================
# SAMPLE MESSAGES
# ==========================================

st.subheader("📌 Sample Test Messages")

st.success(
    "Hey bro, are we meeting tomorrow?"
)

st.warning(
    "Your package delivery is pending. Verify your address."
)

st.error(
    "URGENT! Your bank account has been suspended. Verify immediately."
)


# ==========================================
# INPUT BOX
# ==========================================

st.subheader("✉️ Enter Message")

user_input = st.text_area(
    "",
    height=180,
    placeholder="Paste suspicious SMS, email, or message here..."
)


# ==========================================
# PREDICTION
# ==========================================

if st.button("🔍 Analyze Message"):

    if not user_input.strip():

        st.warning("⚠️ Please enter a message first.")

    else:

        # ----------------------------------
        # TEXT PREPROCESSING
        # ----------------------------------

        cleaned = preprocess_text(user_input)


        # ----------------------------------
        # TF-IDF VECTORIZATION
        # ----------------------------------

        vector_input = vectorizer.transform(
            [cleaned]
        )


        # ----------------------------------
        # CHI-SQUARE FEATURE SELECTION
        # ----------------------------------

        selected_input = selector.transform(
            vector_input
        )


        # ----------------------------------
        # MODEL PREDICTION
        # ----------------------------------

        prediction = model.predict(
            selected_input
        )[0]


        # ----------------------------------
        # PREDICTION PROBABILITY
        # ----------------------------------

        probability = model.predict_proba(
            selected_input
        )[0]

        safe_prob = probability[0] * 100

        phishing_prob = probability[1] * 100


        # ==================================
        # THREAT ANALYSIS
        # ==================================

        st.subheader("📊 Threat Analysis Result")


        # Model predicts HAM
        if prediction == 0:

            st.success(
                f"🟢 SAFE MESSAGE\n\n"
                f"Model spam probability: {phishing_prob:.2f}%"
            )


        # Model predicts SPAM
        else:

            if phishing_prob >= 80:

                st.error(
                    f"🔴 PHISHING / HIGH-RISK MESSAGE\n\n"
                    f"Model spam probability: {phishing_prob:.2f}%"
                )

            else:

                st.warning(
                    f"🟡 SUSPICIOUS MESSAGE\n\n"
                    f"Model spam probability: {phishing_prob:.2f}%"
                )


        # ==================================
        # PROBABILITY VISUALIZATION
        # ==================================

        chart_data = pd.DataFrame({

            "Category": [
                "Safe",
                "Spam / Suspicious"
            ],

            "Probability": [
                safe_prob,
                phishing_prob
            ]

        })

        st.subheader("📈 Detection Probability")

        st.bar_chart(
            chart_data.set_index("Category")
        )


        # ==================================
        # SUSPICIOUS KEYWORD DETECTION
        # ==================================

        suspicious_keywords = [

            "urgent",
            "verify",
            "click",
            "bank",
            "suspended",
            "reward",
            "blocked",
            "kyc",
            "otp",
            "account"

        ]

        found_keywords = []

        for word in suspicious_keywords:

            if word.lower() in user_input.lower():

                found_keywords.append(word)


        if found_keywords:

            st.subheader("⚠️ Suspicious Keywords Found")

            for keyword in found_keywords:

                st.markdown(
                    f"- 🔹 {keyword}"
                )

        else:

            st.info(
                "No predefined suspicious keywords detected."
            )


        # ==================================
        # MODEL INFORMATION
        # ==================================

        st.markdown("---")

        st.caption(
            "Prediction generated using the trained "
            "Support Vector Machine classification model."
        )

        st.caption(
            "This is an experimental SMS spam/phishing "
            "screening tool. Predictions are not a guarantee "
            "that a message is safe or malicious."
        )


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    """
    <center>
    ⚡ Developed for Cybersecurity & AI Hackathons ⚡
    </center>
    """,
    unsafe_allow_html=True
)
