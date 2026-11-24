
import re


def preprocess_text(text):
    text = str(text).lower()

    # Replace URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URL ",
        text
    )

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
