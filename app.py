import streamlit as st
import re
from transformers import pipeline

# ✅ Initialize sentiment analyzer using CPU only
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1  # Force CPU to avoid meta tensor issues
)

# ✅ Text cleaning function
def clean_text(text: str) -> str:
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)       # remove URLs
    text = re.sub(r'@\w+', '', text)                          # remove mentions
    text = re.sub(r'#\w+', '', text)                          # remove hashtags
    text = re.sub(r'[^\w\s]', '', text)                       # remove punctuation
    text = text.lower().strip()
    return text

# ✅ Sentiment prediction using BERT
def get_bert_sentiment(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "Neutral"
    result = sentiment_analyzer(text)[0]
    label = result["label"]
    return "Positive" if label == "POSITIVE" else "Negative"

# ✅ Generate chatbot reply
def chatbot_response(sentiment: str) -> str:
    if sentiment == "Negative":
        return st.selectbox(
            "Suggested Response:",
            [
                "We're sorry to hear about your experience. Please contact us so we can make it right.",
                "Thank you for your feedback. We are looking into this issue.",
                "We apologize for the inconvenience you experienced. Your feedback is valuable to us."
            ]
        )
    elif sentiment == "Positive":
        return "We're glad you had a great experience! Thank you for sharing your feedback."
    return "Thank you for your review."

# ✅ Streamlit UI
st.set_page_config(page_title="Travel Review Sentiment Analyzer", layout="centered")
st.title("✈️ Travel Review Sentiment Analyzer")
st.write("Enter a traveler review below and click **Analyze** to detect the sentiment and get a chatbot reply.")

# Text input
review_input = st.text_area("✍️ Review Text", height=150)

# Analyze button
if st.button("Analyze"):
    cleaned = clean_text(review_input)
    sentiment = get_bert_sentiment(cleaned)
    reply = chatbot_response(sentiment)

    st.markdown("### ✅ Results")
    st.markdown("**🧼 Cleaned Review:**")
    st.write(cleaned or "<empty>")

    st.markdown("**🧠 Predicted Sentiment:**")
    st.write(f"🔹 {sentiment}")

    st.markdown("**💬 Chatbot Reply:**")
    st.write(reply)
