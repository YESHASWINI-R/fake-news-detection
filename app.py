import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download stopwords (only once needed)
nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open('fake_news_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Preprocessing function
def preprocess(text):
    ps = PorterStemmer()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# Page config
st.set_page_config(page_title="Fake News Detector", layout="centered")

# Inject dark theme CSS
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stTextArea textarea {
        background-color: #262730;
        color: #ffffff;
    }
    .stButton > button {
        background-color: #4a6cf7;
        color: white;
        font-size: 16px;
        padding: 8px 20px;
        border-radius: 8px;
        border: none;
    }
    .stAlert {
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title("Fake News Prediction App")

# Text input
user_input = st.text_area("Enter a news article paragraph:")

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Process and predict
        processed = preprocess(user_input)
        vector_input = vectorizer.transform([processed]).toarray()
        prediction = model.predict(vector_input)[0]
        proba = model.predict_proba(vector_input)[0]
        confidence = round(max(proba) * 100, 2)

        if prediction == 'REAL':
            st.success("This news is REAL.")
        else:
            st.error("This news is FAKE.")

        st.info(f"Prediction confidence: {confidence}%")

