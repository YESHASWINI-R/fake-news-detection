# 📰 Fake News Prediction using Machine Learning

This project focuses on detecting whether a news article is **Real** or **Fake** using Natural Language Processing (NLP) and Machine Learning techniques.
The model analyzes news content and predicts the authenticity of the article based on textual patterns and word usage. 

The project was implemented using **Logistic Regression** with **TF-IDF Vectorization** for text feature extraction.

---

## 📌 Project Objective

The main goal of this project is to classify news articles into:

* REAL News
* FAKE News

using NLP preprocessing and machine learning algorithms.

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* NLTK
* Matplotlib
* Seaborn
* Pickle

---

## 📂 Dataset Information

The dataset contains news articles along with their labels indicating whether the article is real or fake. 

### Dataset Features

| Column       | Description                |
| ------------ | -------------------------- |
| id           | Unique ID for news article |
| title        | News headline              |
| text/content | News article content       |
| label        | REAL or FAKE               |

### Dataset Size

* 6335 news articles
* Balanced dataset containing both real and fake news articles

---

## 📊 Exploratory Data Analysis

Performed:

* Dataset inspection
* Missing value checking
* Label analysis
* Text preprocessing
* Dataset shape verification

The dataset did not contain missing values. 

---

## ⚙️ Text Preprocessing

Several NLP preprocessing steps were applied before training:

* Removing special characters
* Converting text to lowercase
* Tokenization
* Stopword removal
* Stemming using PorterStemmer

Example:

```python id="z8k2xw"
running → run
playing → play
```

---

## 🔢 Feature Extraction

Text data was converted into numerical vectors using:

### TF-IDF Vectorization

TF-IDF helps identify important words in news articles by assigning weights to frequently meaningful terms.

---

## 🤖 Machine Learning Model

### Logistic Regression

The Logistic Regression model was trained on TF-IDF transformed text data for binary classification.

---

## 📈 Model Performance

| Metric            | Accuracy |
| ----------------- | -------- |
| Training Accuracy | 95.18%   |
| Testing Accuracy  | 91.63%   |

The model achieved strong performance in distinguishing fake news from real news articles. 

---

## 🧪 Prediction Examples

### Example 1

```text id="1g7xpw"
"Hillary Clinton allies begin an unprecedented attack on the FBI..."
```

Prediction:

```text id="7m4zje"
FAKE
```

---

### Example 2

```text id="0x2ylt"
"U.S. Secretary of State John Kerry met with French President..."
```

Prediction:

```text id="d2j9kp"
REAL
```

---

## 💾 Saved Files

The trained files are saved using Pickle:

```bash id="l0n7rw"
fake_news_model.pkl
tfidf_vectorizer.pkl
```

These files can later be used for deployment or real-time predictions.

---

## ▶️ How to Run the Project

### 1️⃣ Install Required Libraries

```bash id="s8f4va"
pip install numpy pandas nltk scikit-learn matplotlib seaborn
```

### 2️⃣ Download NLTK Stopwords

```python id="g7k1nm"
import nltk
nltk.download('stopwords')
```

### 3️⃣ Run the Notebook

Execute all notebook cells step by step.

---

## 🔄 Project Workflow

```text id="w2n6dj"
Dataset Collection
        ↓
Text Preprocessing
        ↓
Stopword Removal & Stemming
        ↓
TF-IDF Vectorization
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Fake News Prediction
```

---

## 🚀 Future Improvements

* Add Deep Learning models like LSTM and BERT
* Create a web application using Flask or Streamlit
* Real-time news verification system
* Multi-language fake news detection
* Deploy the project on cloud platforms

---

## 👩‍💻 Developed By

**Yeshaswini R**
Computer Science Engineering Student

---

## ⭐ GitHub

If you found this project useful, consider giving the repository a star ⭐
