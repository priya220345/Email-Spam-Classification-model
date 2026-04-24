"""
Run this script ONCE locally to generate model.pkl and tfidf.pkl
    python train.py
"""

import pandas as pd
import pickle, re, string, os, warnings
warnings.filterwarnings("ignore")

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)
from nltk.corpus import stopwords
from nltk.stem   import WordNetLemmatizer

from sklearn.model_selection         import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model            import LogisticRegression
from sklearn.metrics                 import accuracy_score, classification_report

# ── Preprocessing ────────────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))

def preprocess_text(text):
    if pd.isna(text): return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " url ",   text)
    text = re.sub(r"\S+@\S+",                 " email ", text)
    text = re.sub(r"\d+",                     " num ",   text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

# ── Load data ────────────────────────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv("enron_spam_data.csv")
print(f"   {len(df):,} emails loaded")
print(f"   Spam: {(df['Spam/Ham']=='spam').sum():,}  Ham: {(df['Spam/Ham']=='ham').sum():,}")

# ── Preprocess ───────────────────────────────────────────────────────────────
print("⚙️  Preprocessing text (may take ~60s)...")
df["combined_text"] = df["Subject"].fillna("") + " " + df["Message"].fillna("")
df["clean_text"]    = df["combined_text"].apply(preprocess_text)
df["label"]         = (df["Spam/Ham"] == "spam").astype(int)

# ── Split ────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)
print(f"   Train: {len(X_train):,}  Test: {len(X_test):,}")

# ── Vectorise ─────────────────────────────────────────────────────────────────
print("🔢 Fitting TF-IDF vectoriser...")
tfidf = TfidfVectorizer(
    max_features=50000, ngram_range=(1, 2),
    sublinear_tf=True,  min_df=2, max_df=0.95
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)
print(f"   Vocabulary size: {len(tfidf.vocabulary_):,}")

# ── Train ─────────────────────────────────────────────────────────────────────
print("🤖 Training Logistic Regression...")
model = LogisticRegression(C=5, max_iter=1000, solver="lbfgs")
model.fit(X_train_tfidf, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
preds = model.predict(X_test_tfidf)
acc   = accuracy_score(y_test, preds)
print(f"\n✅ Accuracy: {acc:.4f}")
print(classification_report(y_test, preds, target_names=["Ham", "Spam"]))

# ── Save ──────────────────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f: pickle.dump(model, f)
with open("tfidf.pkl", "wb") as f: pickle.dump(tfidf, f)
print("💾 Saved: model.pkl  tfidf.pkl")
print("\n🚀 Now run:  streamlit run app.py")
