import streamlit as st
import pickle
import re
import string
import os
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Download NLTK data once ──────────────────────────────────────────────────
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.insert(0, nltk_data_path)

for pkg, sub in [("stopwords", "corpora"), ("wordnet", "corpora"), ("omw-1.4", "corpora")]:
    try:
        nltk.data.find(f"{sub}/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=nltk_data_path, quiet=True)

# ── Load model & vectorizer ──────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(__file__)
    with open(os.path.join(base, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base, "tfidf.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf

# ── Preprocessing ────────────────────────────────────────────────────────────
@st.cache_resource
def get_nlp_tools():
    lemmatizer = WordNetLemmatizer()
    stop_words  = set(stopwords.words("english"))
    return lemmatizer, stop_words

def preprocess_text(text, lemmatizer, stop_words):
    if not text or pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = re.sub(r"\d+", " num ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

def predict(subject, body, model, tfidf, lemmatizer, stop_words):
    combined = subject + " " + body
    clean    = preprocess_text(combined, lemmatizer, stop_words)
    vec      = tfidf.transform([clean])
    pred     = model.predict(vec)[0]
    proba    = model.predict_proba(vec)[0]
    return pred, proba

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }

    .title-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .title-box h1 { color: #e2e8f0; font-size: 2.2rem; margin: 0; }
    .title-box p  { color: #94a3b8; font-size: 1rem; margin-top: 0.5rem; }

    .result-spam {
        background: linear-gradient(135deg, #ff4757, #c0392b);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(255,71,87,0.4);
    }
    .result-ham {
        background: linear-gradient(135deg, #2ed573, #1e9e5e);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(46,213,115,0.4);
    }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .metric-card .val  { font-size: 2rem; font-weight: 700; }
    .metric-card .lbl  { font-size: 0.85rem; color: #64748b; }

    .info-badge {
        background: #e2e8f0;
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        font-size: 0.8rem;
        color: #475569;
        display: inline-block;
        margin: 0.2rem;
    }
    div[data-testid="stTextArea"] textarea {
        border-radius: 10px !important;
        border: 1.5px solid #cbd5e1 !important;
    }
    div[data-testid="stTextInput"] input {
        border-radius: 10px !important;
        border: 1.5px solid #cbd5e1 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-box">
    <h1>📧 Email Spam Detector</h1>
    <p>Powered by Machine Learning · Trained on the Enron Dataset · 98%+ Accuracy</p>
</div>
""", unsafe_allow_html=True)

# ── Load artifacts ───────────────────────────────────────────────────────────
try:
    model, tfidf = load_artifacts()
    lemmatizer, stop_words = get_nlp_tools()
    st.success("✅ Model loaded successfully!", icon="🤖")
except FileNotFoundError:
    st.error("⚠️ model.pkl or tfidf.pkl not found. Please run `train.py` first to generate them.")
    st.stop()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Single Email", "📋 Batch Check", "ℹ️ About Model"])

# ════════════════════════════════════════════════════════
# TAB 1 — Single Email Prediction
# ════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Enter Email Details")

    subject = st.text_input(
        "📌 Subject Line",
        placeholder="e.g. Congratulations! You've won a prize...",
    )
    body = st.text_area(
        "📝 Email Body",
        placeholder="Paste the full email body here...",
        height=200
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🚀 Analyse Email", use_container_width=True, type="primary")

    if predict_btn:
        if not subject.strip() and not body.strip():
            st.warning("Please enter at least a subject or body.")
        else:
            with st.spinner("Analysing..."):
                pred, proba = predict(subject, body, model, tfidf, lemmatizer, stop_words)

            # Result banner
            if pred == 1:
                st.markdown('<div class="result-spam">🚨 SPAM DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-ham">✅ LEGITIMATE EMAIL (Ham)</div>', unsafe_allow_html=True)

            # Confidence metrics
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="val" style="color:#2ed573">{proba[0]*100:.1f}%</div>
                    <div class="lbl">Ham Confidence</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="val" style="color:#ff4757">{proba[1]*100:.1f}%</div>
                    <div class="lbl">Spam Confidence</div>
                </div>""", unsafe_allow_html=True)

            # Confidence bar
            st.markdown("#### Confidence Distribution")
            st.progress(float(proba[1]), text=f"Spam probability: {proba[1]*100:.1f}%")

    # Quick test examples
    st.markdown("---")
    st.markdown("#### 💡 Try a Quick Example")
    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        if st.button("🚨 Spam Example", use_container_width=True):
            st.session_state["ex_subject"] = "WINNER!! Claim your $1,000,000 prize NOW"
            st.session_state["ex_body"]    = "Congratulations! You have been selected. Click here to claim your FREE prize. Limited time offer. Send your bank details immediately to receive your cash reward!"
            st.rerun()
    with ex_col2:
        if st.button("✅ Ham Example", use_container_width=True):
            st.session_state["ex_subject"] = "Team standup meeting - Thursday 10am"
            st.session_state["ex_body"]    = "Hi team, just a reminder about our weekly sync tomorrow at 10am. Please review the Q3 report before joining. Let me know if you can't make it."
            st.rerun()

    # Auto-fill from quick examples
    if "ex_subject" in st.session_state:
        st.info(f"**Subject:** {st.session_state['ex_subject']}\n\n**Body:** {st.session_state['ex_body']}")

# ════════════════════════════════════════════════════════
# TAB 2 — Batch Prediction
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📋 Batch Email Classification")
    st.info("Upload a CSV file with columns: `Subject` and `Message`")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(batch_df):,} rows")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("🚀 Classify All Emails", type="primary"):
                with st.spinner(f"Classifying {len(batch_df):,} emails..."):
                    batch_df["combined"] = (
                        batch_df.get("Subject", pd.Series([""] * len(batch_df))).fillna("") +
                        " " +
                        batch_df.get("Message", pd.Series([""] * len(batch_df))).fillna("")
                    )
                    batch_df["clean"]      = batch_df["combined"].apply(lambda t: preprocess_text(t, lemmatizer, stop_words))
                    vecs                   = tfidf.transform(batch_df["clean"])
                    batch_df["Prediction"] = model.predict(vecs)
                    probas                 = model.predict_proba(vecs)
                    batch_df["Ham %"]      = (probas[:, 0] * 100).round(1)
                    batch_df["Spam %"]     = (probas[:, 1] * 100).round(1)
                    batch_df["Prediction"] = batch_df["Prediction"].map({0: "✅ Ham", 1: "🚨 Spam"})

                spam_count = (batch_df["Prediction"] == "🚨 Spam").sum()
                ham_count  = (batch_df["Prediction"] == "✅ Ham").sum()

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Total Emails",  len(batch_df))
                mc2.metric("🚨 Spam Found", spam_count)
                mc3.metric("✅ Ham Found",  ham_count)

                result_df = batch_df.drop(columns=["combined", "clean"])
                st.dataframe(result_df, use_container_width=True)

                csv_out = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results CSV",
                    data=csv_out,
                    file_name="spam_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ════════════════════════════════════════════════════════
# TAB 3 — About
# ════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🧠 About This Model")

    st.markdown("""
    <div style="background:white;padding:1.5rem;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.08)">
    <h4>📊 Dataset</h4>
    <p>Trained on the <strong>Enron Email Dataset</strong> — 33,716 emails (17,171 spam + 16,545 ham), one of the most widely used email classification benchmarks.</p>

    <h4>⚙️ Pipeline</h4>
    <ul>
        <li>Text cleaning: lowercase, URL/email/number replacement, punctuation removal</li>
        <li>NLP: stopword removal + WordNet lemmatization</li>
        <li>Features: TF-IDF with unigrams + bigrams (50,000 features)</li>
        <li>Classifier: Logistic Regression (C=5)</li>
    </ul>

    <h4>📈 Performance</h4>
    </div>
    """, unsafe_allow_html=True)

    perf_data = {
        "Metric": ["Accuracy", "Precision (Spam)", "Recall (Spam)", "F1-Score (Spam)", "ROC-AUC"],
        "Score":  ["~98.5%",   "~98%",             "~99%",          "~98.5%",          "~99.8%"]
    }
    st.table(pd.DataFrame(perf_data))

    st.markdown("""
    <div style="background:#f1f5f9;padding:1rem;border-radius:10px;margin-top:1rem">
        <span class="info-badge">🐍 Python 3.10+</span>
        <span class="info-badge">🤖 scikit-learn</span>
        <span class="info-badge">📝 NLTK</span>
        <span class="info-badge">🎈 Streamlit</span>
        <span class="info-badge">🗂 pandas</span>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#94a3b8;font-size:0.8rem'>"
    "Built with ❤️ using Streamlit · Enron Spam Classifier</p>",
    unsafe_allow_html=True
)
