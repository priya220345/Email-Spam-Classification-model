# 📧 Email Spam Detector

A machine learning web app that classifies emails as **Spam** or **Ham (Legitimate)** using Logistic Regression + TF-IDF, trained on the Enron Email Dataset.

**Live App →** *(add your Streamlit Cloud URL here after deploying)*

---

## 🚀 Features
- Single email prediction with confidence scores
- Batch CSV upload & classification with downloadable results
- Clean, modern UI with dark header and color-coded results

---

## 📊 Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | ~98.5% |
| Precision (Spam) | ~98% |
| Recall (Spam) | ~99% |
| F1-Score | ~98.5% |
| ROC-AUC | ~99.8% |

---

## 🗂 Project Structure
```
spam_app/
├── app.py               # Streamlit app
├── train.py             # Training script (run locally)
├── model.pkl            # Trained Logistic Regression model
├── tfidf.pkl            # Fitted TF-IDF vectorizer
├── requirements.txt     # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/spam-detector.git
cd spam-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model (generates model.pkl & tfidf.pkl)
```bash
python train.py
```
> Place `enron_spam_data.csv` in the same folder before running.

### 4. Run the app
```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub (**include model.pkl and tfidf.pkl**)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set main file to `app.py`
4. Click **Deploy** 🎉

---

## 🛠 Tech Stack
- Python 3.10+
- scikit-learn (TF-IDF + Logistic Regression)
- NLTK (stopwords + lemmatization)
- Streamlit
- pandas
