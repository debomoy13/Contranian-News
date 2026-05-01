import numpy as np
import pandas as pd
import re
import joblib
import os
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- SST-2 Dataset Download ---
def download_sst2():
    url = "https://raw.githubusercontent.com/clairett/pytorch-sentiment-classification/master/data/SST2/train.tsv"
    filename = "sst2_train.tsv"
    if not os.path.exists(filename):
        print("Downloading SST-2 dataset...")
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)
    return pd.read_csv(filename, sep='\t', names=['text', 'label'])

# --- Loading Data ---
try:
    # Load your local news data
    df_local = pd.read_csv("expanded_data.csv")
    print(f"Loaded {len(df_local)} local articles.")
    
    # Load SST-2 data
    df_sst = download_sst2()
    # Map SST-2 labels (0: negative, 1: positive) to (-1, 1)
    df_sst['classification'] = df_sst['label'].map({0: -1, 1: 1})
    df_sst = df_sst.rename(columns={'text': 'content'})
    print(f"Loaded {len(df_sst)} SST-2 samples.")
    
    # Combine datasets
    df = pd.concat([
        df_local[['content', 'classification']],
        df_sst[['content', 'classification']]
    ], ignore_index=True)
    
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

def clean_text(text):
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Preprocessing data...")
df['clean_text'] = df['content'].apply(clean_text)

# Keyword scores (Design features)
supportive_words = ['benefit', 'improve', 'enhance', 'support', 'help', 'personalize', 'effective', 'efficient', 'innovative', 'accessible', 'opportunity', 'revolutionize', 'bridge', 'inclusive', 'empower', 'potential', 'success', 'breakthrough', 'advance', 'growth', 'positive', 'win', 'excellent', 'future', 'transform', 'visionary', 'leader']
opposing_words = ['harm', 'risk', 'cheat', 'replace', 'danger', 'mislead', 'bias', 'concern', 'threat', 'plagiarism', 'dependency', 'decline', 'weaken', 'destroy', 'erosion', 'shocking', 'trap', 'shrinking', 'avoid', 'bad', 'failure', 'warning', 'loss', 'crisis', 'negative', 'scam', 'wrong', 'politics', 'catch-up', 'undress', 'sexualized', 'losing', 'kill', 'warned', 'substitution']

def get_scores(text):
    tokens = text.split()
    sup = sum(1 for w in tokens if w in supportive_words)
    opp = sum(1 for w in tokens if w in opposing_words)
    return sup, opp

scores = df['clean_text'].apply(get_scores)
df['support_score'] = [s[0] for s in scores]
df['oppose_score'] = [s[1] for s in scores]

# --- Training ---
print("Training model (this may take a moment)...")
tfidf = TfidfVectorizer(max_features=1000) # Increased features for better accuracy
x_tfidf = tfidf.fit_transform(df['clean_text'])
custom_features = df[['support_score', 'oppose_score']].values
x = np.hstack((x_tfidf.toarray(), custom_features))
y = df['classification']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=2000)
model.fit(x_train, y_train)

print("\nModel Evaluation:")
print(classification_report(y_test, model.predict(x_test)))

# Save
joblib.dump(model, 'model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("\nSuccess: Saved improved model to 'model.pkl'")