import numpy as np
import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

try:
    df = pd.read_csv("expanded_data.csv")
    print(f"Loaded {len(df)} articles from expanded_data.csv")
except FileNotFoundError:
    print("Error: expanded_data.csv not found. Run build_dataset.py first!")
    exit()

def clean_text(text):
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['content'].apply(clean_text)
df['tokens'] = df['clean_text'].apply(lambda x: x.split())

supportive_words = ['benefit', 'improve', 'enhance', 'support', 'help', 'personalize', 'effective', 'efficient', 'innovative', 'accessible', 'opportunity', 'revolutionize', 'bridge', 'inclusive', 'empower', 'potential', 'success', 'breakthrough', 'advance', 'growth', 'positive', 'win', 'excellent', 'future', 'transform', 'visionary', 'leader']
opposing_words = ['harm', 'risk', 'cheat', 'replace', 'danger', 'mislead', 'bias', 'concern', 'threat', 'plagiarism', 'dependency', 'decline', 'weaken', 'destroy', 'erosion', 'shocking', 'trap', 'shrinking', 'avoid', 'bad', 'failure', 'warning', 'loss', 'crisis', 'negative', 'scam', 'wrong', 'politics', 'catch-up', 'undress', 'sexualized', 'losing', 'kill', 'warned', 'substitution']

def count_keywords(tokens, keyword_list):
    return sum(1 for word in tokens if word in keyword_list)

df['support_score'] = df['tokens'].apply(lambda x: count_keywords(x, supportive_words))
df['oppose_score'] = df['tokens'].apply(lambda x: count_keywords(x, opposing_words))

tfidf = TfidfVectorizer(max_features=200)
x_tfidf = tfidf.fit_transform(df['clean_text'])

custom_features = df[['support_score', 'oppose_score']].values
x = np.hstack((x_tfidf.toarray(), custom_features))
y = df['classification']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred))

joblib.dump(model, 'model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("\nSuccess: Saved 'model.pkl' and 'tfidf_vectorizer.pkl'")