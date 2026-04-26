import numpy as np
import pandas as pd
import re
from textblob import TextBlob
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("Data_news_1.csv")

# Clean text
def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['content'].apply(clean_text)
df['tokens'] = df['clean_text'].apply(lambda x: x.split())

# Keyword scores (still useful for analysis)
supportive_words = [
    'benefit', 'improve', 'enhance', 'support', 'help', 'personalize',
    'effective', 'efficient', 'innovative', 'accessible', 'opportunity',
    'revolutionize', 'bridge', 'inclusive', 'empower', 'potential'
]

opposing_words = [
    'harm', 'risk', 'cheat', 'replace', 'danger', 'mislead', 'bias',
    'concern', 'threat', 'plagiarism', 'dependency', 'decline', 'weaken',
    'destroy', 'erosion', 'shocking', 'trap', 'shrinking', 'avoid', 'bad'
]
neutral_words = [
    'study', 'research', 'report', 'show', 'find', 'suggest',
    'analyze', 'data', 'result', 'examine', 'according', 'indicate'
]

def count_keywords(tokens, keyword_list):
    return sum(1 for word in tokens if word in keyword_list)

df['support_score'] = df['tokens'].apply(lambda x: count_keywords(x, supportive_words))
df['oppose_score']  = df['tokens'].apply(lambda x: count_keywords(x, opposing_words))
df['neutral_score'] = df['tokens'].apply(lambda x: count_keywords(x, neutral_words))

# ✅ TextBlob sentiment — works well on small datasets
def get_sentiment(text, tokens):
    sup  = count_keywords(tokens, supportive_words)
    opp  = count_keywords(tokens, opposing_words)
    blob = TextBlob(text).sentiment.polarity

    if opp > sup and (opp >= 2 or blob < -0.05):
        return -1
    elif sup > opp and (sup >= 2 or blob > 0.1):
        return 1
    else:
        return 0

df['predicted'] = df.apply(
    lambda row: get_sentiment(row['clean_text'], row['tokens']), axis=1
)

# Compare against your labels
y_true = df['classification']   # ground truth: -1, 0, 1
y_pred = df['predicted']

print("=== Results on all 19 rows ===")
print(classification_report(y_true, y_pred, target_names=['negative', 'neutral', 'positive']))
print(confusion_matrix(y_true, y_pred))

# View predictions
print(df[['title', 'classification', 'predicted', 'support_score', 'oppose_score']].to_string())