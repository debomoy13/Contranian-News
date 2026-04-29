import joblib
import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity

try:
    model = joblib.load('model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    df = pd.read_csv("expanded_data.csv")
    tfidf_matrix = tfidf.transform(df['clean_text'])
except Exception as e:
    print(f"Warning: Could not load model or data: {e}")
    print("Make sure you have run build_dataset.py and file.py first.")

supportive_words = ['benefit', 'improve', 'enhance', 'support', 'help', 'personalize', 'effective', 'efficient', 'innovative', 'accessible', 'opportunity', 'revolutionize', 'bridge', 'inclusive', 'empower', 'potential', 'success', 'breakthrough', 'advance', 'growth', 'positive', 'win', 'excellent', 'future', 'transform', 'visionary', 'leader']
opposing_words = ['harm', 'risk', 'cheat', 'replace', 'danger', 'mislead', 'bias', 'concern', 'threat', 'plagiarism', 'dependency', 'decline', 'weaken', 'destroy', 'erosion', 'shocking', 'trap', 'shrinking', 'avoid', 'bad', 'failure', 'warning', 'loss', 'crisis', 'negative', 'scam', 'wrong', 'politics', 'catch-up', 'undress', 'sexualized', 'losing', 'kill', 'warned', 'substitution']

def clean_text(text):
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_custom_features(text):
    tokens = text.lower().split()
    sup = sum(1 for word in tokens if word in supportive_words)
    opp = sum(1 for word in tokens if word in opposing_words)
    return np.array([[sup, opp]])

def contranian_from_text(text, top=3):
    clean = clean_text(text)
    
    vec = tfidf.transform([clean])
    custom = get_custom_features(clean)
    features = np.hstack((vec.toarray(), custom))
    
    target_label = int(model.predict(features)[0])
    
    similarities = cosine_similarity(vec, tfidf_matrix)[0]
    similar_indices = similarities.argsort()[::-1]
    
    results = []
    for i in similar_indices:
        if similarities[i] > 0.95: continue
        if similarities[i] < 0.1: break
        
        article_label = df.iloc[i]['classification']
        
        if target_label == 1 and article_label == -1:
            results.append({
                "title": df.iloc[i]['title'],
                "predicted": int(article_label),
                "similarity": round(float(similarities[i]), 2)
            })
        elif target_label == -1 and article_label == 1:
            results.append({
                "title": df.iloc[i]['title'],
                "predicted": int(article_label),
                "similarity": round(float(similarities[i]), 2)
            })
        elif target_label == 0 and article_label != 0:
            results.append({
                "title": df.iloc[i]['title'],
                "predicted": int(article_label),
                "similarity": round(float(similarities[i]), 2)
            })
            
        if len(results) >= top:
            break
            
    return {
        "input_predicted": target_label,
        "results": results
    }

if __name__ == "__main__":
    test_text = "AI is amazing and will revolutionize education by providing personal help to every student."
    print(f"Testing with: {test_text}")
    print(contranian_from_text(test_text))