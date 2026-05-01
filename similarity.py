import joblib
import numpy as np
import pandas as pd
import re
import os
import requests
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

IGNORE_WORDS = {'is', 'to', 'the', 'and', 'a', 'in', 'of', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'it', 'its', 'they', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'but', 'if', 'or', 'because', 'until', 'while', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'causing', 'die', 'doing'}

try:
    model = joblib.load('model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    print("Make sure you have run build_dataset.py and file.py first.")

def fetch_live_news(query):
    if not NEWS_API_KEY:
        print("Error: NEWS_API_KEY not found in .env")
        return []
    
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=relevancy&pageSize=20&apiKey={NEWS_API_KEY}"
    
    print(f"\n[DEBUG] Fetching live news for query: '{query}'")
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("status") == "ok":
            articles = data.get("articles", [])
            print(f"[DEBUG] Successfully fetched {len(articles)} articles.")
            return articles
        else:
            print(f"[DEBUG] NewsAPI Error: {data.get('message')}")
            return []
    except Exception as e:
        print(f"[DEBUG] Error fetching news: {e}")
        return []

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
    clean_input = clean_text(text)
    vec_input = tfidf.transform([clean_input])
    custom_input = get_custom_features(clean_input)
    features_input = np.hstack((vec_input.toarray(), custom_input))
    target_label = int(model.predict(features_input)[0])
    query_words = [w for w in clean_input.split() if w not in IGNORE_WORDS]
    search_query = " ".join(query_words[:10])
    articles = fetch_live_news(search_query)
    results = []
    print(f"[DEBUG] Analyzing articles for contrarian views (Input Sentiment: {target_label})...")
    for art in articles:
        title = art.get('title', '')
        if not title: continue
        clean_title = clean_text(title)
        vec_art = tfidf.transform([clean_title])
        custom_art = get_custom_features(clean_title)
        features_art = np.hstack((vec_art.toarray(), custom_art))
        article_label = int(model.predict(features_art)[0])
        base_similarity = cosine_similarity(vec_input, vec_art)[0][0]
        
        matches = sum(1 for kw in query_words if kw in clean_title)
        boost = matches * 0.2
        final_similarity = min(1.0, base_similarity + boost)
        
        is_contrarian = False
        if target_label == 1 and article_label == -1:
            is_contrarian = True
        elif target_label == -1 and article_label == 1:
            is_contrarian = True
        elif target_label == 0 and article_label != 0:
            is_contrarian = True
            
        print(f"  - Title: {title[:50]}... | Base Sim: {base_similarity:.2f} | Final Sim: {final_similarity:.2f} | Sentiment: {article_label}")

        if is_contrarian and final_similarity > 0.0:
            results.append({
                "title": title,
                "predicted": article_label,
                "similarity": round(float(final_similarity), 2),
                "url": art.get('url')
            })
            
        if len(results) >= top:
            break
            
    if not results:
        print("[DEBUG] No contrarian articles were found in the results.")
            
    return {
        "input_predicted": target_label,
        "results": results
    }

if __name__ == "__main__":
    test_text = "college education making students outdated."
    print(f"Testing with: {test_text}")
    print(contranian_from_text(test_text))