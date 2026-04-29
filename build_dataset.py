import pandas as pd
import re
from newsapi import NewsApiClient
from textblob import TextBlob
import time
import os
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv('NEWS_API_KEY')
TOPICS = ['technology', 'AI risks', 'AI benefits', 'climate crisis', 'climate solutions', 'politics controversy', 'economic growth', 'market crash', 'health breakthrough']
ARTICLES_PER_TOPIC = 40

supportive_words = ['benefit', 'improve', 'enhance', 'support', 'help', 'personalize', 'effective', 'efficient', 'innovative', 'accessible', 'opportunity', 'revolutionize', 'bridge', 'inclusive', 'empower', 'potential', 'success', 'breakthrough', 'advance', 'growth', 'positive', 'win', 'excellent', 'future', 'transform', 'visionary', 'leader']
opposing_words = ['harm', 'risk', 'cheat', 'replace', 'danger', 'mislead', 'bias', 'concern', 'threat', 'plagiarism', 'dependency', 'decline', 'weaken', 'destroy', 'erosion', 'shocking', 'trap', 'shrinking', 'avoid', 'bad', 'failure', 'warning', 'loss', 'crisis', 'negative', 'scam', 'wrong', 'politics', 'catch-up', 'undress', 'sexualized', 'losing', 'kill', 'warned', 'substitution']

def clean_text(text):
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_sentiment_label(text):
    tokens = text.lower().split()
    sup = sum(1 for word in tokens if word in supportive_words) #comprehensive looping 
    opp = sum(1 for word in tokens if word in opposing_words) #comprehensive looping 
    blob = TextBlob(text).sentiment.polarity
    
    if opp > sup and (opp >= 1 or blob < -0.01):
        return -1
    elif sup > opp and (sup >= 1 or blob > 0.01):
        return 1
    else:
        return 0

def build_dataset():
    if NEWS_API_KEY == '8bb3dc3bb11546d1ab5f9dbe8d0691a8E': ### DATA ENV file leaked Check later 
        print("Error: Please paste your News API key in build_dataset.py")
        return

    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    all_articles = []

    print(f"Fetching articles for topics: {TOPICS}...")
    
    for topic in TOPICS:
        print(f"  Fetching: {topic}")
        try:
            response = newsapi.get_everything(q=topic, language='en', sort_by='relevancy', page_size=ARTICLES_PER_TOPIC)
            articles = response.get('articles', [])
            
            for art in articles:
                content = art.get('content') or art.get('description') or ''
                if len(content) < 50: continue
                
                clean = clean_text(content)
                label = get_sentiment_label(clean)
                
                all_articles.append({
                    'title': art.get('title'),
                    'content': content,
                    'clean_text': clean,
                    'classification': label
                })
        except Exception as e:
            print(f"  Error fetching {topic}: {e}")
        
        time.sleep(1) #putting api to sleep . CLosing the api to work 

    df = pd.DataFrame(all_articles)
    df.to_csv('expanded_data.csv', index=False)
    print(f"\nSuccess! Created expanded_data.csv with {len(df)} articles.")

if __name__ == "__main__":
    build_dataset()
