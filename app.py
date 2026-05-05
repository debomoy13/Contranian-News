import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from similarity import contranian_from_text

# Load environment variables
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

app = Flask(__name__)
CORS(app)

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    data = request.json
    text = data.get('text', '')
    
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400
        
    try:
        result = contranian_from_text(text, top=3)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/generic-news', methods=['GET'])
def get_generic_news():
    category = request.args.get('category', 'general')
    
    # Map front-end labels to NewsAPI categories or search queries
    category_map = {
        'general': 'general',
        'technology': 'technology',
        'health': 'health',
        'economy': 'business',
        'climate': 'climate change',
        'immigration': 'immigration'
    }
    
    topic = category_map.get(category, 'general')
    
    # If it's a standard category, use top-headlines
    standard_categories = ['business', 'entertainment', 'general', 'health', 'science', 'sports', 'technology']
    
    if topic in standard_categories:
        url = f"https://newsapi.org/v2/top-headlines?category={topic}&language=en&apiKey={NEWS_API_KEY}"
    else:
        # For non-standard categories (Climate, Immigration), use /everything endpoint
        url = f"https://newsapi.org/v2/everything?q={topic}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
