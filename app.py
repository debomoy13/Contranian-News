from flask import Flask, request, jsonify
from flask_cors import CORS
from similarity import contranian_from_text

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

if __name__ == '__main__':
    app.run(debug=True, port=5000)
