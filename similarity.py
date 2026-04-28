# engine.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pre_trained import df, clean_text, classify

tfidf = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf.fit_transform(df['clean_text'])

def get_contrarian_results(input_text, top=3):
    clean = clean_text(input_text)

    # vectorize input
    vec = tfidf.transform([clean])

    # similarity
    similarities = cosine_similarity(vec, tfidf_matrix)[0]

    # predict input stance
    target_label = classify(input_text)

    sorted_indices = similarities.argsort()[::-1]

    results = []

    for i in sorted_indices:
        if similarities[i] < 0.2:
            continue

        # strict opposition
        if target_label == 1 and df.iloc[i]['predicted'] == -1:
            results.append({
                "title": df.iloc[i]['title'],
                "similarity": float(similarities[i]),
                "stance": -1
            })

        elif target_label == -1 and df.iloc[i]['predicted'] == 1:
            results.append({
                "title": df.iloc[i]['title'],
                "similarity": float(similarities[i]),
                "stance": 1
            })

        if len(results) >= top:
            break

    return {
        "input_stance": target_label,
        "results": results
    }