from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pre_trained import clean_text
from pre_trained import df

# TF-IDF
tfidf = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf.fit_transform(df['clean_text'])

similarity_matrix = cosine_similarity(tfidf_matrix)

# ---------------- EXISTING FUNCTION (UNCHANGED) ----------------
def contranian(index, top=3):
    target_label = df.iloc[index]['predicted']
    similarities = similarity_matrix[index]

    # sort by similarity (highest first)
    similar_indices = similarities.argsort()[::-1]

    results = []

    for i in similar_indices:
        if i == index:
            continue

        if target_label == 1 and df.iloc[i]['predicted'] == -1:
            results.append((i, similarities[i]))

        elif target_label == -1 and df.iloc[i]['predicted'] == 1:
            results.append((i, similarities[i]))

        if len(results) >= top:
            break

    return results


# ---------------- NEW FUNCTION (FOR API) ----------------
def contranian_from_text(text, top=3):
    clean = clean_text(text)

    # vectorize input
    vec = tfidf.transform([clean])

    # similarity with dataset
    similarities = cosine_similarity(vec, tfidf_matrix)[0]

    # classify input
    tokens = clean.split()

    # reuse your same logic indirectly via df structure
    from pre_trained import get_sentiment  # assuming your function exists
    target_label = get_sentiment(clean, tokens)

    similar_indices = similarities.argsort()[::-1]

    results = []

    for i in similar_indices:
        if similarities[i] < 0.2:
            continue

        if target_label == 1 and df.iloc[i]['predicted'] == -1:
            results.append({
                "title": df.iloc[i]['title'],
                "predicted": int(df.iloc[i]['predicted']),
                "similarity": float(similarities[i])
            })

        elif target_label == -1 and df.iloc[i]['predicted'] == 1:
            results.append({
                "title": df.iloc[i]['title'],
                "predicted": int(df.iloc[i]['predicted']),
                "similarity": float(similarities[i])
            })

        if len(results) >= top:
            break

    return {
        "input_predicted": int(target_label),
        "results": results
    }


# ---------------- TEST (OPTIONAL) ----------------
if __name__ == "__main__":
    i = 0

    results = contranian(i)

    print("INPUT ARTICLE:\n")
    print(df.iloc[i]['title'])
    print(df.iloc[i]['predicted'])

    print("\nCONTRARIAN ARTICLES:\n")

    for r in results:
        idx, score = r
        print(df.iloc[idx]['title'])
        print("Predicted:", df.iloc[idx]['predicted'])
        print("Similarity:", score)
        print("------")