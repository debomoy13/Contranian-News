from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pre_trained import clean_text
from pre_trained import df

tfidf=TfidfVectorizer(max_features=1000)
tfidf_matrix=tfidf.fit_transform(df['clean_text'])

similarity_matrix=cosine_similarity(tfidf_matrix)

def contranian(index, top=3):
    target_label=df.iloc[index]['predicted']
    similarities=similarity_matrix[index]
    

    #sort by similarity(highest first)

    similar_indices=similarities.argsort()[::-1]

    results=[]

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

i = 0  # choose any article index

results = contranian(i)

print("INPUT ARTICLE:\n")
print(df.iloc[i]['title'])
print(df.iloc[i]['predicted'])

print("\nCONTRARIAN ARTICLES:\n")

for r in results:
    i, score = r
    print(df.iloc[i]['title'])
    print("Predicted:", df.iloc[i]['predicted'])
    print("Similarity:", score)
    print("------")