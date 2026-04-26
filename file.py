import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("Data_news_1.csv")
print(df.head())
print(df.info())

def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'\n', ' ', text)  # remove newlines
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text
df['clean_text'] = df['content'].apply(clean_text)

#tokenization
df['tokens']= df['clean_text'].apply(lambda x:x.split())

#words storing
supportive_words = ["improve", "enhance", "enable", "support", "benefit", 
 "personalized", "efficient", "transform", "innovation"]

opposing_words =["cheating", "decline", "harm", "undermine", "loss", 
 "dependency", "bias", "risk", "problem", "erode", "crisis"]

neutral_words = ["however", "but", "while", "although", "depends", 
 "balance", "concerns", "mixed", "tradeoff"]

def count_keywords(tokens, keyword_list):
    return sum(1 for word in tokens if word in keyword_list)

df['support_score'] = df['tokens'].apply(lambda x: count_keywords(x, supportive_words))
df['oppose_score'] = df['tokens'].apply(lambda x: count_keywords(x, opposing_words))
df['neutral_score'] = df['tokens'].apply(lambda x: count_keywords(x, neutral_words))


tfidf = TfidfVectorizer(max_features=1000)
x_tfidf = tfidf.fit_transform(df['clean_text'])

#new features 

custom_features = df[['support_score','oppose_score','neutral_score']].values
x=np.hstack((x_tfidf.toarray(), custom_features))
y=df['classification']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))