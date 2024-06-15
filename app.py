import pandas as pd
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image

data = pd.read_csv('amazon_product.csv')

data = data.drop('id', axis = 1)

stemmer = SnowballStemmer('english')

def tokenize_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    stemmed= [stemmer.stem(w) for w in tokens]
    return " ".join(stemmed)


#tokenizing and stemming of title and description column
data['stem_token'] = data.apply(lambda row: tokenize_stem(row['Title'] + " " + row['Description']),axis=1)


tfidf = TfidfVectorizer(tokenizer=tokenize_stem)

def cosine_sim(txt1,txt2):
    # matrix = tfidf.fit_transform([txt1,txt2])
    txt1_concatenate = ' '.join(txt1)
    txt2_concatenate = ' '.join(txt2)
    tfidf_matrix = tfidf.fit_transform(txt1_concatenate,txt2_concatenate)
    return cosine_similarity(tfidf_matrix)[0][1]


def search_product(query):
    stemmed_q = tokenize_stem(query)

    text_data = [stemmed_q] + data['stem_token'].tolist()
    vectorizer = CountVectorizer().fit_transform(text_data)
    vectors = vectorizer.toarray()
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    data['similarity'] = cosine_similarities

    #showing the 10 contents(with columns like title,desc...) which has more(that is why ascending = false(descending)) similarity with user input
    res = data.sort_values(by = ['similarity'], ascending =  False).head(10)[['Title','Description','Category']]
    return res

# web app
img = Image.open('amazon logo.png')
st.image(img,width = 600)
st.title("Search Engine and Product Recommendation system On an Data")


query = st.text_input("Enter Product Name")
submit = st.button("Search")
if submit:
    res = search_product(query)
    st.write(res)