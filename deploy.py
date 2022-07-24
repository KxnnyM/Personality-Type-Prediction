import streamlit as st
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
import pickle

a = st.text_input("Describe yourself in few words:")
sample_test = [a]
button = st.button("Submit")

loaded_model_1 = pickle.load(open("C:/Users/kenny/Desktop/capstone/trained_model_1.sav","rb"))
loaded_model_2 = pickle.load(open("C:/Users/kenny/Desktop/capstone/trained_model_2.sav","rb"))
loaded_model_3 = pickle.load(open("C:/Users/kenny/Desktop/capstone/trained_model_3.sav","rb"))
loaded_model_4 = pickle.load(open("C:/Users/kenny/Desktop/capstone/trained_model_4.sav","rb"))

postdf=pd.read_csv("post.csv")

def pred(sample_test):
    vector = CountVectorizer(stop_words='english', max_features=1500)
    vector = vector.fit(postdf.post_list)




    features = vector.transform(sample_test)
    transform = TfidfTransformer()
    finalfeatures = transform.fit_transform(features).toarray()
    sample_input = transform.fit_transform(features).toarray()

    vect = vector.transform(sample_test).toarray()
    loaded_model_1.predict(vect)
    if loaded_model_1.predict(vect) == 1:
        a = "I"

    else:
        a = "E"




    vect = vector.transform(sample_test).toarray()
    loaded_model_2.predict(vect)
    if loaded_model_2.predict(vect) == 1:
        b = "N"

    else:
        b = "S"




    vect = vector.transform(sample_test).toarray()
    loaded_model_3.predict(vect)
    if loaded_model_3.predict(vect) == 1:
        c = "T"

    else:
        c = "F"




    vect = vector.transform(sample_test).toarray()
    loaded_model_4.predict(vect)
    if loaded_model_4.predict(vect) == 1:
        d = "J"

    else:
        d = "P"


    st.write(a, b, c, d)

if button:
    vector = CountVectorizer(stop_words='english', max_features=1500)
    features = vector.fit_transform(sample_test)

    transform = TfidfTransformer()
    sample_input = transform.fit_transform(features).toarray()

    pred(sample_test)


