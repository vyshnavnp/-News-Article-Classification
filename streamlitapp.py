import streamlit as st
import joblib

import nltk
nltk.download('punkt_tab')

model = joblib.load("linearsvc.pkl")

st.title("News Classification")
st.write("Enter the news article you want to classify")

news_input = st.text_area("Paste or type here")

if st.button("Classify"):
    if news_input == "":
        st.warning("Please enter a news article")
    else:
        prediction = model.predict([news_input])[0]
        st.success(f"The news article is: {prediction}")