#  News Article Classification using Machine Learning

This project classifies news articles into categories using a **Linear Support Vector Classifier (LinearSVC)**. 
The model is trained on labeled news data and deployed as a **Streamlit web app**.

## 🔍 Project Overview

- **Goal**: Automatically predict the category of a news article based on its content.
- **Model**: LinearSVC with TF-IDF vectorization
- **Interface**: Built with Streamlit for easy web interaction
- **Use Case**: Media classification, content moderation, news aggregation

##  Folder Structure

news-article-classification/
├── streamlitapp.py # Streamlit web app
├── preprocess.py # Python function to preprocess and clean data
├── linearsvc.pkl # Trained ML model
├── requirements.txt # Python dependencies for streamlit
├── NewArticleClassification.ipynb # EDA + Model building + Insights 
└── README.md # Project documentation
