import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

all_stopwords = stopwords.words('english')
ps = PorterStemmer()
tf = TfidfVectorizer()

# Load CSV data
csv_path = Path("Cleaned_Datasets/Dataset.csv")
news = pd.read_csv(csv_path)

if 'relevant_documents' not in st.session_state:
    st.session_state.relevant_documents = []
retrieved = 0
relevant = 0
map_val = 0

# Streamlit App
st.title("News Document Retrieval App")
st.markdown('<h3 style="color:blue;">User Input Form:</h3>', unsafe_allow_html=True)
name = st.text_input("Enter your name:")
query = st.text_input("Search for nature-related news document:")

# Function to retrieve news documents
def retrieve_news_documents(name, query):
    if not name or not query:
        return None, None, None

    proc_query = re.sub('[^a-zA-Z]', ' ', query)
    proc_query = proc_query.lower()
    proc_query = proc_query.split()
    proc_query = [ps.stem(word) for word in proc_query if not word in set(all_stopwords)]
    proc_query = ' '.join(proc_query)
    tfidf_news = tf.fit_transform(news["Snippet"])
    tfidf_proc_query = tf.transform([proc_query])
    similarity_scores = cosine_similarity(tfidf_news, tfidf_proc_query).flatten()

    return name, query, similarity_scores

# Retrieve news documents
name, query, similarity_scores = retrieve_news_documents(name, query)

# Display documents that matches search
if name and query and similarity_scores is not None:
    st.success(f"Hey {name}! Here's what you're looking for:")
    if np.any(similarity_scores > 0):
        top_indices = np.argsort(similarity_scores)[-10:][::-1]
        top_urls = news.loc[top_indices, "URL"].tolist()
        st.subheader("Top 10 news documents that match your search:")
        for i, url in enumerate(top_urls, start=0):
            st.write(f"{i}. {url}")

        relevant_ind = st.text_input(f"Please select the relevant documents. Enter indices (separated-space):").split()

        # Submit Relevance Feedback
        if st.button("Submit Relevance Feedback"):
            for i in relevant_ind:
                docs = top_urls[int(i)]
                st.session_state.relevant_documents.append(docs)

            # Display relevant documents
            st.write("Relevant Documents that you selected:", st.session_state.relevant_documents)

            recall = list()
            precision = list()
            for key in top_urls:
                try:
                    retrieved += 1
                    if str(key) in st.session_state.relevant_documents:
                        relevant += 1
                        map_val += round((relevant/retrieved), 2)
                    total_relevant = len(st.session_state.relevant_documents)
                    recall.append(round((relevant/total_relevant),2))
                    precision.append(round((relevant/retrieved), 2))
                except ZeroDivisionError:
                    st.write("Error: Division by zero.")
            plt.plot(recall,precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title("PRECISION-RECALL CURVE")
            st.pyplot(plt)
            
            relevance_feedback_scores = np.zeros(len(top_urls))
            for i, url in enumerate(top_urls):
                if url in st.session_state.relevant_documents:
                    relevance_feedback_scores[i] = 1
            combined_scores = similarity_scores[top_indices] + relevance_feedback_scores
            updated_indices = np.argsort(combined_scores)[-10:][::-1]
            updated_top_urls = news.loc[updated_indices, "URL"].tolist()

            # Display documents based on relevance feedback
            st.subheader("Top 10 news documents based on your Relevance Feedback:")
            for i, url in enumerate(updated_top_urls, start=0):
                st.write(f"{i}. {url}")

    else:
        st.write("No matches Found!")
    st.success("Done!")
