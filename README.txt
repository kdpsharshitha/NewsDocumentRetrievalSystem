# Project Name: News Document Retreival(Environment)

# Project Drive Link:
https://github.com/kdpsharshitha/NewsDocumentRetrievalSystem

# How to run the Project:
    cd code
To install required modules, run the below command
    pip install -r requirements.txt
To run the model, run the below command
    python -m streamlit run model.py


Components:

1. Indexing:
    It generally refers to the ordering of information, it reduces the documents to the informative terms in them
 
2. Query Processsing:
    We take the query input from the user by using Streamlit and performed the following:
        - remove all non alphabets regex = [^a-zA-Z], 
        - remove whitespaces
        - convert case to lowercase 
        - tokenize words
        - remove stopwords
        - stemming
  
3. Searching:
    we calculated the tf and idf values for the snippets of the documents and the user query.
    Then we calculate the similarity between query and each document by using cosine similarity ,based on that we rank the documents and return the top 10 documents.
  
4. Refining:
   The retrieved list of document information is checked for duplicated and removed if any.
   Top 10 list of news documents are displayed that matches the search.

5. Capturing Relevance feedback -
   Relevance feedback is taken from the user and relevant documents selected are displayed.
   The results are modified based on the relevance feedback from the user.

6. Assessment Components (Precision,Recall,P-R curve) -
    We calculate the Precison , Recall and plot P-R Curve.



Team Members:
   Kattuboina Durga Pavani Sai Harshitha   -->   S20210010116
