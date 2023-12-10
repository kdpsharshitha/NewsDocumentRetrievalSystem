import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


print("Starting Data preprocessing phase ...")
dataset = pd.read_csv("./Datasets/Dataset.csv")
dataset = dataset.loc[:, ["URL", "Snippet"]]
all_stopwords = stopwords.words('english')

print("Processing dataset ...")
for i in range(len(dataset)):
    # Remove non-alphabetic characters and replace them with spaces.
    snippet = re.sub('[^a-zA-Z]', ' ', dataset["Snippet"][i]) 
    # Converting to lower case
    snippet = snippet.lower()
    # Tokenize the snippet into list of words
    snippet = snippet.split()
    # Apply stemming using the Porter Stemmer
    ps = PorterStemmer()
    snippet = [ps.stem(word) for word in snippet if not word in set(all_stopwords)]
    # join stemmed words back into single string
    snippet = ' '.join(snippet)
    # update the it data set 
    dataset["Snippet"][i] = snippet

dataset.to_csv("./Cleaned_Datasets/Dataset.csv", index=False)

print("Dataset successfully cleaned! ..")
print("Preprocessing phase finished!")