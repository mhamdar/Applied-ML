import string
import csv
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Importing data
data = pd.read_csv("reddit_train.csv").to_numpy()
data = data[:, [1,2]] # Removing ID column

# Lowercasing all
for i, comment in enumerate(data[:, 0]):
    data[i, 0] = data[i, 0].lower()
print("All words have been lowercased.")

# Tokenizing & removing stop words
stopwords = set(stopwords.words("english"))
for i, comment in enumerate(data[:, 0]):
    words = word_tokenize(comment)
    filtered_comm = []
    for word in words:
       if word not in stopwords:
           filtered_comm.append(word)
    data[i, 0] = filtered_comm
print("The following words have been removed from all comments: " + str(stopwords))

# Removing punctuation
punctuations = list(string.punctuation)
for i, comment in enumerate(data[:, 0]):
    newlist = []
    for word in comment:
        if word not in punctuations:
            newlist.append(word)
    data[i, 0] = newlist
print("The following punctuation has been removed from all comments: " + str(punctuations))

# Stemming
for i, comment in enumerate(data[:, 0]):
    newlist = []
    ps = PorterStemmer()
    for word in comment:
        newlist.append(ps.stem(word))
    data[i, 0] = newlist


#counting words
#from collections import Counter
#word_count = Counter()
#for comment in data[:, 0]: word_count.update(comment)

#removing rare words
#freq_threshold = 3

#for i, comment in enumerate(data[:,0]):
 #   data[i, 0] = [word for word in comment if word_count[word] >= freq_threshold]

#print("Removed rare words")
data[:, 0] = [" ".join(comment) for comment in data[:, 0]]

## PROCESSING THE TEST DATA
# Importing data
test = pd.read_csv("reddit_test.csv").to_numpy()
test = test[:, 1] # Removing ID column

# Lowercasing all
for i, comment in enumerate(test):
    test[i] = test[i].lower()
print("All words have been lowercased.")

# Tokenizing & removing stop words
for i, comment in enumerate(test):
    words = word_tokenize(comment)
    filtered_comm = []
    for word in words:
       if word not in stopwords:
           filtered_comm.append(word)
    test[i] = filtered_comm
print("The following words have been removed from all comments: " + str(stopwords))

# Removing punctuation
for i, comment in enumerate(test):
    newlist = []
    for word in comment:
        if word not in punctuations:
            newlist.append(word)
    test[i] = newlist
print("The following punctuation has been removed from all comments: " + str(punctuations))

# Stemming
for i, comment in enumerate(test):
    newlist = []
    ps = PorterStemmer()
    for word in comment:
        newlist.append(ps.stem(word))
    test[i] = newlist

#removing rare words
#for i, comment in enumerate(test):
   # test[i] = [word for word in comment if word_count[word] >= freq_threshold]

#print("Removed rare words")
test = [" ".join(comment) for comment in test]

test = np.array(test)
print(len(test))

np.savetxt("train_clean.csv", data, fmt="%s", encoding="utf-8", delimiter="\t")
np.savetxt("test_clean.csv", test, fmt="%s", encoding="utf-8", delimiter="\t")
