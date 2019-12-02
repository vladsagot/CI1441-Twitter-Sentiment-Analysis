import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pickle
import random
import re
import seaborn as sns
import tensorflow as tf

from nltk.corpus import stopwords
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from numpy import array
from numpy import asarray
from numpy import zeros


def preprocess_text(sen):
    # Transform to lowercase
    sentence = sen.lower()

    # Removing URLs
    sentence = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', sentence)

    # Removing username mentions "@"
    sentence = re.sub(r'@[a-zA-Z0-9_]*', ' ', sentence)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Záéíóúñ]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


# -------------------------
# Load and sanitize data
# -------------------------

print("Load tweets...")

# Load the tweets dataset
tweet_reviews = pd.read_csv("/home/vladimir/Desktop/dataset.csv")

# Clean dataset of tweets
X = []
sentences = list(tweet_reviews['tweet'])
for sen in sentences:
    X.append(preprocess_text(sen))

# Transform positive sentiment to '1' and negative to '0'
y = tweet_reviews['class']
y = np.array(list(map(lambda x: 1 if x == "pos" else 0, y)))

print("Creating train and test data...")

# Create train and test data in a random order
# For this case 80% of data is for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# -------------------------
# Embedding layer
# -------------------------
# Transform textual data into numeric data for the first layer in the deep learning models in Keras

# Vectorize a text corpus, by turning each text into either a sequence of integer

# Create a word-to-index dictionary
# The most common words will be kept
tokenizer = Tokenizer(num_words=21000)

# Updates internal vocabulary based on a list of texts. This method creates the vocabulary index based on word
# frequency. So if you give it something like, "The cat sat on the mat." It will create a dictionary s.t. word_index[
# "the"] = 1; word_index["cat"] = 2 it is word -> index dictionary so every word gets a unique integer value.
# https://stackoverflow.com/questions/51956000/what-does-keras-tokenizer-method-exactly-do
tokenizer.fit_on_texts(X_train)

# Transforms each text in texts to a sequence of integers. So it basically takes each word in the text and replaces
# it with its corresponding integer value from the word_index dictionary.
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

# Tweet old size of characters
maxlen = 140

# pad_sequences is used to ensure that all sequences in a list have the same length
# padding: String, 'pre' or 'post': pad either before or after each sequence
# https://stackoverflow.com/questions/42943291/what-does-keras-io-preprocessing-sequence-pad-sequences-do
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

print("Load GloVe spanish file...")

# In word embeddings, every word is represented as an n-dimensional dense vector. The words that are similar will
# have similar vector. https://stackabuse.com/python-for-nlp-word-embeddings-for-deep-learning-in-keras/
#
# Data source in Spanish: https://github.com/dccuchile/spanish-word-embeddings#glove-embeddings-from-sbwc
# Algorithm: Implementation: GloVe
# Parameters: vector-size = 300, iter = 25, min-count = 5, all other parameters set as default
embeddings_dictionary = dict()
glove_file = open('/home/vladimir/Downloads/Compressed/glove-sbwc.i25.vec', encoding="utf8")

for line in glove_file:
    # splits a string into a list
    records = line.split()
    # the word is in position 0
    word = records[0]
    # get n-dimensions of array
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

# Save embeddings_dictionary into a file
file = open('embeddings_dictionary.obj', 'wb')
pickle.dump(embeddings_dictionary, file)
file.close()

print("Creating embedding matrix...")

# Creates an embedding matrix where each row number will correspond to the index of the word in the corpus
# The matrix will have 300 columns where each column will contain the GloVe word embeddings for the words in our corpus
# Create a matrix with zeros
embedding_matrix = zeros((vocab_size, 300))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

# Save embedding_matrix into a file
file = open('embedding_matrix.obj', 'wb')
pickle.dump(embedding_matrix, file)
file.close()
