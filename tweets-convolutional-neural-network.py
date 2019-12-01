import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pickle
import random
import re
import seaborn as sns
import tensorflow as tf

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from nltk.corpus import stopwords
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, Conv1D
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


# Define variables
# vocab_size = len(tokenizer.word_index) + 1
vocab_size = 17383
# Tweet old size of characters
maxlen = 140

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
tokenizer = Tokenizer(num_words=10000)

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

# -------------------------------------
# Open embedding matrix from file
# -------------------------------------
file = open('embedding_matrix.obj', 'rb')
embedding_matrix = pickle.load(file)
file.close()

# --------------------------------------------------
# Text Classification with Simple Neural Network
# --------------------------------------------------

print("Classification with Convolutional Neural Network...")

model = Sequential()

embedding_layer = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=maxlen, trainable=False)
model.add(embedding_layer)

model.add(Conv1D(300, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())

print("Training neural network...")

# Model train
# fit: Trains the model for a fixed number of epochs (iterations on a dataset)
# batch_size: Integer or None. Number of samples per gradient update
# verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
# validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data
history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

score = model.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
