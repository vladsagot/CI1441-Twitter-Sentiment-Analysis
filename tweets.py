import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
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


def preprocess_text(sen):
    # Removing html tags
    sentence = sen

    # Transform to lowercase
    sentence = sentence.lower()

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


tweet_reviews = pd.read_csv("/home/vladimir/Desktop/dataset.csv")

# print(tweet_reviews)

# sns.countplot(x='class', data=tweet_reviews)

# plt.show()

# Clean dataset of tweets

X = []
sentences = list(tweet_reviews['tweet'])
for sen in sentences:
    X.append(preprocess_text(sen))

# Transform positive sentiment to '1' and negative to '0'

y = tweet_reviews['class']
y = np.array(list(map(lambda x: 1 if x == "pos" else 0, y)))

# Create train and test data in a random order

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
