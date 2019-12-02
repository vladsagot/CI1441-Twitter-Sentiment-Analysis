import re

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize


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
nltk.download('punkt')
print("Load tweets...")

# Load the tweets dataset
tweet_reviews = pd.read_csv("/home/vladimir/Desktop/dataset.csv")

# Clean dataset of tweets
X = []
sentences = list(tweet_reviews['tweet'])
for sen in sentences:
    X.append(preprocess_text(sen))

print(X)

all_words = []
for sent in X:
    tokenize_word = word_tokenize(sent)
    for word in tokenize_word:
        all_words.append(word)

unique_words = set(all_words)
print(len(unique_words))

# Unique words in dataset: 20342
