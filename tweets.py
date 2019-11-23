import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

tweet_reviews = pd.read_csv("/home/vladimir/Desktop/dataset.csv")

print(tweet_reviews)

sns.countplot(x='class', data=tweet_reviews)

plt.show()