import re
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#%matplotlib inline

train = pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv')

train_original=train.copy()

test = pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/test.csv')

test_original=test.copy()

combine = pd.concat([test,train])



combine_result = combine.dropna()
print(combine_result)



# combine_result["label"].value_counts()

# def remove_pattern(text,pattern):
#     r = re.findall(pattern,text)
#     for i in r:
#         text = re.sub(i,"",text)
#     return text

# combine_result['Tidy_Tweets'] = np.vectorize(remove_pattern)(combine_result['tweet'], "@[\w]*")

# combine_result.head()

# combine_result['Tidy_Tweets'] = combine_result['Tidy_Tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

# combine_result.head(10)

from nltk import PorterStemmer

tokenized_tweet = combine_result['Tidy_Tweets'].apply(lambda x: x.split())

tokenized_tweet.head()

ps = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [ps.stem(i) for i in x])

tokenized_tweet.head()


