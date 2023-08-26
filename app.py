from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from tensorflow.keras.layers import Softmax
import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app)

# Sentiment analysis function
def analyze_sentiment(input_text):
    roberta = "cardiffnlp/twitter-roberta-base-sentiment"
    model = TFAutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    labels = ['Negative', 'Neutral', 'Positive']

    user_in = input_text

    tweet_words = []

    for word in user_in.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)

    tweet_proc = " ".join(tweet_words)

    encoded_tweet = tokenizer(tweet_proc, return_tensors='tf')
    output = model(encoded_tweet)

    scores = output.logits[0].numpy()
    scores = tf.nn.softmax(scores)

    max_score_index = np.argmax(scores)
    max_label = labels[max_score_index]
    return max_label

@app.route('/', methods=['POST'])
def perform_sentiment_analysis():
    data = request.form
    user_input = data.get('user_input')
    processed_result = analyze_sentiment(user_input)
    
    return jsonify({"sentiment": processed_result})

if __name__ == '__main__':
    app.run(debug=True)
