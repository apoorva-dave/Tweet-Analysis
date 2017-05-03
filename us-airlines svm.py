import string

import pandas as pd
from decorator import getfullargspec
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.util import mark_negation
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
import nltk
from sklearn.base import TransformerMixin
import random
import re

# Read data
from sklearn.svm import LinearSVC


def read_data():
    data = pd.read_csv("C:/Users/Apoorva/PycharmProjects/TweetsClassification/Tweets.csv", header=0, delimiter=",", quoting=15)

    #14640 us airlines tweets
    print (data.shape)  # (25000, 3)
    #print (data["review"][0])  # Check out the review
    #print (data["sentiment"][0])  # Check out the sentiment (0/1)
    return data

# Shuffle the data

def randomize(data):
    sentiment_data = list(zip(data["text"], data["airline_sentiment"]))
    # random.shuffle(sentiment_data)
    return sentiment_data

# Generate training data

def train_data(sentiment_data):
    #return random_data
    # 80% for training
    train_X, train_y = zip(*sentiment_data[:13500])
    return list(train_X),list(train_y)

# Generate test data

def test_data(sentiment_data):
    # Keep 20% for testin
    test_X, test_y = zip(*sentiment_data[13501:])
    return list(test_X),list(test_y)

# Clean data.

def clean_text(text):
    text = text.replace("<br />", " ")
    # text = re.sub('[^A-Za-z0-9 ]+', '', str(text))
    text = text.encode().decode("utf-8")
    # Removing urls from text
    text = re.sub(r'http\S+', '', text,flags=re.MULTILINE)

    return text



from nltk import bigrams

# print(list(bigrams("We are playing with bigrams".split())))

# [('We', 'are'), ('are', 'playing'), ('playing', 'with'), ('with', 'bigrams')]

def stemming_tokenizer(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in word_tokenize(text)]

# Get Features matrix
def get_tweets_feature_matrix(classifier, data):
    vocab = classifier.named_steps['vectorizer'].get_feature_names()
    print(vocab)
    feature_matrix = []
    for line in data:
        tokenized_line = re.sub('[^A-Za-z0-9 ]+', '', str(line)).split(" ")
        line_matrix = [0] * len(vocab)
        for word in tokenized_line:
            if word in vocab:
                index_found = vocab.index(word)
                line_matrix[index_found] = line_matrix[index_found] + 1
            print(line_matrix)
        feature_matrix.append(line_matrix)

    return feature_matrix


def unigrams(train_X,train_y,test_X,test_y):


    clf = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       tokenizer=word_tokenize,
                                       # ! Comment line to include mark_negation and uncomment next line
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "),
                                       max_features=10000)),
        ('classifier', LinearSVC())
    ])
    # clf = Pipeline([
    #     ('vectorizer', TfidfVectorizer(tokenizer=stemming_tokenizer,
    #                                    stop_words=stopwords.words('english') + list(string.punctuation))),
    #     ('classifier', MultinomialNB(alpha=0.05)),
    # ])

    clf.fit(train_X, train_y)
    print("Printing accuracy")
    print(clf.score(test_X, test_y))

    # with mark_negation 0.84760000000000002
    # without mark_negation 0.84440000000000004

    #feature_matrix = get_tweets_feature_matrix(clf, train_X)
    # print("Printing feature matrix")
    # print(feature_matrix)
    # print("Getting feature names")
    # print(clf.named_steps['vectorizer'].get_feature_names())

def bigrams(train_X,train_y,test_X,test_y):

    bigram_clf = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(2, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier',LinearSVC())
    ])
    # bigram_clf = Pipeline([
    #     ('vectorizer', TfidfVectorizer( ngram_range=(2, 2),tokenizer=stemming_tokenizer,
    #                                    stop_words=stopwords.words('english') + list(string.punctuation))),
    #     ('classifier', MultinomialNB(alpha=0.05))
    # ])

    bigram_clf.fit(train_X, train_y)
    print("Printing accuracy")
    print(bigram_clf.score(test_X, test_y))

    # feature_matrix = get_tweets_feature_matrix(bigram_clf, train_X)
    # print("Printing feature matrix")
    # print(feature_matrix)

    #print("Getting feature names")
    #print(bigram_clf.named_steps['vectorizer'].get_feature_names())

def unigram_bigram(train_X,train_y,test_X,test_y):
#
    unigram_bigram_clf = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(1, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', LinearSVC())
    ])

    unigram_bigram_clf.fit(train_X, train_y)
    print("Printing accuracy")
    print(unigram_bigram_clf.score(test_X, test_y))


    # feature_matrix = get_tweets_feature_matrix(unigram_bigram_clf, train_X)
    # print("Printing feature matrix")
    # print(feature_matrix)

    # test_data = ["I am happy with your airlines","I hate you"]
    # predicted = unigram_bigram_clf.predict(test_data)
    # print(predicted)

    # Check the feature names
    #print("Getting feature names")
    #print (unigram_bigram_clf.named_steps['vectorizer'].get_feature_names())


def main():
    data = read_data()
    # print(data["review"][0])
    sentiment_data = randomize(data)
    train_X,train_y = train_data(sentiment_data)
    test_X,test_y = test_data(sentiment_data)
    for i in range(0, len(train_X)):
        train_X[i] = clean_text(train_X[i])
    # print(train_X[0])
    # print(train_y[0])

    for i in range(0, len(test_X)):
        test_X[i] = clean_text(test_X[i])
    # print(test_X[0])
    # print(test_y[0])
    #print(random_data)
    unigrams(train_X,train_y,test_X,test_y)     #0.8314
    # bigrams(train_X,train_y,test_X,test_y)      #0.82
    # unigram_bigram(train_X,train_y,test_X,test_y)  #0.8472



main()

