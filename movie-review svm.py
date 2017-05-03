import pandas as pd
from nltk import word_tokenize
from nltk.sentiment.util import mark_negation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import nltk
from sklearn.base import TransformerMixin
import random
import re
def read_data():
    data = pd.read_csv("C:/Users/Apoorva/PycharmProjects/TweetsClassification/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

    # 25000 movie reviews
    print (data.shape)  # (25000, 3)
    #print (data["review"][0])  # Check out the review
    #print (data["sentiment"][0])  # Check out the sentiment (0/1)
    return data

def randomize(data):
    sentiment_data = list(zip(data["review"], data["sentiment"]))
    random.shuffle(sentiment_data)
    return sentiment_data

def train_data(sentiment_data):
    #return random_data
    # 80% for training
    train_X, train_y = zip(*sentiment_data[:20000])
    return list(train_X),list(train_y)

def test_data(sentiment_data):
    # Keep 20% for testin
    test_X, test_y = zip(*sentiment_data[20000:])
    return list(test_X),list(test_y)

def clean_text(text):
    text = text.replace("<br />", " ")
    text = text.encode().decode("utf-8")
    # Removing urls from text
    text = re.sub(r'http\S+', '', text,flags=re.MULTILINE)

    return text





from nltk import bigrams

# print(list(bigrams("We are playing with bigrams".split())))

# [('We', 'are'), ('are', 'playing'), ('playing', 'with'), ('with', 'bigrams')]

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

    clf.fit(train_X, train_y)
    print(clf.score(test_X, test_y))
    # print("Accuracy of unigram classifier:"+accuracy)
    # with mark_negation 0.84760000000000002
    # without mark_negation 0.84440000000000004
    print(clf.named_steps['vectorizer'].get_feature_names())

def bigrams(train_X,train_y,test_X,test_y):

    bigram_clf = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(2, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "), )),
        ('classifier', LinearSVC())
    ])

    bigram_clf.fit(train_X, train_y)
    print(bigram_clf.score(test_X, test_y))

    # with mark_negation 0.86760000000000004
    # without mark_negation 0.87119999999999997
    print(bigram_clf.named_steps['vectorizer'].get_feature_names())

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
    print(unigram_bigram_clf.score(test_X, test_y))

    # with mark_negation 0.88219999999999998
    # without mark_negation 0.88300000000000001
    # Check the feature names
    print (unigram_bigram_clf.named_steps['vectorizer'].get_feature_names())

def main():
    data = read_data()
    # print(data["review"][0])
    sentiment_data = randomize(data)
    train_X,train_y = train_data(sentiment_data)
    test_X,test_y = test_data(sentiment_data)
    for i in range(0, len(train_X)):
        train_X[i] = clean_text(train_X[i])
    print(train_X[0])
    for i in range(0, len(test_X)):
        test_X[i] = clean_text(test_X[i])
    print(test_X[0])
    #print(random_data)
    unigrams(train_X,train_y,test_X,test_y)
    # bigrams(train_X,train_y,test_X,test_y)
    # print("Accuracy of unigram bigram classifier: ")
    # unigram_bigram(train_X,train_y,test_X,test_y)

main()